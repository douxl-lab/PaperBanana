#!/usr/bin/env python3
"""Local native-host adapter. No network, providers, credentials or model execution.
Uses the repository RFS contract. Receipts are host attestations, not
platform-signed proof. A host must execute each requested stage explicitly.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rfs.interop import (
    prepare_request,
    validate_request,
    read_bytes,
    read_object,
    json_bytes,
    ordinary_path,
)
from rfs.contract import scientific_fingerprint, validate_repair

if not __debug__:
    raise RuntimeError("Run without -O: contract guards must remain enabled")
STAGES = ["Retriever", "Planner", "Stylist", "Visualizer", "Critic"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write(p, v):
    with p.open("xb") as f:
        f.write(json_bytes(v))


def validate_payload(stage, payload):
    assert payload.get("status") == "succeeded" and not payload.get("error"), (
        "stage has not succeeded"
    )
    required = {
        "Retriever": ["observations", "source_paths"],
        "Planner": ["design_description"],
        "Stylist": ["prompt", "design_description"],
        "Critic": [
            "image_sha256",
            "verdict",
            "observed_scientific_fidelity",
            "aesthetics",
            "issues",
        ],
    }[stage]
    assert all(k in payload for k in required), "missing stage fields"
    assert all(payload[k] for k in required if k != "issues"), (
        "empty substantive output"
    )
    if stage in ["Planner", "Stylist"]:
        assert (
            isinstance(payload["design_description"], str)
            and payload["design_description"].strip()
        ), "design_description must be nonempty text"
    if stage == "Stylist":
        assert isinstance(payload["prompt"], str) and payload["prompt"].strip(), (
            "prompt must be nonempty text"
        )
        if "referenced_image_paths" in payload:
            assert isinstance(payload["referenced_image_paths"], list) and all(
                isinstance(x, str) and x for x in payload["referenced_image_paths"]
            ), "reference paths must be a list of nonempty strings"
    if stage == "Retriever":
        assert isinstance(payload["source_paths"], list) and all(
            isinstance(x, str) and x for x in payload["source_paths"]
        ), "source_paths must be a list of paths"
    if stage == "Critic":
        assert payload["verdict"] in ["accept", "revise"]
        assert isinstance(payload["issues"], list), "issues must be a list"


def validate_style_references(job, payload):
    refs = payload.get("referenced_image_paths", [])
    if refs:
        assert (job / "context.json").is_file(), (
            "style references require bound context"
        )
        allowed = {e["path"] for e in read_object(job / "context.json")["files"]}
        assert set(refs) <= allowed, "unbound style image reference"


def visual_arguments(job):
    payload = read_object(job / "Stylist.output.json")
    args = {"prompt": payload["prompt"], "transparent_background": False}
    if (job / "repair.json").exists():
        repair = read_object(job / "repair.json")
        _, spec = validate_request(job / "request.json")
        validate_repair(repair["edit"], spec)
        assert (
            sha(read_bytes(job / "repair_reference.png", 100 * 1024 * 1024))
            == repair["image_sha256"]
        )
        if repair.get("prompt_mode") == "repair_only":
            args["prompt"] = (
                "Edit the supplied scientific diagram in place. The full scientific contract remains locked. "
            )
        args["prompt"] += (
            "\n\nBOUNDED REFERENCE EDIT: "
            + json.dumps(repair["edit"], ensure_ascii=False)
            + "\nPreserve the existing image except these specified corrections. Do not regenerate an unrelated layout."
        )
        args["referenced_image_paths"] = [str(job / "repair_reference.png")]
    elif payload.get("referenced_image_paths"):
        refs = payload["referenced_image_paths"]
        assert isinstance(refs, list) and all(isinstance(x, str) for x in refs)
        allowed = {e["path"] for e in read_object(job / "context.json")["files"]}
        assert set(refs) <= allowed, "unbound style image reference"
        args["referenced_image_paths"] = refs
    return args


def repair_job(source, out, edit_path, repair_only=False):
    _, spec, rs = state(source)
    assert len(rs) == 5 and rs[-1]["verdict"] == "revise", (
        "source must be reviewed revise"
    )
    edit = validate_repair(read_object(edit_path), spec)
    prepare(source / "figure-spec.json", out)
    for name in ["context.json"] + [
        x + suffix for x in STAGES[:3] for suffix in [".json", ".output.json"]
    ]:
        if (source / name).exists():
            shutil.copyfile(source / name, out / name)
    shutil.copyfile(source / "image.png", out / "repair_reference.png")
    write(
        out / "repair.json",
        {
            "format": "native-host-bounded-repair/1",
            "prompt_mode": "repair_only" if repair_only else "append_original",
            "inherited_from": str(source),
            "critic_receipt_sha256": sha(read_bytes(source / "Critic.json")),
            "image_sha256": sha(read_bytes(source / "image.png", 100 * 1024 * 1024)),
            "edit": edit,
        },
    )
    return next_task(out)


def state(job):
    req, spec = validate_request(job / "request.json")
    records = []
    previous = None
    context_hash = None
    if (job / "context.json").exists():
        context_hash = sha(read_bytes(job / "context.json"))
        for e in read_object(job / "context.json")["files"]:
            assert sha(read_bytes(e["path"], 100 * 1024 * 1024)) == e["sha256"], (
                "reference context changed"
            )
    for stage in STAGES:
        p = job / (stage + ".json")
        if not p.exists():
            assert not any(
                (job / (later + ".json")).exists()
                for later in STAGES[STAGES.index(stage) + 1 :]
            ), "out-of-order receipts"
            break
        rec = read_object(p)
        assert rec["stage"] == stage and rec[
            "scientific_fingerprint"
        ] == scientific_fingerprint(spec)
        assert isinstance(rec.get("host_task_id"), str) and rec["host_task_id"].strip()
        assert rec.get("previous_receipt_sha256") == previous, "broken stage chain"
        assert rec.get("context_sha256") == context_hash, "context binding changed"
        expected = (
            {"image.png", "prompt.txt"}
            if stage == "Visualizer"
            else {stage + ".output.json"}
        )
        assert set(rec["artifacts"]) == expected, "incomplete artifact set"
        for name, digest in rec["artifacts"].items():
            assert sha(read_bytes(job / name, 100 * 1024 * 1024)) == digest, (
                "artifact changed: " + name
            )
        if stage == "Visualizer":
            from PIL import Image

            with Image.open(job / "image.png") as im:
                assert im.format == "PNG"
                im.verify()
            prompt = visual_arguments(job)["prompt"].encode()
            assert read_bytes(job / "prompt.txt") == prompt and rec[
                "prompt_sha256"
            ] == sha(prompt), "prompt mismatch"
        else:
            payload = read_object(job / (stage + ".output.json"))
            validate_payload(stage, payload)
            if stage == "Critic":
                assert payload["image_sha256"] == sha(
                    read_bytes(job / "image.png", 100 * 1024 * 1024)
                )
                assert rec["verdict"] == payload["verdict"], "verdict mismatch"
        records.append(rec)
        previous = sha(read_bytes(p))
    return req, spec, records


def prepare(spec, out):
    return str(prepare_request(spec, out, candidates=1, max_concurrent=1))


def next_task(job):
    req, spec, rs = state(job)
    if len(rs) == 5:
        return {
            "status": "ACCEPTED"
            if rs[-1]["verdict"] == "accept"
            else "REVISION_REQUIRED",
            "scientific_truth": "NOT_ESTABLISHED_BY_IMAGE_REVIEW",
        }
    role = STAGES[len(rs)]
    task = {
        "status": "AWAITING_HOST_EXECUTION",
        "stage": role,
        "scientific_fingerprint": scientific_fingerprint(spec),
        "spec_path": str(job / "figure-spec.json"),
        "context_path": str(job / "context.json")
        if (job / "context.json").exists()
        else None,
        "previous_records": [str(job / (r["stage"] + ".json")) for r in rs],
    }
    if role == "Visualizer":
        args = visual_arguments(job)
        task.update(
            tool="image_gen__imagegen",
            tool_arguments=args,
            prompt_sha256=sha(args["prompt"].encode()),
        )
    elif role == "Critic":
        task.update(
            image_path=str(job / "image.png"),
            image_sha256=sha(read_bytes(job / "image.png", 100 * 1024 * 1024)),
            instructions="Open actual image. Assess scientific fidelity and aesthetics separately. Return image_sha256, verdict accept/revise, observed_scientific_fidelity, aesthetics, and issues. Acceptance is visual fidelity, not empirical validation.",
        )
    else:
        task["instructions"] = {
            "Retriever": "Inspect supplied sources and user reference images; record paths and observed style constraints, without altering science.",
            "Planner": "Design topology preserving every scientific node, edge and formula. Record intended composition and empirical raster slots.",
            "Stylist": "Return exact prompt and design_description; use reference style. Do not invent empirical imagery, data or implemented mechanisms. Use placeholders for original rasters to be composited unchanged later.",
        }[role]
    return task


def record(job, stage, artifact, task_id, prompt_hash=None):
    _, spec, rs = state(job)
    assert len(rs) < 5 and STAGES[len(rs)] == stage, "wrong stage or already recorded"
    assert task_id and task_id.strip(), (
        "actual host task/tool ID or explicit tool-managed-not-reported required"
    )
    rec = {
        "stage": stage,
        "host_task_id": task_id,
        "receipt_type": "host_attestation_not_platform_signature",
        "scientific_fingerprint": scientific_fingerprint(spec),
        "artifacts": {},
        "context_sha256": sha(read_bytes(job / "context.json"))
        if (job / "context.json").exists()
        else None,
        "previous_receipt_sha256": sha(
            read_bytes(job / (STAGES[len(rs) - 1] + ".json"))
        )
        if rs
        else None,
    }
    raw = read_bytes(artifact, 100 * 1024 * 1024)
    if stage == "Visualizer":
        from PIL import Image

        with Image.open(artifact) as im:
            assert im.format == "PNG"
            im.verify()
        task = next_task(job)
        assert prompt_hash == task["prompt_sha256"], "submitted prompt mismatch"
        dest = "image.png"
        rec["prompt_sha256"] = prompt_hash
        with (job / "prompt.txt").open("xb") as f:
            f.write(task["tool_arguments"]["prompt"].encode())
        rec["artifacts"]["prompt.txt"] = prompt_hash
    else:
        payload = read_object(artifact)
        validate_payload(stage, payload)
        if stage == "Stylist":
            validate_style_references(job, payload)
        if stage == "Critic":
            assert payload["image_sha256"] == sha(
                read_bytes(job / "image.png", 100 * 1024 * 1024)
            ), "review bound to wrong image"
            assert payload["verdict"] in ["accept", "revise"]
            assert all(
                k in payload
                for k in ["observed_scientific_fidelity", "aesthetics", "issues"]
            )
            rec["verdict"] = payload["verdict"]
        dest = stage + ".output.json"
    with (job / dest).open("xb") as f:
        f.write(raw)
    rec["artifacts"][dest] = sha(raw)
    write(job / (stage + ".json"), rec)
    return next_task(job)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["prepare", "next", "record", "verify", "repair"])
    p.add_argument("--job", required=True)
    p.add_argument("--source-job")
    p.add_argument("--edit")
    p.add_argument("--repair-only", action="store_true")
    p.add_argument("--spec")
    p.add_argument("--stage", choices=STAGES)
    p.add_argument("--artifact")
    p.add_argument("--task-id")
    p.add_argument("--prompt-sha256")
    a = p.parse_args()
    required = {
        "prepare": ["spec"],
        "record": ["stage", "artifact", "task_id"],
        "repair": ["source_job", "edit"],
    }.get(a.command, [])
    for field in required:
        if not getattr(a, field):
            p.error("--" + field.replace("_", "-") + " is required for " + a.command)
    if a.command == "record" and a.stage == "Visualizer" and not a.prompt_sha256:
        p.error("--prompt-sha256 is required for Visualizer")
    job = ordinary_path(a.job)
    if a.command == "prepare":
        out = {"request": prepare(a.spec, job), "status": "AWAITING_HOST_EXECUTION"}
    elif a.command == "repair":
        out = repair_job(ordinary_path(a.source_job), job, a.edit, a.repair_only)
    elif a.command == "record":
        out = record(job, a.stage, a.artifact, a.task_id, a.prompt_sha256)
    else:
        out = next_task(job)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
