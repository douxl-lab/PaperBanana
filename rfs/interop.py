"""RFS Exchange v1 boundary shared by the producer and consumer.

Copyright 2026 douxl-lab. Licensed under the Apache License, Version 2.0.
No model SDK, credential discovery, network call, or generated-code execution.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import stat
from pathlib import Path

from jsonschema import Draft202012Validator

from .contract import (
    SCHEMA_VERSION, SCHEMA_PATH, RFSValidationError, canonical_json,
    scientific_fingerprint, sha256_text, strict_json_input,
    validate_figure_spec, validate_relative_path,
)

MAX_TEXT = 4 * 1024 * 1024
MAX_FILE = 100 * 1024 * 1024
MAX_BUNDLE = 250 * 1024 * 1024
MANIFEST_SCHEMA = Path(__file__).parent / "schemas" / "manifest-v1.schema.json"
REQUEST_FORMAT = "rfs-generation-request/1.0"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ordinary_path(path: str | Path) -> Path:
    p = Path(path).expanduser().absolute()
    if p.is_symlink() or any(parent.is_symlink() for parent in p.parents):
        raise RFSValidationError("Symlink traversal is not allowed at the exchange boundary")
    return p


def read_bytes(path: str | Path, limit: int = MAX_TEXT) -> bytes:
    p = ordinary_path(path)
    if not p.exists() or not stat.S_ISREG(p.stat().st_mode) or p.stat().st_size > limit:
        raise RFSValidationError("Exchange input must be an existing bounded regular file")
    with p.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise RFSValidationError("Exchange input exceeds the size limit")
    return data


def read_object(path: str | Path) -> dict:
    try:
        result = strict_json_input(read_bytes(path).decode("utf-8"))
    except UnicodeError as exc:
        raise RFSValidationError("Exchange JSON must be UTF-8") from exc
    if not isinstance(result, dict):
        raise RFSValidationError("Exchange JSON root must be an object")
    return result


def load_spec(path: str | Path) -> dict:
    return validate_figure_spec(read_object(path))


def descriptor() -> dict:
    return {
        "protocol": "paperbanana-rfs-exchange", "schema_version": SCHEMA_VERSION,
        "figure_schema_sha256": sha256_text(canonical_json(read_object(SCHEMA_PATH))),
        "manifest_schema_sha256": sha256_text(canonical_json(read_object(MANIFEST_SCHEMA))),
        "validator_sha256": digest(read_bytes(Path(__file__).parent / "contract.py")),
        "request_format": REQUEST_FORMAT, "network_calls": False,
    }


def assert_compatible(other: dict) -> None:
    if not isinstance(other, dict):
        raise RFSValidationError("Protocol descriptor must be an object")
    ours = descriptor()
    if any(other.get(key) != ours[key] for key in (
        "protocol", "schema_version", "figure_schema_sha256",
        "manifest_schema_sha256", "validator_sha256", "request_format",
    )):
        raise RFSValidationError("Producer and consumer protocol differ; update the paired installations")


def write_new_folder(path: str | Path, payloads: dict[str, bytes]) -> Path:
    """Never overwrite an accepted reference, prior candidate or arbitrary file."""
    target = ordinary_path(path)
    for rel, raw in payloads.items():
        validate_relative_path(rel)
        if rel != Path(rel).as_posix() or "//" in rel or not isinstance(raw, bytes):
            raise RFSValidationError("Invalid exchange output entry")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    try:
        for rel, raw in payloads.items():
            p = target / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("xb") as stream:
                stream.write(raw)
    except BaseException:
        shutil.rmtree(target)
        raise
    return target


def json_bytes(value: dict) -> bytes:
    canonical_json(value)
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def source_text(spec: dict) -> str:
    # No factual summarization. Preserve one source verbatim; enumerate multiple
    # supplied sources without silently replacing or merging their evidence.
    sources = spec["sources"]
    return sources[0]["content"] if len(sources) == 1 else canonical_json(sources)


def validate_options(options: dict) -> dict:
    allowed = {"caption", "candidates", "max_concurrent", "max_critic_rounds",
               "aspect_ratio", "image_model", "main_model"}
    if not isinstance(options, dict) or set(options) - allowed:
        raise RFSValidationError("Unknown generation option; credentials and executable paths are not request fields")
    out = {"caption": "Scientific architecture overview", "candidates": 2,
           "max_concurrent": 2, "max_critic_rounds": 3, "aspect_ratio": "21:9", **options}
    if not isinstance(out["caption"], str) or not out["caption"].strip() or len(out["caption"]) > 2000:
        raise RFSValidationError("A bounded nonempty caption is required")
    for key, lo, hi in (("candidates", 1, 8), ("max_concurrent", 1, 8), ("max_critic_rounds", 0, 8)):
        if type(out[key]) is not int or not lo <= out[key] <= hi:
            raise RFSValidationError("Invalid generation limit: " + key)
    if out["aspect_ratio"] not in {"21:9", "16:9", "3:2", "1:1", "9:16"}:
        raise RFSValidationError("Aspect ratio is not supported by the paired CLI")
    for key in ("image_model", "main_model"):
        if key in out and (not isinstance(out[key], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,128}", out[key])):
            raise RFSValidationError("Invalid model identifier")
    return out


def prepare_request(spec_path: str | Path, output: str | Path, **options) -> Path:
    spec = load_spec(spec_path)
    if spec["kind"] != "diagram" or spec["datasets"]:
        raise RFSValidationError("Generative request requires diagram; measured plots use deterministic route C, hybrid figures must be split")
    checked = validate_options(options)
    spec_raw, source_raw = json_bytes(spec), source_text(spec).encode("utf-8")
    request = {
        "format": REQUEST_FORMAT, "protocol": descriptor(), "figure_id": spec["figure_id"],
        "scientific_fingerprint": scientific_fingerprint(spec), "options": checked,
        "files": {
            "figure_spec": {"path": "figure-spec.json", "sha256": digest(spec_raw)},
            "source": {"path": "source.txt", "sha256": digest(source_raw)},
        },
    }
    target = write_new_folder(output, {"figure-spec.json": spec_raw, "source.txt": source_raw,
                                      "request.json": json_bytes(request)})
    return target / "request.json"


def validate_request(path: str | Path) -> tuple[dict, dict]:
    request = read_object(path)
    required = {"format", "protocol", "figure_id", "scientific_fingerprint", "options", "files"}
    if set(request) != required or request["format"] != REQUEST_FORMAT:
        raise RFSValidationError("Invalid generation request envelope")
    assert_compatible(request["protocol"])
    validate_options(request["options"])
    if not isinstance(request["files"], dict) or set(request["files"]) != {"figure_spec", "source"}:
        raise RFSValidationError("Request must contain exact source/spec bindings")
    root = ordinary_path(path).parent
    for role, name in (("figure_spec", "figure-spec.json"), ("source", "source.txt")):
        entry = request["files"][role]
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"} or entry["path"] != name:
            raise RFSValidationError("Unexpected request source path")
        if digest(read_bytes(root / name)) != entry["sha256"]:
            raise RFSValidationError("Request source/spec checksum mismatch")
    spec = load_spec(root / "figure-spec.json")
    if spec["kind"] != "diagram" or spec["datasets"] or spec["figure_id"] != request["figure_id"]:
        raise RFSValidationError("Request figure identity/kind mismatch")
    if scientific_fingerprint(spec) != request["scientific_fingerprint"]:
        raise RFSValidationError("Request scientific fingerprint mismatch")
    if read_bytes(root / "source.txt") != source_text(spec).encode("utf-8"):
        raise RFSValidationError("Request source text does not match supplied scientific sources")
    return request, spec


def request_argv(path: str | Path) -> list[str]:
    request, _ = validate_request(path)
    root = ordinary_path(path).parent
    opts = validate_options(request["options"])
    args = ["--content-file", str(root / "source.txt"), "--rfs-file", str(root / "figure-spec.json"),
            "--render-mode", "rfs", "--task", "diagram", "--caption", opts["caption"],
            "--retrieval-setting", "none", "--dataset-policy", "never", "--exp-mode", "demo_full",
            "--output", str(root / "generated" / "figure.png"),
            "--rfs-output-dir", str(root / "generated" / "bundles")]
    for field, flag in (("candidates", "--num-candidates"), ("max_concurrent", "--max-concurrent"),
                        ("max_critic_rounds", "--max-critic-rounds"), ("aspect_ratio", "--aspect-ratio"),
                        ("image_model", "--image-gen-model-name"), ("main_model", "--main-model-name")):
        if field in opts:
            args.extend([flag, str(opts[field])])
    return args


def verify_result(folder: str | Path, expected_spec: dict | None = None,
                  *, require_success: bool = True) -> tuple[dict, dict | None, dict[str, bytes]]:
    root = ordinary_path(folder)
    manifest = read_object(root / "manifest.json")
    errors = list(Draft202012Validator(read_object(MANIFEST_SCHEMA)).iter_errors(manifest))
    if errors:
        raise RFSValidationError("Malformed result manifest")
    if require_success and (manifest["status"] != "succeeded" or manifest["review_status"] in {"failed", "invalid"}):
        raise RFSValidationError("Failed/partial result is not an accepted reconstruction source")
    entries = manifest["files"]
    if len(entries) > 32 or sum(e["bytes"] for e in entries.values()) > MAX_BUNDLE:
        raise RFSValidationError("Bundle exceeds size or file-count limits")
    names, payloads = set(), {}
    for role, entry in entries.items():
        rel = validate_relative_path(entry["path"])
        if rel in names or rel == "manifest.json" or "//" in rel or Path(rel).as_posix() != rel:
            raise RFSValidationError("Duplicate, reserved or ambiguous bundle path")
        names.add(rel)
        raw = read_bytes(root / rel, MAX_FILE)
        if len(raw) != entry["bytes"] or digest(raw) != entry["sha256"]:
            raise RFSValidationError("Bundle artifact checksum mismatch")
        payloads[role] = raw
        if entry["mime"].startswith("image/"):
            from PIL import Image
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    if im.width * im.height > 64_000_000 or Image.MIME.get(im.format) != entry["mime"]:
                        raise RFSValidationError("Image MIME/pixel-limit mismatch")
                    im.verify()
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                raise RFSValidationError("Invalid result image") from exc
    spec = None
    if "figure_spec" in payloads:
        spec = validate_figure_spec(strict_json_input(payloads["figure_spec"].decode("utf-8")))
        if scientific_fingerprint(spec) != manifest["scientific_fingerprint"]:
            raise RFSValidationError("Result scientific fingerprint mismatch")
        if manifest["provenance"]["scientific_verification"] != spec["provenance"]["scientific_verification"]:
            raise RFSValidationError("Result verification claim disagrees with the FigureSpec")
        if expected_spec is not None and scientific_fingerprint(spec) != scientific_fingerprint(expected_spec):
            raise RFSValidationError("Result changed the caller's locked scientific contract")
    if "input_figure_spec" in payloads:
        initial = validate_figure_spec(strict_json_input(payloads["input_figure_spec"].decode("utf-8")))
        if spec is None or scientific_fingerprint(initial) != scientific_fingerprint(spec):
            raise RFSValidationError("Input and output scientific contracts disagree")
        if expected_spec is not None and canonical_json(initial) != canonical_json(expected_spec):
            raise RFSValidationError("Input FigureSpec is not the caller's supplied snapshot")
    if manifest["status"] == "succeeded":
        if manifest["review_status"] in {"failed", "invalid"}:
            raise RFSValidationError("Successful bundle cannot contain a failed review")
        if spec is None or not entries["image"]["mime"].startswith("image/"):
            raise RFSValidationError("Successful bundle needs a scientific contract and real image")
        for role in ("prompt", "design"):
            if entries[role]["mime"] not in {"text/plain", "text/markdown"} or not payloads[role].decode("utf-8").strip():
                raise RFSValidationError("Successful bundle needs nonempty generation prompt and written design")
        if manifest["error"] is not None or not manifest["description_key"]:
            raise RFSValidationError("Successful bundle has an error or lacks a selected stage")
    if expected_spec is not None and spec is None:
        raise RFSValidationError("Expected scientific contract is missing from result")
    metadata = manifest["provenance"]["generation"].get("compiler_metadata", {})
    if not isinstance(metadata, dict):
        raise RFSValidationError("Compiler metadata must be an object")
    recorded_hash = metadata.get("prompt_sha256")
    if recorded_hash is not None and recorded_hash != digest(payloads.get("prompt", b"")):
        raise RFSValidationError("Submitted prompt hash disagrees with the selected result")
    return manifest, spec, payloads
