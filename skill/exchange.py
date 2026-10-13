#!/usr/bin/env python3
"""Stable RFS v1 boundary around the unchanged five-agent CLI."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PaperBanana canonical RFS exchange")
    parser.add_argument("command", choices=["describe", "validate-request", "run", "verify-bundle", "vector-handoff"])
    parser.add_argument("--request")
    parser.add_argument("--bundle")
    parser.add_argument("--spec")
    parser.add_argument("--out", help="New directory for verified vector reconstruction references")
    parser.add_argument("--allow-external", action="store_true")
    args = parser.parse_args(argv)
    try:
        from rfs.interop import descriptor, load_spec, ordinary_path, request_argv, validate_request, verify_result
        if args.command == "describe":
            result = descriptor()
        elif args.command == "verify-bundle":
            if not args.spec or not args.bundle:
                raise ValueError("verify-bundle requires --spec and --bundle")
            manifest, _, _ = verify_result(args.bundle, load_spec(args.spec))
            result = {"integrity": "PASS", "candidate_id": manifest["candidate_id"], "scientific_review": "PENDING"}
        elif args.command == "vector-handoff":
            if not args.spec or not args.bundle or not args.out:
                raise ValueError("vector-handoff requires --spec, --bundle and --out")
            from rfs.contract import scientific_fingerprint
            from rfs.interop import digest, json_bytes, write_new_folder
            spec = load_spec(args.spec)
            if spec["kind"] != "diagram":
                raise ValueError("TikZ handoff supports canonical diagram specs only; measured plots stay source-based")
            manifest, bound_spec, payloads = verify_result(args.bundle, spec)
            receipt = {
                "format": "paperbanana-vector-handoff/1.0",
                "status": "REFERENCE_VERIFIED_NOT_VECTOR_RENDERED",
                "candidate_id": manifest["candidate_id"],
                "scientific_fingerprint": scientific_fingerprint(bound_spec),
                "source_image_sha256": digest(payloads["image"]),
                "design_sha256": digest(payloads["design"]),
                "science_review": "PENDING",
                "visual_reconstruction": "PENDING",
                "notes": "RFS --engine tikz renders FigureSpec independently. This reference does not guarantee pixel-identical reconstruction or PPTX editability."
            }
            folder = write_new_folder(args.out, {
                "figure-spec.json": json_bytes(bound_spec),
                "reference.png": payloads["image"],
                "design.md": payloads["design"],
                "manifest.json": json_bytes(receipt),
            })
            result = {"status": receipt["status"], "path": str(folder),
                      "candidate_id": receipt["candidate_id"], "scientific_fingerprint": receipt["scientific_fingerprint"],
                      "local_command": f"python /PATH/research-figure/scripts/rfs.py run --spec {shlex.quote(str(folder / 'figure-spec.json'))} --engine tikz --out /NEW/TIKZ-OUTPUT"}
        else:
            if not args.request:
                raise ValueError("--request is required")
            request, _ = validate_request(args.request)
            if args.command == "validate-request":
                result = {"status": "VALID_REQUEST_NOT_EXECUTED", "figure_id": request["figure_id"]}
            else:
                if not args.allow_external:
                    raise ValueError("Explicit --allow-external is required for provider requests and possible fees")
                if (ordinary_path(args.request).parent / "generated").exists():
                    raise ValueError("Existing generation output is protected; prepare a new request directory")
                from skill.run import main as generate
                return generate(request_argv(args.request))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(json.dumps({"status": "ERROR", "reason": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
