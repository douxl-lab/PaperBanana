#!/usr/bin/env python3
"""Stable RFS v1 boundary around the unchanged five-agent CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PaperBanana canonical RFS exchange")
    parser.add_argument("command", choices=["describe", "validate-request", "run", "verify-bundle"])
    parser.add_argument("--request")
    parser.add_argument("--bundle")
    parser.add_argument("--spec")
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
