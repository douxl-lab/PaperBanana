#!/usr/bin/env python3
"""Block tracked private developer state and high-confidence credential patterns.

This is a publication guard, not a proof that all content is safe to disclose.
Only public runtime docs/README/legal files are permitted as Markdown.
"""
import re
import subprocess
import sys
from pathlib import Path

PRIVATE = {"docs", "internal", "private", "notes", "reports", "reviews", "plans", "drafts", "work",
           "data", "datasets", "results", "output", "outputs", "checkpoints", "logs", "_peer",
           "jobs", "native_runtime", "native-jobs", "artifacts", "deliverables", "attachments",
           "manuscripts", "research", ".vscode", ".idea", ".cursor", ".claude", ".codex", ".agents", ".venv", "__pycache__"}
ROOT_DOCS = {"README.md", "SECURITY.md", "CONTRIBUTING.md", "code-of-conduct.md"}
SECRET = [re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
          re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b"),
          re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{24,})\b"),
          re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")]


def check_path(name):
    path = Path(name)
    if any(p.lower() in PRIVATE for p in path.parts):
        return False
    if path.name.startswith(".env") or path.suffix.lower() in {".key", ".pem", ".p12", ".pfx"}:
        return False
    if path.name in {"context.json", "repair.json", "request.json", "prompt.txt"} or path.name.endswith((".receipt.json", ".output.json")):
        return False
    if name == "configs/model_config.yaml" or re.search(r"(?i)(credentials|service[-_]account)", path.name):
        return False
    if path.suffix.lower() == ".md":
        return (name in ROOT_DOCS or name == "rfs/README.md" or name == "skill/SKILL.md"
                or name.startswith("style_guides/") or name.startswith("skills/paperbanana/"))
    return True


def main():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=False)
    if result.returncode:
        print("Publication check needs a Git checkout", file=sys.stderr)
        return 2
    paths = [p.decode() for p in result.stdout.split(b"\0") if p]
    failures = []
    for name in paths:
        path = root / name
        if not check_path(name) or path.is_symlink():
            failures.append(name + ": private/unapproved tracked path")
            continue
        raw = path.read_bytes()
        if b"\0" in raw or len(raw) > 2 * 1024 * 1024:
            continue  # Retained upstream binary examples: inspect separately.
        text = raw.decode("utf-8", errors="replace")
        if any(pattern.search(text) for pattern in SECRET):
            failures.append(name + ": credential signature (value redacted)")
    for failure in failures:
        print("BLOCK: " + failure, file=sys.stderr)
    if not failures:
        print(f"Publication path/secret preflight PASS: {len(paths)} tracked files; binary semantics not scanned")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
