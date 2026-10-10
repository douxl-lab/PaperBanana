"""Lazy benchmark preparation. No import or no-retrieval path uses the network."""
from __future__ import annotations

from pathlib import Path
from typing import Callable


def ensure_dataset(task_name: str, *, work_dir: Path, retrieval_setting: str = "auto",
                   policy: str = "auto", offline: bool = False,
                   downloader: Callable | None = None) -> bool:
    if task_name not in ("diagram", "plot"):
        raise ValueError("Dataset task must be diagram or plot")
    if policy not in ("auto", "never", "download"):
        raise ValueError("Unknown dataset download policy")
    if retrieval_setting == "none":
        return False
    root = Path(work_dir) / "data" / "PaperBananaBench"
    task_root = root / task_name
    reference = task_root / ("agent_selected_12.json" if retrieval_setting == "manual" else "ref.json")
    if reference.is_file() and (task_root / "images").is_dir():
        return True
    if offline or policy == "never":
        return False
    if downloader is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is needed to download references; use --offline to skip") from exc
        downloader = snapshot_download
    try:
        downloader("dwzhu/PaperBananaBench", repo_type="dataset", allow_patterns=[f"{task_name}/*"], local_dir=str(root))
    except Exception as exc:
        # Do not include remote exception strings: SDKs may contain credentials.
        if policy == "download":
            raise RuntimeError(f"Reference dataset download failed ({type(exc).__name__})") from exc
        return False
    return reference.is_file() and (task_root / "images").is_dir()
