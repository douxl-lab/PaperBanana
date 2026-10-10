"""Atomic checkpoint records and deterministic candidate identities."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

CHECKPOINT_VERSION = 2
RUNTIME_VERSION = "rfs-runtime-1"


def stable_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def generation_binding(config) -> dict[str, Any]:
    fields = ("task_name", "exp_mode", "retrieval_setting", "planner_metaphor", "max_critic_rounds",
              "temperature", "main_model_name", "image_gen_model_name", "main_model_provider",
              "image_gen_provider", "evaluation_model_name", "render_mode", "allow_generated_plot_code",
              "plot_timeout_seconds", "offline", "dataset_policy", "model_profiles")
    binding = {name: getattr(config, name, None) for name in fields}
    binding["runtime_version"] = RUNTIME_VERSION
    root = Path(config.work_dir)
    dependencies = list((root / "agents").glob("*.py")) + list((root / "rfs").glob("*.py"))
    dependencies += [root / "utils" / filename for filename in ("generation_utils.py", "model_registry.py", "plot_execution.py", "paperviz_processor.py", "config.py", "runtime.py", "dataset.py", "image_utils.py", "legacy_generation_options.py", "legacy_ui_results.py", "eval_toolkits.py")]
    dependencies += list((root / "rfs" / "schemas").glob("*.json"))
    dependencies += [root / "configs" / "model_registry.yaml"]
    data_root = root / "data" / "PaperBananaBench" / config.task_name
    dependencies += [path for path in data_root.rglob("*") if path.is_file() and path.suffix.lower() in (".json", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg")]
    guide = getattr(config, "style_guide_path", None)
    dependencies += [Path(guide)] if guide else list((root / "style_guides").glob(f"*{config.task_name}*.md"))
    binding["dependencies"] = {str(path.relative_to(root)) if path.is_relative_to(root) else path.name:
        hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(set(dependencies)) if path.is_file()}
    return binding


def prepare_candidate(data: dict, index: int, binding: dict) -> dict:
    result = dict(data)
    supplied = result.get("candidate_id")
    if type(supplied) is int:
        supplied = str(supplied)
    if supplied is not None and (not isinstance(supplied, str) or not supplied.strip()):
        raise ValueError("candidate_id must be a nonempty string")
    identity_input = {key: value for key, value in result.items() if key not in ("candidate_index", "candidate_id") and not key.startswith("_runtime_")}
    fingerprint = stable_digest({"input": identity_input, "configuration": binding})
    result["candidate_id"] = supplied or f"candidate-{index:04d}-{fingerprint[:12]}"
    result["candidate_index"] = index
    result["_runtime_fingerprint"] = fingerprint
    return result


def valid_success_result(result: dict) -> bool:
    """A successful checkpoint must still contain real, internally bound outputs."""
    import base64
    from io import BytesIO
    from PIL import Image
    from utils.legacy_ui_results import resolve_final_output
    field = resolve_final_output(result, task_name=result.get("task_name")).image_key
    if field is None:
        return isinstance(result.get("top10_references"), list) and result.get("exp_mode") == "dev_retriever"
    try:
        encoded = result[field]
        with Image.open(BytesIO(base64.b64decode(encoded.split(",", 1)[-1], validate=True))) as image:
            image.verify()
        if result.get("render_mode") == "rfs":
            from rfs import validate_figure_spec
            from rfs.contract import canonical_json, sha256_text
            selected = result.get("eval_image_field", "").removesuffix("_base64_jpg")
            record = result.get("rfs_artifacts", {}).get(selected)
            if not isinstance(record, dict) or record.get("candidate_id") != result["candidate_id"]:
                return False
            if any(not isinstance(record.get(key), str) or not record[key].strip() for key in ("prompt", "provider", "model", "stage", "design_description")):
                return False
            spec = validate_figure_spec(record.get("figure_spec"))
            if record.get("figure_spec_sha256") != sha256_text(canonical_json(spec)):
                return False
            with Image.open(BytesIO(base64.b64decode(record["image_base64"], validate=True))) as image:
                image.verify()
                if Image.MIME.get(image.format) != record.get("image_mime"):
                    return False
        return True
    except (OSError, ValueError, TypeError, KeyError, AttributeError, Image.DecompressionBombError):
        return False


class CandidateCheckpoints:
    def __init__(self, directory: Path | None):
        self.directory = Path(directory) if directory is not None else None

    def path_for(self, candidate_id: str) -> Path:
        # IDs are data, never paths. This also prevents traversal in caller IDs.
        return self.directory / f"{hashlib.sha256(candidate_id.encode()).hexdigest()}.json"

    def load(self, candidate: dict) -> dict | None:
        if self.directory is None:
            return None
        try:
            path = self.path_for(candidate["candidate_id"])
            if path.is_symlink():
                return None
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("version") != CHECKPOINT_VERSION or record.get("fingerprint") != candidate["_runtime_fingerprint"]:
                return None
            result = record["result"]
            if not isinstance(result, dict) or result.get("candidate_id") != candidate["candidate_id"] or result.get("status") != "succeeded":
                return None
            if record.get("result_sha256") != stable_digest(result) or not valid_success_result(result):
                return None
            result["candidate_index"] = candidate["candidate_index"]
            result["resumed"] = True
            return result
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, candidate: dict, result: dict) -> None:
        if self.directory is not None:
            atomic_write_json(self.path_for(candidate["candidate_id"]), {
                "version": CHECKPOINT_VERSION, "fingerprint": candidate["_runtime_fingerprint"], "result": result, "result_sha256": stable_digest(result)})


def write_or_reuse_rfs_bundle(result: dict, output_dir: Path) -> dict:
    """Reuse only a verified bundle for a matching successful resumed candidate."""
    import base64
    from rfs import export_bundle, verify_bundle
    output_dir = Path(output_dir)
    if not result.get("resumed") or not output_dir.exists():
        return export_bundle(result, output_dir, candidate_id=result["candidate_id"])
    manifest = verify_bundle(output_dir)
    if manifest["candidate_id"] != result["candidate_id"] or manifest["status"] != "succeeded":
        raise ValueError("Existing RFS bundle does not match this successful candidate")
    key = result.get("eval_image_field", "").removesuffix("_base64_jpg")
    record = result.get("rfs_artifacts", {}).get(key, {})
    prompt = record.get("prompt")
    image = record.get("image_base64")
    if not isinstance(prompt, str) or not isinstance(image, str):
        raise ValueError("Resumed candidate lacks its original prompt or image")
    files = manifest["files"]
    image_file = files.get("original_image", files.get("image"))
    if not image_file or hashlib.sha256(base64.b64decode(image, validate=True)).hexdigest() != image_file["sha256"]:
        raise ValueError("Existing RFS image differs from the resumed generation")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != files.get("prompt", {}).get("sha256"):
        raise ValueError("Existing RFS prompt differs from the resumed generation")
    spec = record.get("figure_spec", result.get("figure_spec"))
    exported_spec = json.loads((output_dir / files["figure_spec"]["path"]).read_text(encoding="utf-8"))
    if stable_digest(spec) != stable_digest(exported_spec):
        raise ValueError("Existing RFS scientific contract differs from the resumed candidate")
    return manifest


def candidate_artifact_directory(root: Path, candidate_id: str) -> Path:
    """Map a data identity to an ordinary single path component without traversal."""
    import re
    root = Path(root)
    identifier = str(candidate_id)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identifier):
        component = identifier
    else:
        component = "candidate-" + hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:24]
    directory = root / component
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError("Candidate artifact directory escapes the configured root")
    return directory


def strict_scientific_json(text: str) -> Any:
    """Parse scientific JSON without repairs, duplicate keys, or NaN/Infinity."""
    def object_pairs(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Scientific JSON cannot contain duplicate object keys")
            value[key] = item
        return value
    def invalid_constant(value):
        raise ValueError("Scientific JSON cannot contain non-finite numeric constants")
    try:
        return json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid_constant)
    except json.JSONDecodeError as exc:
        raise ValueError("Scientific input requires valid JSON; automatic repairs are disabled") from exc


def normalize_scientific_content(content: Any, task_name: str, render_mode: str) -> Any:
    """Keep legacy input parsing, while never repairing RFS quantitative data."""
    from utils.legacy_generation_options import normalize_legacy_input_content
    if render_mode != "rfs" or task_name != "plot":
        return normalize_legacy_input_content(content, task_name)
    parsed = strict_scientific_json(content) if isinstance(content, str) else content
    if not isinstance(parsed, (dict, list)) or not parsed:
        raise ValueError("RFS plot input requires a nonempty JSON object or array of real source data")
    # Validate pre-parsed objects as well: no non-JSON or non-finite values can
    # bypass the strict text parser via a benchmark input object.
    stable_digest(parsed)
    return parsed


def check_figure_task(spec: dict, task_name: str) -> None:
    if not isinstance(spec, dict) or spec.get("kind") not in (task_name, "hybrid"):
        raise ValueError("FigureSpec kind does not match the selected task")
