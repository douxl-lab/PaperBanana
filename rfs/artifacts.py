"""Lossless exchange bundle export; never overwrites an earlier result."""
from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from .contract import (RFSValidationError, SCHEMA_VERSION, canonical_json, scientific_fingerprint,
                       sha256_text, validate_figure_spec, validate_relative_path)

MANIFEST_SCHEMA_PATH = Path(__file__).parent / "schemas" / "manifest-v1.schema.json"


def _safe_metadata(value: Any) -> Any:
    """Do not export credentials from provider configuration or exception text."""
    if isinstance(value, dict):
        return {str(k): _safe_metadata(v) for k, v in value.items()
                if not re.search(r"api.?key|access.?token|authorization|credential|password|secret", str(k), re.I)}
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[redacted]", value)
        value = re.sub(r"\bAIza[A-Za-z0-9_-]{20,}\b", "[redacted]", value)
        value = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{8,}", "Bearer [redacted]", value)
        return re.sub(r"(?i)\b(api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*[^\s,;]+", r"\1=[redacted]", value)
    if value is None or type(value) in (bool, int, float):
        return value
    return str(value)


def _selected_record(data: dict) -> tuple[str | None, dict | None]:
    artifacts = data.get("rfs_artifacts", {})
    if not isinstance(artifacts, dict):
        raise RFSValidationError("rfs_artifacts must be a dictionary keyed by legacy description key")
    field = data.get("eval_image_field")
    if isinstance(field, str):
        desc_key = re.sub(r"_base64_(?:jpg|png)$", "", field)
        return desc_key, artifacts.get(desc_key)
    return None, None


def _image_files(record: dict) -> dict[str, tuple[str, bytes, str]]:
    encoded = record.get("image_base64")
    if not isinstance(encoded, str) or not encoded:
        raise RFSValidationError("Selected image record is missing its original base64 image")
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            image_format = image.format
            mime = Image.MIME.get(image_format)
            if mime not in ("image/png", "image/jpeg", "image/webp"):
                raise RFSValidationError("Exchange supports PNG, JPEG and WebP image bytes")
            supplied_mime = record.get("image_mime")
            if supplied_mime is not None and supplied_mime != mime:
                raise RFSValidationError("Image bytes and recorded MIME disagree")
            if mime == "image/png":
                return {"image": ("image.png", raw, mime)}
            output = io.BytesIO()
            image.save(output, format="PNG")
    except (ValueError, OSError, Image.DecompressionBombError) as exc:
        raise RFSValidationError("Selected image is not valid decodable base64 image bytes") from exc
    suffix = "jpg" if mime == "image/jpeg" else "webp"
    return {"image": ("image.png", output.getvalue(), "image/png"),
            "original_image": (f"image.original.{suffix}", raw, mime)}


def _design_document(spec: dict, description: str, *, status: str) -> str:
    lines = ["# Figure design and reconstruction handoff", "", description, "", "## Scientific boundaries",
             f"- Candidate status: {status}",
             f"- Scientific verification: {spec['provenance']['scientific_verification']}",
             "- Source quotes and checksums establish traceability; semantic scientific claims still require review.",
             "- External RFS consumer schema acceptance has not been verified.",
             "- The image is raster artwork. Native editable PPTX reconstruction has not been produced or tested.",
             "", "## Reconstruction instructions",
             "Use image.png as the appearance reference, figure_spec.json as the immutable scientific structure, "
             "and this document for layout/style decisions. Recreate independent editable nodes, typed directed "
             "edges, groups and text from the contract. Preserve every exact label, formula and data row. "
             "Do not infer missing scientific facts from image pixels. prompt.txt is the exact submitted generation text; "
             "prompt.system.txt, when present, is the separately submitted system instruction. "
             "For quantitative figures, plot_code.py and the complete original datasets are the reproducible handoff.",
             "", "## Layout and composition", spec["design"]["layout"], spec["design"]["composition"],
             "", "## Visual and text design", "Palette: " + ", ".join(spec["design"]["palette"]), spec["design"]["typography"],
             "", "## Locked scientific structure", "```json", canonical_json({key: spec[key] for key in ("nodes", "edges", "groups", "labels", "datasets", "constraints")}), "```", ""]
    return "\n".join(lines)


def export_bundle(data: dict, output_dir: str | Path, *, candidate_id: str | int | None = None) -> dict:
    """Export final image, actual prompt, design and FigureSpec, or a failure manifest.

    Output must be a new directory. No model-provided path is ever dereferenced.
    A failed review can retain the last valid image without relabeling it a success.
    """
    if not isinstance(data, dict):
        raise RFSValidationError("Candidate result must be an object")
    identifier = candidate_id if candidate_id is not None else data.get("candidate_id")
    if type(identifier) not in (str, int) or not str(identifier):
        raise RFSValidationError("Candidate identity is required for export")
    identifier = str(identifier)
    status = data.get("status", data.get("candidate_status"))
    if status not in ("succeeded", "failed", "error", "cancelled", "partial"):
        raise RFSValidationError("Exchange export requires an explicit terminal candidate status")
    failed = status in ("failed", "error", "cancelled", "partial") or data.get("review_status") in ("failed", "invalid")
    status = "failed" if failed else "succeeded"
    desc_key, record = _selected_record(data)
    if record is not None and not isinstance(record, dict):
        raise RFSValidationError("Selected artifact record must be an object")
    spec_value = record.get("figure_spec", data.get("figure_spec")) if record else data.get("figure_spec")
    spec = validate_figure_spec(spec_value) if spec_value is not None else None
    if status == "succeeded" and (record is None or spec is None):
        raise RFSValidationError("Successful exchange requires a selected image record and scientific contract")
    payloads: dict[str, tuple[str, bytes, str]] = {}
    description = record.get("design_description", data.get("rfs_design_description", "")) if record else data.get("rfs_design_description", "")
    if record is not None:
        for key in ("candidate_id", "stage", "provider", "model", "prompt"):
            if key not in record or (key != "candidate_id" and (not isinstance(record[key], str) or not record[key].strip())):
                raise RFSValidationError(f"Selected artifact missing actual {key}")
        if str(record["candidate_id"]) != identifier:
            raise RFSValidationError("Selected artifact belongs to a different candidate")
        if spec is None:
            raise RFSValidationError("An image artifact requires its FigureSpec snapshot")
        expected_hash = sha256_text(canonical_json(spec))
        if record.get("figure_spec_sha256", expected_hash) != expected_hash:
            raise RFSValidationError("Selected artifact and FigureSpec checksum disagree")
        payloads.update(_image_files(record))
        payloads["prompt"] = ("prompt.txt", record["prompt"].encode("utf-8"), "text/plain")
        system_prompt = record.get("system_prompt")
        if system_prompt is not None:
            if not isinstance(system_prompt, str):
                raise RFSValidationError("System prompt must be actual submitted text or null")
            payloads["system_prompt"] = ("prompt.system.txt", system_prompt.encode("utf-8"), "text/plain")
        plot_code = record.get("code", data.get(f"{desc_key}_code"))
        if isinstance(plot_code, str) and plot_code.strip():
            payloads["plot_code"] = ("plot_code.py", plot_code.encode("utf-8"), "text/x-python")
    if spec is not None:
        payloads["figure_spec"] = ("figure_spec.json", (json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"), "application/json")
        input_spec = data.get("rfs_input_figure_spec")
        if input_spec is not None:
            original = validate_figure_spec(input_spec)
            payloads["input_figure_spec"] = ("figure_spec.input.json", (json.dumps(original, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"), "application/json")
        if isinstance(description, str) and description.strip():
            payloads["design"] = ("design.md", _design_document(spec, description, status=status).encode("utf-8"), "text/markdown")
        elif status == "succeeded":
            raise RFSValidationError("Successful exchange requires a detailed design description")
    files = {key: {"path": path, "sha256": hashlib.sha256(contents).hexdigest(), "bytes": len(contents), "mime": mime}
             for key, (path, contents, mime) in payloads.items()}
    manifest = {
        "schema_version": SCHEMA_VERSION, "candidate_id": identifier, "status": status,
        "stage": record.get("stage") if record else None, "provider": record.get("provider") if record else None,
        "model": record.get("model") if record else None, "description_key": desc_key,
        "options": _safe_metadata(record.get("options", {})) if record else {},
        "provenance": {"scientific_verification": spec["provenance"]["scientific_verification"] if spec else "unverified",
                       "external_consumer_verified": False, "editable_reconstruction_verified": False,
                       "image_bytes_preserved": record is not None,
                       "generation": _safe_metadata({**record.get("provenance", {}),
                           "request_id": record.get("request_id"), "usage": record.get("usage"),
                           "attempts": record.get("attempts"), "compiler_metadata": record.get("compiler_metadata", {})}) if record else {}},
        "scientific_fingerprint": scientific_fingerprint(spec) if spec else None,
        "review": _safe_metadata(data.get("rfs_reviews", [])),
        "review_status": data.get("review_status", "not_requested"),
        "error": _safe_metadata(data.get("error")) if failed else None,
        "files": files,
    }
    canonical_json(manifest)
    from jsonschema import Draft202012Validator
    errors = list(Draft202012Validator(json.loads(MANIFEST_SCHEMA_PATH.read_text(encoding="utf-8"))).iter_errors(manifest))
    if errors:
        raise RFSValidationError(f"Invalid result manifest: {errors[0].message}")
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    target = Path(output_dir).absolute()
    if target.is_symlink() or any(parent.is_symlink() for parent in target.parents):
        raise RFSValidationError("Export destination cannot traverse a symlink")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing exchange directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)  # Reserve a new destination, including under concurrent exports.
    try:
        for path, contents, _mime in payloads.values():
            validate_relative_path(path)
            with (target / path).open("xb") as stream:
                stream.write(contents)
        with (target / "manifest.json").open("xb") as stream:
            stream.write(manifest_bytes)
    except Exception:
        shutil.rmtree(target)  # Only the directory reserved by this call.
        raise
    verify_bundle(target)
    return manifest


def verify_bundle(output_dir: str | Path) -> dict:
    """Check a saved exchange manifest and all bytes before checkpoint reuse."""
    target = Path(output_dir)
    if target.is_symlink() or not target.is_dir():
        raise RFSValidationError("Exchange directory must be an ordinary existing directory")
    manifest_path = target / "manifest.json"
    if manifest_path.is_symlink():
        raise RFSValidationError("Manifest cannot be a symlink")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RFSValidationError("Exchange manifest is missing or malformed") from exc
    canonical_json(manifest)  # Reject nonfinite values even in free-form metadata.
    from jsonschema import Draft202012Validator
    schema = json.loads(MANIFEST_SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema).iter_errors(manifest))
    if errors:
        raise RFSValidationError(f"Invalid exchange manifest: {errors[0].message}")
    for file in manifest["files"].values():
        path = validate_relative_path(file["path"])
        local = target / path
        if local.is_symlink() or any(parent.is_symlink() for parent in local.parents):
            raise RFSValidationError("Exchange file cannot traverse a symlink")
        try:
            contents = local.read_bytes()
        except OSError as exc:
            raise RFSValidationError(f"Missing exchange artifact: {path}") from exc
        if len(contents) != file["bytes"] or hashlib.sha256(contents).hexdigest() != file["sha256"]:
            raise RFSValidationError(f"Exchange artifact checksum mismatch: {path}")
        if file["mime"].startswith("image/"):
            try:
                with Image.open(io.BytesIO(contents)) as image:
                    image.verify()
                    if Image.MIME.get(image.format) != file["mime"]:
                        raise RFSValidationError(f"Exchange image MIME mismatch: {path}")
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                raise RFSValidationError(f"Invalid exchange image: {path}") from exc
    if "figure_spec" in manifest["files"]:
        spec = validate_figure_spec(json.loads((target / manifest["files"]["figure_spec"]["path"]).read_text(encoding="utf-8")))
        if scientific_fingerprint(spec) != manifest["scientific_fingerprint"]:
            raise RFSValidationError("Manifest scientific fingerprint mismatch")
    return copy.deepcopy(manifest)
