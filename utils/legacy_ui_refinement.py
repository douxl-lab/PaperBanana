# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Shared single-user UI image editing and lossless export helpers.

Credentials belong to the server process, never to a browser-session export.
Editing uses the selected model's provider and never silently falls back.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4

from PIL import Image

from utils.legacy_ui_results import resolve_final_output


def image_mime_type(image_bytes: bytes) -> str:
    """Validate encoded pixels and inspect their real format, not their filename."""
    with Image.open(BytesIO(image_bytes)) as image:
        mime = Image.MIME.get(image.format)
        image.verify()
    if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ValueError("Unsupported image format; upload PNG, JPEG, WebP, or GIF.")
    return mime


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


async def refine_image(
    image_bytes: bytes, edit_prompt: str, *, model_name: str,
    aspect_ratio: str = "21:9", image_size: str = "2K", provider: str | None = None,
    overrides: dict | None = None,
) -> tuple[bytes | None, str]:
    """Submit the actual reference image through the capability-checked edit route."""
    from utils.generation_utils import call_image_with_retry_async, GenerationError
    from utils.model_registry import resolve_model, ModelConfigurationError

    if not edit_prompt or not edit_prompt.strip():
        return None, "Please provide edit instructions."
    try:
        mime = image_mime_type(image_bytes)
        profile = resolve_model(model_name, operation="image_edit", provider=provider, overrides=overrides)
        responses = await call_image_with_retry_async(
            model_name=model_name,
            provider=provider, overrides=overrides,
            operation="image_edit",
            contents=[
                {"type": "image", "source": {
                    "type": "base64", "media_type": mime,
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                }},
                {"type": "text", "text": edit_prompt},
            ],
            config={"system_prompt": "", "temperature": 1.0,
                    "aspect_ratio": aspect_ratio, "image_size": image_size,
                    "output_format": "png"},
            max_attempts=3, retry_delay=1, error_context="ui_image_edit",
        )
        if not responses or not isinstance(responses[0], str) or responses[0] == "Error":
            raise ValueError("No edited image returned")
        encoded = responses[0].split(",", 1)[-1] if responses[0].startswith("data:") else responses[0]
        edited = base64.b64decode(encoded, validate=True)
        image_mime_type(edited)
        # Return original provider bytes, preserving PNG/alpha. UIs can encode a
        # separate PNG download when a provider returns another format.
        return edited, f"Image refined successfully ({profile.provider})."
    except Exception as exc:
        # SDK exceptions may include credential-bearing URLs or request headers.
        # Detailed typed failures belong in the runtime's redacted diagnostics.
        if isinstance(exc, ModelConfigurationError):
            return None, "Image refinement configuration error: " + sanitize_export(str(exc))
        if isinstance(exc, GenerationError):
            return None, f"Image refinement failed ({exc.category}); check the selected provider and credentials."
        return None, f"Image refinement failed ({type(exc).__name__}); check the selected model, credentials, and supported options."


def final_image_bytes(result: dict, exp_mode: str = "") -> bytes | None:
    """Return matching lossless stage bytes when available, with legacy fallback."""
    selection = resolve_final_output(result, exp_mode=exp_mode)
    if not selection.image_key:
        return None
    key = selection.image_key
    stem = key.removesuffix("_base64_jpg")
    artifacts = result.get("rfs_artifacts")
    artifact = artifacts.get(stem, {}) if isinstance(artifacts, dict) else {}
    artifact = artifact if isinstance(artifact, dict) else {}
    values = [artifact.get("image_base64"), result.get(f"{stem}_base64"),
              artifact.get("png_base64"), result.get(f"{stem}_base64_png"), result.get(key)]
    for value in values:
        if isinstance(value, str) and value and value != "Error":
            try:
                value = value.split(",", 1)[-1] if value.startswith("data:") else value
                raw = base64.b64decode(value, validate=True)
                image_mime_type(raw)
                return raw
            except (ValueError, OSError):
                continue
    return None


def final_image(result: dict, exp_mode: str = "") -> Image.Image | None:
    raw = final_image_bytes(result, exp_mode)
    if raw is None:
        return None
    with Image.open(BytesIO(raw)) as image:
        image.load()
        return image.copy()


def summarize_results(results: list[dict], exp_mode: str = "") -> dict[str, int]:
    """Count explicit statuses; old files without a status need valid pixels."""
    counts = {"success": 0, "failed": 0, "incomplete": 0}
    for result in results:
        status = result.get("status") or result.get("generation_status")
        valid = final_image_bytes(result, exp_mode) is not None
        if status in {"success", "succeeded", "completed"} and valid:
            counts["success"] += 1
        elif status is None and valid:
            counts["success"] += 1
        elif status in {"incomplete", "partial", "cancelled", "skipped"}:
            counts["incomplete"] += 1
        else:
            counts["failed"] += 1
    return counts


_SECRET_FIELDS = {"api_key", "api_keys", "google_api_key", "openai_api_key", "anthropic_api_key",
                  "openrouter_api_key", "password", "credentials", "access_token", "authorization"}
_SECRET_PATTERN = re.compile(r"\b(?:sk-(?:or-v1-)?[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{20,})\b")


def sanitize_export(value):
    if isinstance(value, dict):
        return {key: sanitize_export(item) for key, item in value.items()
                if str(key).lower() not in _SECRET_FIELDS}
    if isinstance(value, list):
        return [sanitize_export(item) for item in value]
    if isinstance(value, str):
        return _SECRET_PATTERN.sub("[REDACTED]", value)
    return value


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]


def atomic_write(path: str | Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def save_results_json(results: list[dict], output_dir: str | Path, run_id: str | None = None) -> Path:
    data = json.dumps(sanitize_export(results), ensure_ascii=False, indent=2)
    data = data.encode("utf-8", errors="replace")
    return atomic_write(Path(output_dir) / f"demo_{run_id or new_run_id()}.json", data)
