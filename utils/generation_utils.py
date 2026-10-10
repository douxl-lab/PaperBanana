# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit provider adapters with bounded retry, original-image fidelity, and
legacy List[str] wrappers. Failures are typed exceptions, never success strings.
"""
from __future__ import annotations

import asyncio
import base64
import copy
import inspect
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

import httpx
import yaml
from PIL import Image, UnidentifiedImageError

from utils.model_registry import ModelConfigurationError, ModelProfile, resolve_model

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = types = None
try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

logger = logging.getLogger(__name__)
config_path = Path(__file__).resolve().parents[1] / "configs" / "model_config.yaml"
model_config: dict[str, Any] = {}
_configuration_loaded = False

gemini_client = anthropic_client = openai_client = openrouter_client = None
openrouter_api_key = ""


def configure_generation(path=None):
    """Select one configuration before scheduling calls; never guess a provider.

    Switching an already initialized context requires explicit async cleanup.
    Secret-bearing sections stay in this module and never enter result records.
    """
    global config_path, model_config, _configuration_loaded
    selected = Path(path or config_path).resolve()
    if _configuration_loaded and selected == config_path.resolve():
        return
    if any(client is not None for client in (gemini_client, anthropic_client, openai_client, openrouter_client)):
        raise ModelConfigurationError("Close existing provider clients before switching configuration")
    data = {}
    if selected.exists():
        try:
            with selected.open(encoding="utf-8-sig") as handle:
                data = yaml.safe_load(handle) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ModelConfigurationError("Cannot read the selected provider configuration") from exc
        if not isinstance(data, dict):
            raise ModelConfigurationError("Provider configuration must be a mapping")
    elif path is not None and selected != (Path(__file__).resolve().parents[1] / "configs" / "model_config.yaml"):
        raise ModelConfigurationError("Selected provider configuration does not exist")
    for section in ("api_keys", "defaults", "google_cloud", "provider_settings"):
        if not isinstance(data.get(section, {}), dict):
            raise ModelConfigurationError(f"Provider configuration {section} must be a mapping")
    config_path, model_config, _configuration_loaded = selected, data, True


async def configure_generation_async(path=None):
    await close_clients_async()
    configure_generation(path)


def _configured_profiles():
    return model_config.get("defaults", {}).get("model_profiles", model_config.get("model_profiles", {}))


class GenerationError(RuntimeError):
    """Safe, classified provider failure. Does not include input or credentials."""
    def __init__(self, message: str, *, category: str = "provider_error",
                 retryable: bool = False, provider: str = "", model: str = "",
                 status_code: int | None = None, attempts: int = 1):
        super().__init__(message)
        self.category, self.retryable = category, retryable
        self.provider, self.model = provider, model
        self.status_code, self.attempts = status_code, attempts


class GenerationResponseError(GenerationError):
    """A completed API response was refused, truncated, empty, or malformed."""


@dataclass(frozen=True)
class ImageInput:
    data: bytes
    mime_type: str


@dataclass(frozen=True)
class GenerationSample:
    text: str | None = None
    image_base64: str | None = None
    image_mime: str | None = None
    width: int | None = None
    height: int | None = None
    parsed: Any = None
    finish_reason: str | None = None


@dataclass
class GenerationResult:
    samples: list[GenerationSample]
    provider: str
    model: str
    request_id: str | None = None
    usage: Any = None
    elapsed: float = 0.0
    attempts: int = 1
    warnings: list[str] = field(default_factory=list)
    effective_prompt: str | None = None
    effective_system_prompt: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


def get_config_val(section, key, env_var, default=""):
    value = os.getenv(env_var)
    if not value:
        value = model_config.get(section, {}).get(key)
    return value or default


def reinitialize_clients():
    """Rebuild configured clients without probing or making paid requests.

    Retry belongs to this module, not both this module and the SDK.
    """
    global gemini_client, anthropic_client, openai_client, openrouter_client, openrouter_api_key
    if not _configuration_loaded:
        configure_generation()
    initialized = []
    settings = model_config.get("provider_settings", {})
    key = get_config_val("api_keys", "google_api_key", "GOOGLE_API_KEY", "")
    project = get_config_val("google_cloud", "project_id", "GOOGLE_CLOUD_PROJECT", "")
    gemini_client = None
    if genai is not None and (key or project):
        options = types.HttpOptions(timeout=300000, retry_options=types.HttpRetryOptions(attempts=1))
        if key:
            gemini_client = genai.Client(api_key=key, http_options=options)
        else:
            location = get_config_val("google_cloud", "location", "GOOGLE_CLOUD_LOCATION", "global")
            gemini_client = genai.Client(vertexai=True, project=project, location=location, http_options=options)
        initialized.append("Gemini")
    key = get_config_val("api_keys", "anthropic_api_key", "ANTHROPIC_API_KEY", "")
    anthropic_options = {"base_url": settings["anthropic"]["base_url"]} if settings.get("anthropic", {}).get("base_url") else {}
    anthropic_client = AsyncAnthropic(api_key=key, max_retries=0, timeout=300, **anthropic_options) if key and AsyncAnthropic else None
    if anthropic_client is not None:
        initialized.append("Anthropic")
    key = get_config_val("api_keys", "openai_api_key", "OPENAI_API_KEY", "")
    openai_options = {"base_url": settings["openai"]["base_url"]} if settings.get("openai", {}).get("base_url") else {}
    openai_client = AsyncOpenAI(api_key=key, max_retries=0, timeout=300, **openai_options) if key and AsyncOpenAI else None
    if openai_client is not None:
        initialized.append("OpenAI")
    openrouter_api_key = get_config_val("api_keys", "openrouter_api_key", "OPENROUTER_API_KEY", "")
    openrouter_client = AsyncOpenAI(base_url=settings.get("openrouter", {}).get("base_url", "https://openrouter.ai/api/v1"), api_key=openrouter_api_key,
                                    max_retries=0, timeout=300) if openrouter_api_key and AsyncOpenAI else None
    if openrouter_client is not None:
        initialized.append("OpenRouter")
    return initialized


async def close_clients_async():
    """Release clients explicitly when the owning application shuts down."""
    global gemini_client, anthropic_client, openai_client, openrouter_client
    clients = [gemini_client, anthropic_client, openai_client, openrouter_client]
    gemini_client = anthropic_client = openai_client = openrouter_client = None
    for client in clients:
        if client is None:
            continue
        aio = getattr(client, "aio", None)
        closer = getattr(aio, "aclose", None) or getattr(client, "close", None)
        if closer:
            result = closer()
            if inspect.isawaitable(result):
                await result


def _client(profile: ModelProfile):
    value = {"gemini": gemini_client, "anthropic": anthropic_client,
             "openai": openai_client, "openrouter": openrouter_client}[profile.provider]
    if value is None:
        reinitialize_clients()
        value = {"gemini": gemini_client, "anthropic": anthropic_client,
                 "openai": openai_client, "openrouter": openrouter_client}[profile.provider]
    if value is None:
        raise GenerationError(f"{profile.provider} client is not configured or its SDK is unavailable",
                              category="configuration", provider=profile.provider, model=profile.model_id)
    return value


def _config_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return {key: value for key, value in config.items() if value is not None}
    names = ("system_instruction", "temperature", "candidate_count", "max_output_tokens", "response_schema",
             "response_json_schema", "response_mime_type", "image_config", "thinking_config")
    return {key: getattr(config, key) for key in names if getattr(config, key, None) is not None}


def image_metadata(encoded: str | bytes) -> dict[str, Any]:
    try:
        data = base64.b64decode(encoded, validate=True) if isinstance(encoded, str) else encoded
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            mime = Image.MIME.get(image.format)
            if not mime or not mime.startswith("image/"):
                raise ValueError("Unknown image format")
            return {"bytes": data, "mime_type": mime, "width": image.width, "height": image.height}
    except (ValueError, UnidentifiedImageError, OSError, TypeError) as exc:
        raise ModelConfigurationError("Invalid or unsupported image bytes/base64") from exc


def lossless_png_base64(encoded: str) -> str:
    """Return original PNG bytes, or a lossless PNG container for other formats.

    Re-encoding a JPEG does not restore details lost before receipt.
    """
    meta = image_metadata(encoded)
    if meta["mime_type"] == "image/png":
        return encoded
    with Image.open(BytesIO(meta["bytes"])) as image:
        output = BytesIO()
        image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def normalize_contents(contents: Any) -> list[dict[str, Any]]:
    if isinstance(contents, str):
        contents = [{"type": "text", "text": contents}]
    if not isinstance(contents, (list, tuple)):
        raise ModelConfigurationError("Contents must be text or a list of text/image blocks")
    result = []
    for item in contents:
        if not isinstance(item, Mapping):
            raise ModelConfigurationError("Each content block must be a mapping")
        kind = item.get("type")
        if kind in {"text", "input_text"}:
            if not isinstance(item.get("text"), str):
                raise ModelConfigurationError("Text blocks require string text")
            result.append({"type": "text", "text": item["text"]})
        elif kind in {"image", "input_image", "image_url"}:
            source = item.get("source", {})
            encoded = item.get("image_base64") or item.get("data")
            if source.get("type") == "base64":
                encoded = source.get("data")
            url = item.get("image_url")
            if isinstance(url, Mapping):
                url = url.get("url")
            if url:
                if not isinstance(url, str) or not url.startswith("data:image/") or ";base64," not in url:
                    raise ModelConfigurationError("Only inline image data is accepted; remote URLs need explicit materialization")
                encoded = url.split(";base64,", 1)[1]
            if isinstance(encoded, bytes):
                meta = image_metadata(encoded)
            elif isinstance(encoded, str):
                meta = image_metadata(encoded)
            else:
                raise ModelConfigurationError("Image blocks require bytes or base64 data")
            result.append({"type": "image", "data": meta["bytes"], "mime_type": meta["mime_type"]})
        else:
            raise ModelConfigurationError(f"Unsupported content block type: {kind!r}")
    return result


def _convert_to_gemini_parts(contents):
    if types is None:
        raise GenerationError("google-genai SDK is unavailable", category="configuration")
    return [types.Part.from_text(text=item["text"]) if item["type"] == "text" else
            types.Part.from_bytes(data=item["data"], mime_type=item["mime_type"])
            for item in normalize_contents(contents)]


def _convert_to_openai_format(contents, *, responses=False):
    blocks = []
    for item in normalize_contents(contents):
        if item["type"] == "text":
            blocks.append({"type": "input_text" if responses else "text", "text": item["text"]})
        else:
            url = f"data:{item['mime_type']};base64," + base64.b64encode(item["data"]).decode("ascii")
            blocks.append({"type": "input_image", "image_url": url} if responses else
                          {"type": "image_url", "image_url": {"url": url}})
    return blocks


def _convert_to_claude_format(contents):
    return [{"type": "text", "text": item["text"]} if item["type"] == "text" else
            {"type": "image", "source": {"type": "base64", "media_type": item["mime_type"],
             "data": base64.b64encode(item["data"]).decode("ascii")}} for item in normalize_contents(contents)]


def _schema(config):
    schema = config.get("schema") or config.get("response_json_schema") or config.get("response_schema")
    if schema is None:
        return None
    if hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    elif hasattr(schema, "model_dump"):
        schema = schema.model_dump(exclude_none=True, mode="json")
    if not isinstance(schema, Mapping):
        raise ModelConfigurationError("Structured output schema must be JSON Schema or a Pydantic model")
    return copy.deepcopy(dict(schema))


def _strict_schema(schema):
    value = copy.deepcopy(schema)
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    walk(value)
    if value.get("type") != "object":
        raise ModelConfigurationError("OpenAI strict schema requires an object root")
    return value


def _validate_schema_output(text, schema):
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise GenerationResponseError("Provider returned invalid JSON", category="invalid_json") from None
    if schema is not None:
        try:
            import jsonschema
        except ImportError as exc:
            raise ModelConfigurationError("jsonschema is required to validate structured output") from exc
        try:
            jsonschema.validate(parsed, schema)
        except jsonschema.ValidationError as exc:
            raise GenerationResponseError("Provider output does not match the requested schema", category="invalid_schema") from None
        except jsonschema.SchemaError as exc:
            raise ModelConfigurationError("Requested JSON schema is invalid") from exc
    return parsed


def _text_options(profile, config):
    count = config.get("candidate_count", config.get("candidate_num", 1))
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 10:
        raise ModelConfigurationError("sample_count must be between 1 and 10")
    maximum = config.get("max_output_tokens", config.get("max_completion_tokens", min(8192, profile.max_output_tokens)))
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
        raise ModelConfigurationError("max_output_tokens must be a positive integer")
    # Legacy callers requested 50K uniformly. Cap locally and expose the warning.
    capped = min(maximum, profile.max_output_tokens)
    warnings = [f"max_output_tokens capped from {maximum} to {capped}"] if capped != maximum else []
    temperature = config.get("temperature")
    if temperature is not None and (not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2):
        raise ModelConfigurationError("temperature must be a finite number between 0 and 2")
    if temperature is not None and not profile.temperature_supported:
        warnings.append("temperature omitted because this model profile does not support it")
        temperature = None
    effort = config.get("reasoning_effort", profile.default_reasoning_effort)
    if effort is not None and effort not in profile.reasoning_efforts:
        raise ModelConfigurationError("Unsupported reasoning effort for this model profile")
    return count, capped, temperature, effort, warnings


def _check_images(profile, blocks, operation):
    images = [item for item in blocks if item["type"] == "image"]
    if images and not profile.supports_reference_image:
        raise ModelConfigurationError("This model profile cannot accept reference images")
    if len(images) > profile.max_reference_images:
        raise ModelConfigurationError("Too many reference images for this model profile")
    if operation == "image_edit" and not images:
        raise ModelConfigurationError("Image editing requires an original/reference image")
    return images


def _image_options(profile, config):
    nested = config.get("image_config")
    if nested is not None:
        nested = dict(nested) if isinstance(nested, Mapping) else nested.model_dump(exclude_none=True)
    values = dict(nested or {})
    values.update({key: value for key, value in config.items() if key != "image_config"})
    ratio = str(values.get("aspect_ratio", "1:1"))
    try:
        width, height = map(int, ratio.split(":"))
        if width <= 0 or height <= 0:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ModelConfigurationError("aspect_ratio must be positive W:H integers") from exc
    if profile.aspect_ratios and ratio not in profile.aspect_ratios:
        raise ModelConfigurationError(f"Unsupported aspect ratio {ratio} for {profile.model_id}")
    resolution = str(values.get("image_size", "1K")).upper()
    options = {"aspect_ratio": ratio, "image_size": resolution,
               "output_format": str(values.get("output_format", "png")).lower(),
               "quality": values.get("quality", "high"), "background": values.get("background", "opaque")}
    if options["output_format"] not in {"png", "jpeg", "webp"}:
        raise ModelConfigurationError("output_format must be png, jpeg, or webp")
    if options["quality"] not in profile.image_quality:
        raise ModelConfigurationError("Unsupported image quality for this model profile")
    if options["background"] not in {"auto", "opaque", "transparent"}:
        raise ModelConfigurationError("Unsupported image background")
    if options["background"] == "transparent" and options["output_format"] == "jpeg":
        raise ModelConfigurationError("Transparent images require png or webp")
    if profile.image_size_mode == "gemini":
        if resolution not in profile.image_sizes:
            raise ModelConfigurationError(f"Unsupported resolution {resolution} for {profile.model_id}")
        if options["background"] == "transparent":
            raise ModelConfigurationError("Transparent output is not declared by this Gemini profile")
        if options["output_format"] != "png":
            raise ModelConfigurationError("The generateContent adapter cannot request a non-PNG image format")
        if "quality" in values:
            raise ModelConfigurationError("Gemini generateContent has no declared image quality parameter")
    elif profile.image_size_mode in {"openai_legacy", "openai_flexible"}:
        if profile.image_size_mode == "openai_legacy":
            if "size" not in values:
                if resolution != "1K":
                    raise ModelConfigurationError("This image model does not support requested 2K/4K; select a supported size/model")
                supported = {"1:1": "1024x1024", "3:2": "1536x1024", "2:3": "1024x1536"}
                if ratio not in supported:
                    raise ModelConfigurationError("This image model supports only 1:1, 3:2 and 2:3 output ratios")
                size = supported[ratio]
            else:
                size = str(values["size"])
            if size not in profile.image_sizes:
                raise ModelConfigurationError("Unsupported size for this image model")
        else:
            if max(width / height, height / width) > 3:
                raise ModelConfigurationError("OpenAI image aspect ratio cannot exceed 3:1")
            if resolution not in {"1K", "2K", "4K"}:
                raise ModelConfigurationError("image_size must be 1K, 2K, or 4K")
            if "size" in values:
                size = str(values["size"])
            else:
                edge = {"1K": 1536, "2K": 2048, "4K": 3840}[resolution]
                if width == height:
                    edge = {"1K": 1024, "2K": 2048, "4K": 2880}[resolution]
                scale = edge / max(width, height)
                w, h = max(16, round(width * scale / 16) * 16), max(16, round(height * scale / 16) * 16)
                size = f"{w}x{h}"
            if size != "auto":
                try:
                    w, h = map(int, size.split("x"))
                except ValueError as exc:
                    raise ModelConfigurationError("Image size must be WIDTHxHEIGHT") from exc
                if not (0 < w <= 3840 and 0 < h <= 3840 and w % 16 == h % 16 == 0 and
                        max(w / h, h / w) <= 3 and 655360 <= w * h <= 8294400):
                    raise ModelConfigurationError("Image size violates this model's pixel/edge/aspect limits")
            options["size"] = size
        options["size"] = size
    else:
        raise ModelConfigurationError("Image profile needs an explicit image_size_mode")
    fidelity = values.get("input_fidelity")
    if fidelity is not None:
        if fidelity not in profile.input_fidelity:
            raise ModelConfigurationError("input_fidelity is not supported by this model profile")
        options["input_fidelity"] = fidelity
    return options


def _status(exc):
    result = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None) or getattr(exc, "code", None)
    return result if isinstance(result, int) else None


def _classify(exc, profile):
    if isinstance(exc, (GenerationError, ModelConfigurationError)):
        return exc
    status = _status(exc)
    names = {type(exc).__name__, *(base.__name__ for base in type(exc).__mro__)}
    transient = isinstance(exc, (TimeoutError, httpx.TimeoutException, httpx.NetworkError)) or bool(names & {
        "APIConnectionError", "APITimeoutError", "ConnectionError", "TimeoutException", "NetworkError",
        "ConnectError", "ReadError", "WriteError", "RemoteProtocolError"})
    transient = transient or status in {408, 409, 429, 500, 502, 503, 504}
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        code = error.get("code", "") if isinstance(error, dict) else ""
        if code in {"insufficient_quota", "billing_hard_limit_reached", "billing_not_active"}:
            transient = False
    category = "timeout" if isinstance(exc, (TimeoutError, httpx.TimeoutException)) or "APITimeoutError" in names else "transient" if transient else "provider_error"
    return GenerationError(f"{profile.provider} request failed ({category}, status={status or 'unknown'})",
                           category=category, retryable=transient, provider=profile.provider, model=profile.model_id, status_code=status)


def _retry_after(exc):
    headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0, float(value))
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            return max(0, (date - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None


async def _retry(call: Callable[[], Awaitable[Any]], profile: ModelProfile, *, max_attempts=3,
                 retry_delay=1, timeout=300, total_timeout=None):
    if not isinstance(max_attempts, int) or not 1 <= max_attempts <= 10:
        raise ModelConfigurationError("max_attempts must be between 1 and 10")
    if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(retry_delay) or retry_delay < 0:
        raise ModelConfigurationError("Timeout must be positive and retry delay nonnegative")
    if total_timeout is None:
        total_timeout = timeout * max_attempts + sum(min(retry_delay * 2 ** n, 30) for n in range(max_attempts - 1))
    if not isinstance(total_timeout, (int, float)) or not math.isfinite(total_timeout) or total_timeout <= 0:
        raise GenerationError("Generation deadline exceeded", category="timeout", provider=profile.provider, model=profile.model_id, attempts=0)
    deadline = time.monotonic() + total_timeout
    for attempt in range(1, max_attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GenerationError("Generation deadline exceeded", category="timeout", provider=profile.provider, model=profile.model_id, attempts=attempt - 1)
        try:
            result = await asyncio.wait_for(call(), timeout=min(timeout, remaining))
            return result, attempt
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = _classify(exc, profile)
            if not isinstance(error, GenerationError):
                raise error
            error.attempts = attempt
            if not error.retryable or attempt == max_attempts:
                raise error from None
            delay = _retry_after(exc)
            if delay is None:
                delay = min(retry_delay * 2 ** (attempt - 1), 30) * random.uniform(0.8, 1.2)
            if delay >= deadline - time.monotonic():
                raise GenerationError("Generation retry would exceed deadline", category="timeout", provider=profile.provider, model=profile.model_id, attempts=attempt) from None
            logger.warning("Retrying provider=%s model=%s status=%s attempt=%s", profile.provider, profile.model_id, error.status_code, attempt)
            await asyncio.sleep(delay)


def _safe_dump(value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return value if isinstance(value, (dict, list, str, int, float)) else None


def _finish(value):
    return str(getattr(value, "value", value) or "").upper()


def _image_sample(encoded):
    meta = image_metadata(encoded)
    return GenerationSample(image_base64=base64.b64encode(meta["bytes"]).decode("ascii"),
                            image_mime=meta["mime_type"], width=meta["width"], height=meta["height"])


def _effective_contents(profile, blocks, system):
    """Fold unsupported system instructions into the actual user prompt once."""
    prompt = "\n".join(item["text"] for item in blocks if item["type"] == "text")
    if profile.supports_system_prompt:
        return blocks, prompt, system
    if system:
        prompt = system + "\n\n" + prompt
        # Preserve every reference byte and its order. The prompt is one exact
        # text block, so wire content and artifact metadata cannot diverge.
        blocks = [{"type": "text", "text": prompt}] + [item for item in blocks if item["type"] == "image"]
    return blocks, prompt, None


async def generate_text(model_name, contents, config=None, *, provider=None, overrides=None,
                        max_attempts=3, retry_delay=1, timeout=300, total_timeout=None):
    cfg = _config_dict(config)
    schema = _schema(cfg)
    json_mode = schema is not None or cfg.get("response_mime_type") == "application/json"
    profile = resolve_model(model_name, provider, "structured_text" if json_mode else "text",
                            overrides or _configured_profiles())
    blocks = normalize_contents(contents)
    _check_images(profile, blocks, "text")
    count, max_tokens, temperature, effort, warnings = _text_options(profile, cfg)
    system = cfg.get("system_prompt", cfg.get("system_instruction", "")) or ""
    if not isinstance(system, str):
        raise ModelConfigurationError("system prompt must be text")
    blocks, effective_prompt, effective_system = _effective_contents(profile, blocks, system)
    client = _client(profile)
    samples, usage, ids, total_attempts = [], [], [], 0
    started = time.monotonic()
    strict_schema = _strict_schema(schema) if schema and profile.provider in {"openai", "openrouter"} else None
    # Each sample has its own retry budget; never count text parts as candidates.
    async def one():
        if profile.endpoint == "gemini_generate_content":
            params = {"system_instruction": effective_system, "max_output_tokens": max_tokens, "candidate_count": 1,
                      "http_options": types.HttpOptions(timeout=int(timeout * 1000), retry_options=types.HttpRetryOptions(attempts=1))}
            if temperature is not None:
                params["temperature"] = temperature
            if effort:
                params["thinking_config"] = types.ThinkingConfig(thinking_level=effort)
            elif cfg.get("thinking_config") is not None:
                params["thinking_config"] = cfg["thinking_config"]
            if json_mode:
                params["response_mime_type"] = "application/json"
                if schema:
                    params["response_json_schema"] = schema
            response = await client.aio.models.generate_content(model=profile.model_id,
                        contents=_convert_to_gemini_parts(blocks), config=types.GenerateContentConfig(**params))
            candidates = response.candidates or []
            if not candidates:
                raise GenerationResponseError("Gemini returned no candidate", category="empty_response")
            candidate = candidates[0]
            finish = _finish(getattr(candidate, "finish_reason", None))
            if finish in {"MAX_TOKENS", "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY"}:
                raise GenerationResponseError("Gemini response was blocked or incomplete", category="incomplete" if finish == "MAX_TOKENS" else "refusal")
            parts = getattr(getattr(candidate, "content", None), "parts", None) or []
            text = "".join(part.text for part in parts if getattr(part, "text", None) and not getattr(part, "thought", False))
            return text, _safe_dump(getattr(response, "usage_metadata", None)), getattr(response, "response_id", None), finish
        if profile.endpoint == "anthropic_messages":
            params = {"model": profile.model_id, "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": _convert_to_claude_format(blocks)}], "timeout": timeout}
            if effective_system is not None:
                params["system"] = effective_system
            if temperature is not None:
                params["temperature"] = temperature
            response = await client.messages.create(**params)
            finish = _finish(getattr(response, "stop_reason", None))
            if finish == "MAX_TOKENS":
                raise GenerationResponseError("Anthropic response was incomplete", category="incomplete")
            text = "".join(part.text for part in response.content if getattr(part, "type", "text") == "text")
            return text, _safe_dump(getattr(response, "usage", None)), getattr(response, "id", None), finish
        if profile.endpoint in {"openai_chat", "openrouter_chat"}:
            messages = ([{"role": "system", "content": effective_system}] if effective_system is not None else [])
            messages.append({"role": "user", "content": _convert_to_openai_format(blocks)})
            params = {"model": profile.model_id, "messages": messages,
                      "max_completion_tokens": max_tokens, "timeout": timeout}
            if temperature is not None:
                params["temperature"] = temperature
            if effort:
                params["reasoning_effort"] = effort
            if json_mode:
                params["response_format"] = {"type": "json_schema", "json_schema": {"name": "paperbanana_output", "schema": strict_schema, "strict": True}} if schema else {"type": "json_object"}
            response = await client.chat.completions.create(**params)
            if not response.choices:
                raise GenerationResponseError("Provider returned no choices", category="empty_response")
            choice = response.choices[0]
            if getattr(choice.message, "refusal", None) or choice.finish_reason == "content_filter":
                raise GenerationResponseError("Provider refused the request", category="refusal")
            if choice.finish_reason == "length":
                raise GenerationResponseError("Provider response was incomplete", category="incomplete")
            return choice.message.content or "", _safe_dump(getattr(response, "usage", None)), getattr(response, "_request_id", None), choice.finish_reason
        if profile.endpoint == "openai_responses":
            params = {"model": profile.model_id,
                      "input": [{"role": "user", "content": _convert_to_openai_format(blocks, responses=True)}],
                      "max_output_tokens": max_tokens, "timeout": timeout}
            if effective_system is not None:
                params["instructions"] = effective_system
            if temperature is not None:
                params["temperature"] = temperature
            if effort:
                params["reasoning"] = {"effort": effort}
            if json_mode:
                params["text"] = {"format": {"type": "json_schema", "name": "paperbanana_output", "schema": strict_schema, "strict": True}} if schema else {"format": {"type": "json_object"}}
            response = await client.responses.create(**params)
            if getattr(response, "status", "completed") != "completed":
                raise GenerationResponseError("Responses output was incomplete", category="incomplete")
            for output in getattr(response, "output", []):
                for part in getattr(output, "content", []):
                    if getattr(part, "type", "") == "refusal":
                        raise GenerationResponseError("Provider refused the request", category="refusal")
            return response.output_text or "", _safe_dump(getattr(response, "usage", None)), getattr(response, "_request_id", None), "completed"
        raise ModelConfigurationError("This endpoint has no implemented text adapter")
    deadline = time.monotonic() + total_timeout if total_timeout is not None else None
    for _ in range(count):
        remaining = deadline - time.monotonic() if deadline else None
        (text, sample_usage, request_id, finish), attempts = await _retry(one, profile, max_attempts=max_attempts,
                         retry_delay=retry_delay, timeout=timeout, total_timeout=remaining)
        if not isinstance(text, str) or not text.strip():
            raise GenerationResponseError("Provider returned empty text", category="empty_response", provider=profile.provider, model=profile.model_id)
        parsed = _validate_schema_output(text, schema) if json_mode else None
        samples.append(GenerationSample(text=text, parsed=parsed, finish_reason=finish))
        usage.append(sample_usage)
        ids.append(request_id)
        total_attempts += attempts
    return GenerationResult(samples, profile.provider, profile.model_id, next((item for item in ids if item), None),
                            usage, time.monotonic() - started, total_attempts, warnings,
                            effective_prompt=effective_prompt, effective_system_prompt=effective_system,
                            options={"sample_count": count, "max_output_tokens": max_tokens})


async def generate_images(model_name, contents, config=None, *, provider=None, operation="image_generate",
                          overrides=None, max_attempts=3, retry_delay=1, timeout=300, total_timeout=None):
    if operation not in {"image_generate", "image_edit"}:
        raise ModelConfigurationError("Image operation must be image_generate or image_edit")
    profile = resolve_model(model_name, provider, operation, overrides or _configured_profiles())
    cfg = _config_dict(config)
    blocks = normalize_contents(contents)
    images = _check_images(profile, blocks, operation)
    options = _image_options(profile, cfg)
    count = cfg.get("candidate_count", cfg.get("candidate_num", cfg.get("n", 1)))
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 10:
        raise ModelConfigurationError("Image sample count must be between 1 and 10")
    system = cfg.get("system_prompt", cfg.get("system_instruction", "")) or ""
    prompt = "\n".join(item["text"] for item in blocks if item["type"] == "text")
    if not isinstance(system, str) or not prompt.strip():
        raise ModelConfigurationError("Image generation requires a text prompt")
    blocks, effective_prompt, effective_system = _effective_contents(profile, blocks, system)
    if images and operation == "image_generate" and not profile.supports_editing:
        raise ModelConfigurationError("Reference-conditioned generation requires editing capability")
    client = _client(profile)
    started = time.monotonic()
    async def one():
        if profile.endpoint == "gemini_generate_content":
            params = {"system_instruction": effective_system, "candidate_count": 1,
                      "max_output_tokens": min(cfg.get("max_output_tokens", 8192), profile.max_output_tokens),
                      "response_modalities": list(profile.image_response_modalities),
                      "image_config": types.ImageConfig(aspect_ratio=options["aspect_ratio"], image_size=options["image_size"]),
                      "http_options": types.HttpOptions(timeout=int(timeout * 1000), retry_options=types.HttpRetryOptions(attempts=1))}
            if cfg.get("temperature") is not None and profile.temperature_supported:
                params["temperature"] = cfg["temperature"]
            effort = cfg.get("reasoning_effort")
            if effort is not None:
                if effort not in profile.reasoning_efforts:
                    raise ModelConfigurationError("Unsupported reasoning effort for this image profile")
                params["thinking_config"] = types.ThinkingConfig(thinking_level=effort)
            elif cfg.get("thinking_config") is not None:
                params["thinking_config"] = cfg["thinking_config"]
            response = await client.aio.models.generate_content(model=profile.model_id,
                         contents=_convert_to_gemini_parts(blocks), config=types.GenerateContentConfig(**params))
            candidates = response.candidates or []
            if not candidates:
                raise GenerationResponseError("Gemini returned no image candidate", category="empty_response")
            samples = []
            for candidate in candidates:
                finish = _finish(getattr(candidate, "finish_reason", None))
                if finish in {"SAFETY", "IMAGE_SAFETY", "PROHIBITED_CONTENT", "MAX_TOKENS"}:
                    raise GenerationResponseError("Image was refused or incomplete", category="refusal" if finish != "MAX_TOKENS" else "incomplete")
                for part in getattr(getattr(candidate, "content", None), "parts", []) or []:
                    inline = getattr(part, "inline_data", None)
                    if inline and not getattr(part, "thought", False):
                        data = inline.data
                        encoded = base64.b64encode(data).decode("ascii") if isinstance(data, bytes) else data
                        samples.append(_image_sample(encoded))
            if not samples:
                raise GenerationResponseError("Provider returned no final image", category="empty_response")
            return samples, _safe_dump(getattr(response, "usage_metadata", None)), getattr(response, "response_id", None)
        if profile.endpoint == "openai_images":
            params = {"model": profile.model_id, "prompt": effective_prompt, "n": 1, "timeout": timeout,
                      **{key: options[key] for key in ("size", "quality", "background", "output_format") if key in options}}
            if images:
                # Regeneration with references must use edits too; never drop supplied images.
                params["image"] = [(f"reference-{idx}." + image["mime_type"].split("/")[-1], image["data"], image["mime_type"])
                                   for idx, image in enumerate(images)]
                if "input_fidelity" in options:
                    params["input_fidelity"] = options["input_fidelity"]
                response = await client.images.edit(**params)
            else:
                response = await client.images.generate(**params)
            samples = [_image_sample(image.b64_json) for image in response.data or [] if getattr(image, "b64_json", None)]
            if not samples:
                raise GenerationResponseError("Provider returned no image", category="empty_response")
            return samples, _safe_dump(getattr(response, "usage", None)), getattr(response, "_request_id", None)
        if profile.endpoint == "openrouter_chat":
            messages = ([{"role": "system", "content": effective_system}] if effective_system is not None else [])
            messages.append({"role": "user", "content": _convert_to_openai_format(blocks)})
            params = {"model": profile.model_id, "messages": messages,
                      "extra_body": {"modalities": ["image", "text"], "image_config": {
                           "aspect_ratio": options["aspect_ratio"], "image_size": options["image_size"]}}, "timeout": timeout}
            response = await client.chat.completions.create(**params)
            samples = []
            for choice in response.choices or []:
                finish = _finish(getattr(choice, "finish_reason", None))
                if getattr(choice.message, "refusal", None) or finish == "CONTENT_FILTER":
                    raise GenerationResponseError("Gateway refused the request", category="refusal")
                if finish != "STOP":
                    # Partial image bytes do not establish a completed generation.
                    # Terminal gateway states must never become successful artifacts.
                    raise GenerationResponseError("Gateway image response was incomplete", category="incomplete")
                message = choice.message.model_dump() if hasattr(choice.message, "model_dump") else vars(choice.message)
                image_blocks = list(message.get("images") or [])
                content = message.get("content")
                if isinstance(content, list):
                    image_blocks.extend(content)
                elif isinstance(content, str) and content.startswith("data:image/"):
                    image_blocks.append({"image_url": {"url": content}})
                for block in image_blocks:
                    inline = block.get("inline_data") or block.get("inlineData")
                    if inline and inline.get("data"):
                        samples.append(_image_sample(inline["data"]))
                    else:
                        url = block.get("image_url") or block.get("url")
                        if isinstance(url, dict):
                            url = url.get("url")
                        if isinstance(url, str) and url.startswith("data:image/") and ";base64," in url:
                            samples.append(_image_sample(url.split(";base64,", 1)[1]))
            if not samples:
                raise GenerationResponseError("Gateway returned no inline image", category="empty_response")
            return samples, _safe_dump(getattr(response, "usage", None)), getattr(response, "_request_id", None)
        raise ModelConfigurationError("This endpoint has no implemented image adapter")
    samples, usage, total_attempts, ids = [], [], 0, []
    deadline = time.monotonic() + total_timeout if total_timeout is not None else None
    for _ in range(count):
        remaining = deadline - time.monotonic() if deadline else None
        (new_samples, sample_usage, request_id), attempts = await _retry(one, profile, max_attempts=max_attempts,
                            retry_delay=retry_delay, timeout=timeout, total_timeout=remaining)
        samples.extend(new_samples)
        usage.append(sample_usage)
        ids.append(request_id)
        total_attempts += attempts
    if profile.endpoint == "gemini_generate_content":
        effective_options = {"aspect_ratio": options["aspect_ratio"], "image_size": options["image_size"],
                             "response_modalities": list(profile.image_response_modalities), "candidate_count": 1,
                             "max_output_tokens": min(cfg.get("max_output_tokens", 8192), profile.max_output_tokens)}
        if cfg.get("temperature") is not None and profile.temperature_supported:
            effective_options["temperature"] = cfg["temperature"]
    elif profile.endpoint == "openrouter_chat":
        effective_options = {"modalities": ["image", "text"], "image_config": {
                            "aspect_ratio": options["aspect_ratio"], "image_size": options["image_size"]}}
    else:
        effective_options = {key: options[key] for key in ("size", "quality", "background", "output_format", "input_fidelity") if key in options}
        effective_options["n"] = 1
    effective_options["reference_image_count"] = len(images)
    effective_options["operation"] = "image_edit" if images else operation
    return GenerationResult(samples, profile.provider, profile.model_id, next((item for item in ids if item), None),
                            usage, time.monotonic() - started, total_attempts,
                            effective_prompt=effective_prompt, effective_system_prompt=effective_system, options=effective_options)


async def call_model_with_retry_async(model_name, contents, config, max_attempts=3, retry_delay=1,
                                      error_context="", *, provider=None, timeout=300, overrides=None, total_timeout=None):
    result = await generate_text(model_name, contents, config, provider=provider, overrides=overrides,
                                max_attempts=max_attempts, retry_delay=retry_delay, timeout=timeout, total_timeout=total_timeout)
    return [sample.text for sample in result.samples]


async def call_image_with_retry_async(model_name, contents, config=None, max_attempts=3, retry_delay=1,
                                      error_context="", *, provider=None, operation="image_generate", timeout=300,
                                      overrides=None, total_timeout=None):
    result = await generate_images(model_name, contents, config, provider=provider, operation=operation, overrides=overrides,
                                  max_attempts=max_attempts, retry_delay=retry_delay, timeout=timeout, total_timeout=total_timeout)
    return [sample.image_base64 for sample in result.samples]


async def call_gemini_with_retry_async(model_name, contents, config, max_attempts=3, retry_delay=1, error_context=""):
    cfg = _config_dict(config)
    # A legacy caller's explicit response modalities convey its requested task.
    modalities = getattr(config, "response_modalities", None) if not isinstance(config, Mapping) else config.get("response_modalities")
    image_operation = any(_finish(item) == "IMAGE" for item in modalities or [])
    function = call_image_with_retry_async if image_operation else call_model_with_retry_async
    return await function(model_name, contents, cfg, provider="gemini", max_attempts=max_attempts, retry_delay=retry_delay, error_context=error_context)


async def call_claude_with_retry_async(model_name, contents, config, max_attempts=3, retry_delay=1, error_context=""):
    return await call_model_with_retry_async(model_name, contents, config, provider="anthropic", max_attempts=max_attempts, retry_delay=retry_delay, error_context=error_context)


async def call_openai_with_retry_async(model_name, contents, config, max_attempts=3, retry_delay=1, error_context=""):
    return await call_model_with_retry_async(model_name, contents, config, provider="openai", max_attempts=max_attempts, retry_delay=retry_delay, error_context=error_context)


async def call_openrouter_with_retry_async(model_name, contents, config, max_attempts=3, retry_delay=1, error_context=""):
    return await call_model_with_retry_async(model_name, contents, config, provider="openrouter", max_attempts=max_attempts, retry_delay=retry_delay, error_context=error_context)


async def call_openai_image_generation_with_retry_async(model_name, prompt, config, max_attempts=3, retry_delay=1, error_context=""):
    return await call_image_with_retry_async(model_name, prompt, config, provider="openai", max_attempts=max_attempts, retry_delay=retry_delay, error_context=error_context)


async def call_openrouter_image_generation_with_retry_async(model_name, contents, config, max_attempts=3, retry_delay=1, error_context=""):
    return await call_image_with_retry_async(model_name, contents, config, provider="openrouter", max_attempts=max_attempts, retry_delay=retry_delay, error_context=error_context)


def _to_openrouter_model_id(model_name: str) -> str:
    return resolve_model(model_name, provider="openrouter").model_id


def validate_generation_configuration(exp_config, *, needs_text=True, needs_image=True):
    """Validate exact capabilities and local client setup, without API probes.

    Return only model profiles, never credential values or config contents.
    A constructed client is not evidence of model entitlement or working keys.
    """
    configure_generation(getattr(exp_config, "config_path", None))
    profiles = {}
    overrides = getattr(exp_config, "model_profiles", None)
    if needs_text:
        profiles["text"] = resolve_model(exp_config.main_model_name,
                    getattr(exp_config, "main_model_provider", None) or None, "text", overrides)
        _client(profiles["text"])
    if needs_image:
        profiles["image"] = resolve_model(exp_config.image_gen_model_name,
                    getattr(exp_config, "image_gen_provider", None) or None, "image_generate", overrides)
        _client(profiles["image"])
    return profiles
