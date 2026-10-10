"""Explicit provider/model capabilities. No credential- or name-based routing.

The public catalog is documentation evidence, not an account availability test.
Unknown endpoints require a user-supplied, explicitly scoped capability profile.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml


class ModelConfigurationError(ValueError):
    """A model, provider, capability, or generation option is misconfigured."""


@dataclass(frozen=True)
class ModelProfile:
    provider: str
    model_id: str
    endpoint: str
    operations: tuple[str, ...] = ("text",)
    input_modalities: tuple[str, ...] = ("text",)
    output_modalities: tuple[str, ...] = ("text",)
    supports_system_prompt: bool = True
    max_output_tokens: int = 8192
    input_token_limit: int | None = None
    temperature_supported: bool = True
    reasoning_efforts: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None
    image_sizes: tuple[str, ...] = ()
    aspect_ratios: tuple[str, ...] = ()
    image_size_mode: str = "none"
    image_response_modalities: tuple[str, ...] = ("TEXT", "IMAGE")
    max_reference_images: int = 0
    image_quality: tuple[str, ...] = ("auto", "low", "medium", "high")
    input_fidelity: tuple[str, ...] = ()
    status: str = "documented"
    source: str = ""
    checked_at: str = "2026-10-10"
    replacement: str | None = None
    prompt_notes: str = ""

    @property
    def model(self) -> str:
        return self.model_id

    @property
    def supports_reference_image(self) -> bool:
        return "image" in self.input_modalities

    @property
    def supports_editing(self) -> bool:
        return "image_edit" in self.operations

    def compiler_capabilities(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model_id,
            "supports_system_prompt": self.supports_system_prompt,
            "supports_reference_image": self.supports_reference_image,
            "supports_editing": self.supports_editing,
            "prompt_notes": self.prompt_notes,
        }


_TUPLES = {"operations", "input_modalities", "output_modalities", "reasoning_efforts",
           "image_sizes", "aspect_ratios", "image_response_modalities", "image_quality",
           "input_fidelity"}
_OPERATIONS = {"image": "image_generate", "edit": "image_edit", "vision": "text",
               "structured": "structured_text"}
_PROVIDERS = {"gemini", "openai", "anthropic", "openrouter"}


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "configs" / "model_registry.yaml"
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise ModelConfigurationError("Invalid bundled model registry")
    return data


def _profile_data(entry: Mapping[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    base = entry.get("base_profile")
    if base:
        templates = _catalog().get("profiles", {})
        if base not in templates:
            raise ModelConfigurationError(f"Unknown capability profile: {base}")
        data.update(templates[base])
    data.update({key: value for key, value in entry.items() if key != "base_profile"})
    return data


def _make_profile(data: dict[str, Any], *, user_declared: bool = False) -> ModelProfile:
    allowed = {field.name for field in fields(ModelProfile)}
    unsupported = set(data) - allowed
    if unsupported:
        raise ModelConfigurationError("Unknown profile fields: " + ", ".join(sorted(unsupported)))
    for key in _TUPLES & data.keys():
        if not isinstance(data[key], (list, tuple)):
            raise ModelConfigurationError(f"Profile {key} must be a list")
        data[key] = tuple(data[key])
    if data.get("provider") not in _PROVIDERS:
        raise ModelConfigurationError("An explicit supported provider is required")
    if not data.get("model_id") or not data.get("endpoint"):
        raise ModelConfigurationError("Profile model_id and endpoint are required")
    if user_declared:
        data["status"] = "user-declared"
    result = ModelProfile(**data)
    if result.max_output_tokens < 1 or result.max_reference_images < 0:
        raise ModelConfigurationError("Invalid capability limits")
    endpoint_providers = {
        "gemini_generate_content": "gemini",
        "openai_chat": "openai", "openai_responses": "openai", "openai_images": "openai",
        "anthropic_messages": "anthropic", "openrouter_chat": "openrouter",
    }
    if result.endpoint not in endpoint_providers:
        raise ModelConfigurationError(f"Endpoint adapter is not implemented: {result.endpoint}")
    if endpoint_providers.get(result.endpoint) != result.provider:
        raise ModelConfigurationError("Endpoint does not belong to the selected provider")
    return result


def resolve_model(model_name: str, provider: str | None = None, operation: str = "text",
                  overrides: Mapping[str, Any] | None = None) -> ModelProfile:
    """Resolve exact catalog IDs or explicit provider-qualified IDs.

    OpenRouter's vendor/model syntax selects a provider, never a capability.
    Underlying capabilities are copied only for an exact registered vendor ID.
    """
    operation = _OPERATIONS.get(operation, operation)
    if operation not in {"text", "structured_text", "image_generate", "image_edit"}:
        raise ModelConfigurationError(f"Unknown operation: {operation}")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ModelConfigurationError("A model ID is required")
    name = model_name.strip()
    qualified_provider = None
    for known in _PROVIDERS:
        marker = known + "/"
        if name.startswith(marker):
            qualified_provider, name = known, name[len(marker):]
            break
    if provider and qualified_provider and provider != qualified_provider:
        raise ModelConfigurationError("Model prefix conflicts with explicit provider")
    provider = provider or qualified_provider
    catalog = _catalog()
    aliases = catalog.get("aliases", {})
    name = aliases.get(name, name)
    entry = catalog["models"].get(name)
    data = _profile_data(entry) if entry else None

    if provider == "openrouter":
        vendor_names = {"google": "gemini", "openai": "openai", "anthropic": "anthropic"}
        if data is not None and "/" not in name:
            vendor = {value: key for key, value in vendor_names.items()}.get(data.get("provider"))
            if vendor:
                name = vendor + "/" + name
        vendor, _, underlying = name.partition("/")
        underlying_entry = catalog["models"].get(underlying)
        if underlying_entry:
            candidate = _profile_data(underlying_entry)
            if vendor_names.get(vendor) == candidate.get("provider"):
                data = candidate
                data.update(provider="openrouter", endpoint="openrouter_chat", model_id=name,
                            source="", status="gateway-unverified")
        if data is not None and data.get("provider") != "openrouter":
            raise ModelConfigurationError("Use an exact OpenRouter vendor/model ID or explicit profile")

    custom = None
    if overrides:
        custom = overrides.get(model_name) or overrides.get(name)
    if custom is not None:
        if not isinstance(custom, Mapping):
            raise ModelConfigurationError("Model profile override must be a mapping")
        override_data = _profile_data(custom)
        if data is None and not custom.get("base_profile"):
            needed = {"provider", "endpoint", "operations", "input_modalities", "output_modalities"}
            if not needed <= override_data.keys():
                raise ModelConfigurationError("Unknown models need provider, endpoint, operations and input/output modalities")
        data = dict(data or {})
        data.update(override_data)
        data.setdefault("model_id", name)
    if data is None:
        raise ModelConfigurationError(f"Unregistered model '{model_name}'. Supply an explicit model_profiles capability profile.")
    profile = _make_profile(data, user_declared=custom is not None)
    if provider and provider != profile.provider:
        raise ModelConfigurationError(f"Model '{name}' belongs to {profile.provider}, not {provider}")
    if profile.status == "retired":
        raise ModelConfigurationError(f"Model '{name}' is marked retired; use {profile.replacement or 'a supported explicit profile'}")
    if operation not in profile.operations:
        raise ModelConfigurationError(f"{profile.provider}/{profile.model_id} does not support {operation}")
    return profile


def get_default_model(operation: str = "text", provider: str = "gemini") -> str:
    role = "image" if operation in {"image", "image_generate", "image_edit", "edit"} else "text"
    value = _catalog().get("defaults", {}).get(provider, {}).get(role)
    if not value:
        raise ModelConfigurationError(f"No default {role} model for {provider}")
    return str(value)


def list_model_options(operation: str = "text", provider: str | None = None) -> list[str]:
    operation = _OPERATIONS.get(operation, operation)
    result = []
    for name in _catalog()["models"]:
        try:
            profile = resolve_model(name, provider, operation)
        except ModelConfigurationError:
            continue
        result.append(name)
    return result


DEFAULT_MAIN_MODEL = get_default_model("text", "gemini")
DEFAULT_IMAGE_MODEL = get_default_model("image", "gemini")
