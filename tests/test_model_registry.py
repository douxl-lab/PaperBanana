"""Pure registry checks; no keys, SDK clients, or paid API calls."""
import pytest

from utils.model_registry import (DEFAULT_IMAGE_MODEL, DEFAULT_MAIN_MODEL, ModelConfigurationError,
                                 get_default_model, list_model_options, resolve_model)


def test_stable_defaults_and_exact_capabilities():
    assert DEFAULT_MAIN_MODEL == get_default_model("text") == "gemini-3.8-flash"
    assert DEFAULT_IMAGE_MODEL == get_default_model("image") == "gemini-nano-banana-2.1"
    profile = resolve_model(DEFAULT_IMAGE_MODEL, operation="image_edit")
    assert profile.status == "stable"
    assert profile.max_output_tokens == 32768
    assert profile.compiler_capabilities()["supports_editing"] is True
    with pytest.raises(ModelConfigurationError, match="does not support structured"):
        resolve_model(DEFAULT_IMAGE_MODEL, operation="structured")


def test_models_alias_and_provider_are_exact():
    assert resolve_model("models/gemini-3.8-flash").model_id == "gemini-3.8-flash"
    assert resolve_model("openai/gpt-4.1").provider == "openai"
    with pytest.raises(ModelConfigurationError, match="conflicts"):
        resolve_model("openai/gpt-4.1", provider="gemini")
    with pytest.raises(ModelConfigurationError, match="belongs"):
        resolve_model("gpt-4.1", provider="gemini")


@pytest.mark.parametrize("name", ["some-gpt-image-ish-name", "gpt-unknown", "new-image-model", "gemini-unknown"])
def test_names_are_not_capabilities(name):
    with pytest.raises(ModelConfigurationError, match="Unregistered"):
        resolve_model(name)


def test_unknown_user_declared_capability_profile():
    profile = resolve_model("deployment-v2", operation="image_edit", overrides={
        "deployment-v2": {"base_profile": "openai_image", "model_id": "deployment-v2",
                          "image_size_mode": "openai_flexible"}})
    assert profile.status == "user-declared"
    assert profile.model_id == "deployment-v2"
    assert profile.provider == "openai"
    assert profile.supports_editing


def test_unknown_incomplete_or_mismatched_profile_is_rejected():
    with pytest.raises(ModelConfigurationError, match="Unknown models need"):
        resolve_model("unknown", overrides={"unknown": {"provider": "openai"}})
    with pytest.raises(ModelConfigurationError, match="Endpoint"):
        resolve_model("unknown", overrides={"unknown": {
            "base_profile": "openai_image", "provider": "gemini"}}, operation="image_generate")


def test_unimplemented_endpoint_is_rejected_during_profile_resolution():
    with pytest.raises(ModelConfigurationError, match="adapter is not implemented: gemini_interactions"):
        resolve_model("interactions-deployment", overrides={"interactions-deployment": {
            "base_profile": "gemini_text", "endpoint": "gemini_interactions"}})


def test_openrouter_nested_vendor_id_does_not_use_direct_openai():
    profile = resolve_model("openrouter/openai/gpt-image-1.5", operation="image_edit")
    assert profile.provider == "openrouter"
    assert profile.model_id == "openai/gpt-image-1.5"
    assert profile.endpoint == "openrouter_chat"
    assert profile.status == "gateway-unverified"
    assert resolve_model("gemini-3.8-flash", provider="openrouter").model_id == "google/gemini-3.8-flash"


def test_retired_previews_are_not_defaults_or_silent_aliases():
    assert "gemini-3.1-flash-image-preview" not in list_model_options("image")
    with pytest.raises(ModelConfigurationError, match="retired"):
        resolve_model("gemini-3.1-flash-image-preview", operation="image_generate")


def test_provider_keys_cannot_change_resolution(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-value-not-a-real-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-value-not-a-real-key")
    assert resolve_model("gemini-3.8-flash").provider == "gemini"
    assert resolve_model("gpt-image-1.5", operation="image_generate").provider == "openai"


def test_legacy_claude_profile_is_preserved_without_account_claim():
    profile = resolve_model("claude-sonnet-4-20250514")
    assert profile.provider == "anthropic"
    assert profile.status == "legacy-unverified"
    assert profile.supports_reference_image
