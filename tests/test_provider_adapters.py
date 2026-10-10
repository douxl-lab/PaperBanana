"""Offline mocked protocol tests. These are not live model/account tests."""
import asyncio
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
from PIL import Image

from utils import generation_utils as gu
from utils.model_registry import ModelConfigurationError, resolve_model


@pytest.fixture
def rgba_png():
    output = io.BytesIO()
    Image.new("RGBA", (128, 96), (10, 20, 30, 80)).save(output, format="PNG")
    data = output.getvalue()
    return data, base64.b64encode(data).decode("ascii")


def run(coroutine):
    return asyncio.run(coroutine)


def text_choice(text="done", **kwargs):
    return NS(choices=[NS(message=NS(content=text, refusal=kwargs.get("refusal")),
                          finish_reason=kwargs.get("finish_reason", "stop"))], usage=None, _request_id="mock-request")


def gemini_image_response(data):
    return NS(candidates=[NS(content=NS(parts=[NS(inline_data=NS(data=data), thought=False)]), finish_reason="STOP")],
              usage_metadata=None, response_id="mock-gemini-request")


def test_image_input_forms_preserve_bytes_and_real_mime(rgba_png):
    data, encoded = rgba_png
    inputs = [{"type": "image", "source": {"type": "base64", "data": encoded, "media_type": "image/jpeg"}},
              {"type": "image", "image_base64": encoded},
              {"type": "image", "data": encoded, "mime_type": "image/jpeg"}]
    blocks = gu.normalize_contents(inputs)
    assert len(blocks) == 3
    assert all(block["data"] == data and block["mime_type"] == "image/png" for block in blocks)
    converted = gu._convert_to_openai_format(inputs)
    assert all(block["image_url"]["url"] == "data:image/png;base64," + encoded for block in converted)


@pytest.mark.parametrize("block", [{"type": "image"}, {"type": "file", "data": "unhandled"},
                                    {"type": "image_url", "image_url": {"url": "https://untrusted.invalid/a.png"}}])
def test_unknown_or_missing_content_is_never_silently_dropped(block):
    with pytest.raises(ModelConfigurationError):
        gu.normalize_contents([block])


def test_gemini_legacy_size_canonicalization_and_config_not_mutated(monkeypatch, rgba_png):
    data, encoded = rgba_png
    create = AsyncMock(return_value=gemini_image_response(data))
    monkeypatch.setattr(gu, "gemini_client", NS(aio=NS(models=NS(generate_content=create))))
    config = {"image_size": "4k", "aspect_ratio": "16:9", "max_output_tokens": 50000}
    before = dict(config)
    result = run(gu.generate_images("gemini-3.1-flash-image", "Draw the pipeline", config, max_attempts=1))
    params = create.call_args.kwargs
    assert params["config"].image_config.image_size == "4K"
    assert params["config"].max_output_tokens == 32768
    assert params["config"].http_options.timeout == 300000
    assert params["config"].http_options.retry_options.attempts == 1
    assert params["config"].response_modalities == ["TEXT", "IMAGE"]
    assert config == before
    assert result.samples[0].image_base64 == encoded
    assert result.samples[0].image_mime == "image/png"


def test_gemini_parts_join_one_candidate_not_multiple_samples(monkeypatch):
    create = AsyncMock(return_value=NS(candidates=[NS(content=NS(parts=[NS(text="first", thought=False), NS(text="second", thought=False)]), finish_reason="STOP")], usage_metadata=None))
    monkeypatch.setattr(gu, "gemini_client", NS(aio=NS(models=NS(generate_content=create))))
    result = run(gu.generate_text("gemini-3.8-flash", "hello", {"candidate_count": 1}, max_attempts=1))
    assert [sample.text for sample in result.samples] == ["firstsecond"]
    assert create.await_count == 1


def test_openai_edits_receive_original_reference_with_mime(monkeypatch, rgba_png):
    data, encoded = rgba_png
    edit = AsyncMock(return_value=NS(data=[NS(b64_json=encoded)], usage=None))
    generate = AsyncMock()
    monkeypatch.setattr(gu, "openai_client", NS(images=NS(edit=edit, generate=generate)))
    result = run(gu.generate_images("gpt-image-1.5", [{"type": "text", "text": "Only align the boxes"}, {"type": "image", "data": encoded}],
                {"aspect_ratio": "3:2", "image_size": "1k", "input_fidelity": "high"}, operation="image_edit", max_attempts=1))
    args = edit.call_args.kwargs
    assert args["image"][0][1:] == (data, "image/png")
    assert args["input_fidelity"] == "high"
    assert args["size"] == "1536x1024"
    assert "response_format" not in args
    assert result.samples[0].image_base64 == encoded
    assert generate.await_count == 0


def test_image_edit_without_reference_fails_before_request(monkeypatch):
    client = NS(images=NS(edit=AsyncMock(), generate=AsyncMock()))
    monkeypatch.setattr(gu, "openai_client", client)
    with pytest.raises(ModelConfigurationError, match="requires"):
        run(gu.generate_images("gpt-image-1.5", "Align things", {}, operation="image_edit"))
    assert client.images.edit.await_count == client.images.generate.await_count == 0


def test_openai_flexible_size_tracks_ratio_and_resolution(monkeypatch, rgba_png):
    _, encoded = rgba_png
    create = AsyncMock(return_value=NS(data=[NS(b64_json=encoded)]))
    monkeypatch.setattr(gu, "openai_client", NS(images=NS(generate=create)))
    run(gu.generate_images("gpt-image-2.5-sunburst", "plot diagram", {"aspect_ratio": "16:9", "image_size": "2k"}, max_attempts=1))
    assert create.call_args.kwargs["size"] == "2048x1152"


@pytest.mark.parametrize("model,config", [("gpt-image-1.5", {"aspect_ratio": "16:9"}),
                 ("gpt-image-1.5", {"image_size": "2K"}), ("gpt-image-2", {"input_fidelity": "high"}),
                 ("gpt-image-2", {"size": "1537x1024"}), ("gpt-image-2", {"aspect_ratio": "4:1"}),
                 ("gpt-image-2", {"output_format": "jpeg", "background": "transparent"})])
def test_unsupported_image_options_are_rejected_locally(model, config):
    with pytest.raises(ModelConfigurationError):
        run(gu.generate_images(model, "Draw a diagram", config))


def test_anthropic_token_mapping_and_all_text_blocks(monkeypatch):
    create = AsyncMock(return_value=NS(content=[NS(type="text", text="one"), NS(type="text", text="two")], stop_reason="end_turn", usage=None))
    monkeypatch.setattr(gu, "anthropic_client", NS(messages=NS(create=create)))
    output = run(gu.call_model_with_retry_async("claude-sonnet-4-20250514", "hello", {"max_completion_tokens": 50000}, max_attempts=1))
    assert output == ["onetwo"]
    assert create.call_args.kwargs["max_tokens"] == 8192


def test_openai_reasoning_omits_temperature_and_preserves_schema(monkeypatch):
    create = AsyncMock(return_value=NS(output_text='{"value": 2}', status="completed", output=[], usage=None))
    monkeypatch.setattr(gu, "openai_client", NS(responses=NS(create=create)))
    schema = {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"], "additionalProperties": False}
    result = run(gu.generate_text("gpt-6.1-sol", "Return a number", {"temperature": 1, "schema": schema}, max_attempts=1))
    args = create.call_args.kwargs
    assert "temperature" not in args
    assert args["reasoning"] == {"effort": "low"}
    assert args["text"]["format"]["strict"] is True
    assert result.samples[0].parsed == {"value": 2}
    assert result.warnings


@pytest.mark.parametrize("response,category", [(text_choice(refusal="blocked"), "refusal"),
                       (text_choice(finish_reason="length"), "incomplete"), (text_choice(text=""), "empty_response"),
                       (text_choice(text="not JSON"), "invalid_json")])
def test_refusal_truncation_empty_and_bad_json_are_not_error_strings(monkeypatch, response, category):
    create = AsyncMock(return_value=response)
    monkeypatch.setattr(gu, "openai_client", NS(chat=NS(completions=NS(create=create))))
    with pytest.raises(gu.GenerationResponseError) as failure:
        run(gu.call_model_with_retry_async("gpt-4.1", "hello", {"response_mime_type": "application/json"}, max_attempts=3, retry_delay=0))
    assert failure.value.category == category
    assert create.await_count == 1


def test_permanent_400_does_not_retry_or_expose_provider_message(monkeypatch):
    error = RuntimeError("sensitive payload must never be logged")
    error.status_code = 400
    create = AsyncMock(side_effect=error)
    monkeypatch.setattr(gu, "openai_client", NS(chat=NS(completions=NS(create=create))))
    with pytest.raises(gu.GenerationError) as failure:
        run(gu.call_model_with_retry_async("gpt-4.1", "hello", {}, max_attempts=5, retry_delay=0))
    assert create.await_count == 1
    assert "sensitive" not in str(failure.value)
    assert failure.value.status_code == 400


def test_transient_retry_is_bounded_and_uses_retry_after(monkeypatch):
    error = RuntimeError("unavailable")
    error.status_code = 429
    error.response = NS(headers={"retry-after": "0"})
    create = AsyncMock(side_effect=[error, error, text_choice()])
    monkeypatch.setattr(gu, "openai_client", NS(chat=NS(completions=NS(create=create))))
    result = run(gu.generate_text("gpt-4.1", "hello", {}, max_attempts=3, retry_delay=0))
    assert result.attempts == create.await_count == 3
    assert result.samples[0].text == "done"


def test_insufficient_quota_is_terminal(monkeypatch):
    error = RuntimeError("quota")
    error.status_code = 429
    error.body = {"error": {"code": "insufficient_quota"}}
    create = AsyncMock(side_effect=error)
    monkeypatch.setattr(gu, "openai_client", NS(chat=NS(completions=NS(create=create))))
    with pytest.raises(gu.GenerationError):
        run(gu.generate_text("gpt-4.1", "hello", {}, max_attempts=3, retry_delay=0))
    assert create.await_count == 1


def test_cancellation_is_not_caught_or_retried(monkeypatch):
    create = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(gu, "openai_client", NS(chat=NS(completions=NS(create=create))))
    with pytest.raises(asyncio.CancelledError):
        run(gu.generate_text("gpt-4.1", "hello", {}, max_attempts=3, retry_delay=0))
    assert create.await_count == 1


def test_deadline_cancels_transport_before_retry(monkeypatch):
    cancelled = []
    async def slow():
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
    with pytest.raises(gu.GenerationError, match="deadline"):
        run(gu._retry(slow, resolve_model("gpt-4.1"), timeout=.01, total_timeout=.02, max_attempts=3, retry_delay=1))
    assert cancelled == [True]


def test_original_rgba_png_is_not_reencoded(rgba_png):
    data, encoded = rgba_png
    assert gu.lossless_png_base64(encoded) == encoded
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0))[3] == 80


def test_configuration_selects_explicit_yaml_without_api_calls(tmp_path, monkeypatch):
    selected = tmp_path / "provider.yaml"
    selected.write_text("defaults: {main_model_name: gpt-4.1}\napi_keys: {openai_api_key: test-placeholder}\nprovider_settings: {openai: {base_url: 'https://example.invalid/v1'}}\n")
    for key in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.setenv(key, "")
    for client in ("gemini_client", "openai_client", "anthropic_client", "openrouter_client"):
        monkeypatch.setattr(gu, client, None)
    monkeypatch.setattr(gu, "_configuration_loaded", False)
    monkeypatch.setattr(gu, "model_config", {})
    monkeypatch.setattr(gu, "config_path", Path(tmp_path / "default.yaml"))
    constructor = Mock(return_value=NS())
    monkeypatch.setattr(gu, "AsyncOpenAI", constructor)
    cfg = NS(config_path=selected, main_model_name="gpt-4.1", main_model_provider="openai", model_profiles={})
    profiles = gu.validate_generation_configuration(cfg, needs_image=False)
    assert profiles["text"].model_id == "gpt-4.1"
    assert constructor.call_args.kwargs["base_url"] == "https://example.invalid/v1"
    assert constructor.call_args.kwargs["max_retries"] == 0
    assert not any(key in vars(profiles["text"]) for key in ("api_key", "api_keys"))


def test_openrouter_reference_image_is_not_dropped(monkeypatch, rgba_png):
    data, encoded = rgba_png
    message = NS(content=None, refusal=None, images=[{"image_url": {"url": "data:image/png;base64," + encoded}}])
    create = AsyncMock(return_value=NS(choices=[NS(message=message, finish_reason="stop")], usage=None))
    monkeypatch.setattr(gu, "openrouter_client", NS(chat=NS(completions=NS(create=create))))
    result = run(gu.generate_images("openrouter/google/gemini-3.1-flash-image", [
        {"type": "image", "data": encoded, "mime_type": "image/jpeg"}, {"type": "text", "text": "align boxes"}],
        {"aspect_ratio": "16:9", "image_size": "2k"}, operation="image_edit", max_attempts=1))
    args = create.call_args.kwargs
    assert args["model"] == "google/gemini-3.1-flash-image"
    assert args["messages"][1]["content"][0]["image_url"]["url"] == "data:image/png;base64," + encoded
    assert result.provider == "openrouter"


@pytest.mark.parametrize("finish,category", [("length", "incomplete"), ("content_filter", "refusal"),
                     ("incomplete", "incomplete"), ("tool_calls", "incomplete"), (None, "incomplete")])
def test_openrouter_terminal_image_states_reject_even_valid_inline_png(monkeypatch, rgba_png, finish, category):
    _, encoded = rgba_png
    message = NS(content=None, refusal=None, images=[{"image_url": {"url": "data:image/png;base64," + encoded}}])
    create = AsyncMock(return_value=NS(choices=[NS(message=message, finish_reason=finish)], usage=None))
    monkeypatch.setattr(gu, "openrouter_client", NS(chat=NS(completions=NS(create=create))))
    with pytest.raises(gu.GenerationResponseError) as failure:
        run(gu.generate_images("openrouter/google/gemini-3.1-flash-image", "align boxes",
                              {"aspect_ratio": "16:9", "image_size": "2k"}, max_attempts=3, retry_delay=0))
    assert failure.value.category == category
    assert create.await_count == 1


@pytest.mark.parametrize("model,effort,operation", [("gemini-3.8-flash", "high", "text"),
                        ("gemini-nano-banana-2.1", "minimal", "image_generate")])
def test_gemini_explicit_reasoning_effort_reaches_sdk(monkeypatch, rgba_png, model, effort, operation):
    data, _ = rgba_png
    image_operation = operation == "image_generate"
    response = gemini_image_response(data) if image_operation else NS(candidates=[NS(content=NS(parts=[NS(text="done", thought=False)]), finish_reason="STOP")])
    create = AsyncMock(return_value=response)
    monkeypatch.setattr(gu, "gemini_client", NS(aio=NS(models=NS(generate_content=create))))
    function = gu.generate_images if image_operation else gu.generate_text
    run(function(model, "draw boxes" if image_operation else "hello", {"reasoning_effort": effort}, max_attempts=1))
    assert create.call_args.kwargs["config"].thinking_config.thinking_level.value.lower() == effort


def test_gemini_legacy_thinking_config_is_preserved(monkeypatch):
    create = AsyncMock(return_value=NS(candidates=[NS(content=NS(parts=[NS(text="done", thought=False)]), finish_reason="STOP")]))
    monkeypatch.setattr(gu, "gemini_client", NS(aio=NS(models=NS(generate_content=create))))
    thinking = gu.types.ThinkingConfig(thinking_level="medium", include_thoughts=False)
    run(gu.generate_text("gemini-3.8-flash", "hello", {"thinking_config": thinking}, max_attempts=1))
    assert create.call_args.kwargs["config"].thinking_config == thinking


@pytest.mark.parametrize("provider", ["gemini", "openrouter"])
@pytest.mark.parametrize("precompiled", [False, True])
def test_image_profile_without_system_support_preserves_exact_wire_prompt(monkeypatch, rgba_png, provider, precompiled):
    data, encoded = rgba_png
    model = "gemini-3.1-flash-image" if provider == "gemini" else "openrouter/google/gemini-3.1-flash-image"
    overrides = {model: {"supports_system_prompt": False}}
    profile = resolve_model(model, operation="image_edit", overrides=overrides)
    if precompiled:
        from rfs.prompt_compiler import compile_prompt
        spec = json.loads((Path(__file__).resolve().parents[1] / "rfs/examples/encoder-pipeline.figure-spec.json").read_text())
        compiled = compile_prompt(spec, "Align boxes; preserve exact labels.", profile.compiler_capabilities(),
                                  candidate_id="0", stage="candidate")
        assert compiled.system_prompt is None
        prompt, system, expected = compiled.prompt, compiled.system_prompt, compiled.prompt
    else:
        prompt, system = "Align boxes; preserve exact labels.", "Keep scientific topology unchanged."
        expected = system + "\n\n" + prompt
    if provider == "gemini":
        create = AsyncMock(return_value=gemini_image_response(data))
        monkeypatch.setattr(gu, "gemini_client", NS(aio=NS(models=NS(generate_content=create))))
    else:
        message = NS(content=None, refusal=None, images=[{"image_url": {"url": "data:image/png;base64," + encoded}}])
        create = AsyncMock(return_value=NS(choices=[NS(message=message, finish_reason="stop")]))
        monkeypatch.setattr(gu, "openrouter_client", NS(chat=NS(completions=NS(create=create))))
    result = run(gu.generate_images(model, [{"type": "image", "data": encoded}, {"type": "text", "text": prompt}],
                    {"system_prompt": system}, operation="image_edit", overrides=overrides, max_attempts=1))
    args = create.call_args.kwargs
    if provider == "gemini":
        assert args["config"].system_instruction is None
        texts = [part.text for part in args["contents"] if part.text]
        assert [part.inline_data.data for part in args["contents"] if part.inline_data] == [data]
    else:
        assert [message["role"] for message in args["messages"]] == ["user"]
        texts = [block["text"] for block in args["messages"][0]["content"] if block["type"] == "text"]
        assert [block["image_url"]["url"] for block in args["messages"][0]["content"] if block["type"] == "image_url"] == ["data:image/png;base64," + encoded]
    assert texts == [expected]
    assert result.effective_prompt == expected
    assert result.effective_system_prompt is None
    assert result.samples[0].image_base64 == encoded


@pytest.mark.parametrize("provider", ["gemini", "openrouter"])
def test_text_profile_without_system_support_folds_into_actual_prompt(monkeypatch, provider):
    model = "gemini-3.8-flash" if provider == "gemini" else "openrouter/google/gemini-3.8-flash"
    overrides = {model: {"supports_system_prompt": False}}
    if provider == "gemini":
        create = AsyncMock(return_value=NS(candidates=[NS(content=NS(parts=[NS(text="done", thought=False)]), finish_reason="STOP")]))
        monkeypatch.setattr(gu, "gemini_client", NS(aio=NS(models=NS(generate_content=create))))
    else:
        create = AsyncMock(return_value=text_choice())
        monkeypatch.setattr(gu, "openrouter_client", NS(chat=NS(completions=NS(create=create))))
    result = run(gu.generate_text(model, "hello", {"system_prompt": "Keep labels exact."}, overrides=overrides, max_attempts=1))
    args = create.call_args.kwargs
    if provider == "gemini":
        assert args["config"].system_instruction is None
        wire_prompt = args["contents"][0].text
    else:
        assert [message["role"] for message in args["messages"]] == ["user"]
        wire_prompt = args["messages"][0]["content"][0]["text"]
    assert wire_prompt == result.effective_prompt == "Keep labels exact.\n\nhello"
    assert result.effective_system_prompt is None
