"""Offline legacy UI/evaluation compatibility and real-adapter transport fixtures.

No paid requests, browser credentials, or real Colab runtime are used here.
"""
import ast
import asyncio
import base64
from dataclasses import fields
import importlib
from io import BytesIO
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from utils import generation_utils as generation
from utils import eval_toolkits as evaluation
from utils.config import ExpConfig
from utils.legacy_ui_refinement import (
    atomic_write, final_image, final_image_bytes, image_mime_type, new_run_id,
    pil_to_png_bytes, refine_image, sanitize_export, save_results_json, summarize_results,
)
from utils.model_registry import get_default_model, list_model_options, resolve_model

ROOT = Path(__file__).resolve().parents[1]


def pixels(fmt="PNG", mode="RGBA", color=(20, 30, 40, 0)):
    buffer = BytesIO()
    image = Image.new(mode, (16, 8), color)
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def encoded(raw):
    return base64.b64encode(raw).decode("ascii")


def result_record(raw=None, *, status="succeeded", round_index=5):
    raw = raw or pixels()
    stem = f"target_diagram_critic_desc{round_index}"
    preview = pixels("JPEG", "RGB", "white")
    return {"task_name": "diagram", "status": status, "eval_image_field": stem + "_base64_jpg",
            stem: "Exact selected design.", stem + "_base64_jpg": encoded(preview),
            stem + "_base64_png": encoded(raw),
            "rfs_artifacts": {stem: {"image_base64": encoded(raw), "image_mime": "image/png"}}}


@pytest.mark.parametrize("fmt,mode,color,mime", [
    ("PNG", "RGBA", (5, 10, 15, 0), "image/png"),
    ("JPEG", "RGB", "white", "image/jpeg"),
    ("WEBP", "RGBA", (5, 10, 15, 0), "image/webp"),
])
def test_real_mime_inspection(fmt, mode, color, mime):
    assert image_mime_type(pixels(fmt, mode, color)) == mime


def test_png_alpha_and_stage_binding_are_preserved():
    raw = pixels()
    record = result_record(raw)
    record["target_diagram_critic_desc9_base64_png"] = encoded(pixels(color=(255, 0, 0, 255)))
    assert final_image_bytes(record) == raw
    image = final_image(record)
    assert image.mode == "RGBA" and image.getpixel((0, 0))[3] == 0


@pytest.mark.parametrize("provider", ["gemini", "openai", "openrouter"])
@pytest.mark.parametrize("fmt,mode,color", [("PNG", "RGBA", (1, 2, 3, 0)), ("JPEG", "RGB", "white")])
def test_refinement_real_adapter_passes_reference_bytes(monkeypatch, provider, fmt, mode, color):
    """Run the shared UI helper all the way to each provider's request method."""
    source, output = pixels(fmt, mode, color), pixels()
    calls = []
    if provider == "gemini":
        model = get_default_model("image", "gemini")
        async def request(**kwargs):
            calls.append(kwargs)
            parts = [SimpleNamespace(inline_data=SimpleNamespace(data=output), thought=False)]
            return SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP", content=SimpleNamespace(parts=parts))],
                                   usage_metadata=None, response_id="fixture")
        client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=request)))
    elif provider == "openai":
        model = get_default_model("image", "openai")
        async def request(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(b64_json=encoded(output))], usage=None)
        async def forbidden_generation(**kwargs):
            raise AssertionError("Editing must not discard the reference and use generation")
        client = SimpleNamespace(images=SimpleNamespace(edit=request, generate=forbidden_generation))
    else:
        model = "openrouter/google/" + get_default_model("image", "gemini")
        async def request(**kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(images=[{"image_url": {"url": "data:image/png;base64," + encoded(output)}}])
            return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=request)))
    monkeypatch.setattr(generation, "_client", lambda profile: client)
    actual, message = asyncio.run(refine_image(source, "Increase label size; keep topology.", model_name=model,
                                               aspect_ratio="16:9", image_size="2K"))
    assert actual == output and "successfully" in message
    assert len(calls) == 1
    request = calls[0]
    assert request["model"] == resolve_model(model, operation="image_edit").model_id
    if provider == "gemini":
        inline = next(part.inline_data for part in request["contents"] if part.inline_data)
        assert inline.data == source and inline.mime_type == image_mime_type(source)
    elif provider == "openai":
        _, raw, mime = request["image"][0]
        assert raw == source and mime == image_mime_type(source)
    else:
        image_part = next(part for part in request["messages"][1]["content"] if part["type"] == "image_url")
        assert image_part["image_url"]["url"] == f"data:{image_mime_type(source)};base64,{encoded(source)}"


def test_refinement_failure_is_not_a_provider_fallback(monkeypatch):
    calls = []
    async def request(**kwargs):
        calls.append(kwargs)
        raise generation.GenerationError("Fixture provider rejection", status_code=401)
    monkeypatch.setattr(generation, "call_image_with_retry_async", request)
    actual, message = asyncio.run(refine_image(pixels(), "Keep all facts.", model_name=get_default_model("image")))
    assert actual is None and "failed" in message and len(calls) == 1


def test_refinement_rejects_invalid_reference_without_request(monkeypatch):
    async def request(**kwargs):
        raise AssertionError("Must validate pixels before making a request")
    monkeypatch.setattr(generation, "call_image_with_retry_async", request)
    actual, message = asyncio.run(refine_image(b"not an image", "Keep labels.", model_name=get_default_model("image")))
    assert actual is None and "failed" in message


def test_result_counts_do_not_count_failed_records_with_last_good_pixels():
    counts = summarize_results([result_record(), result_record(status="failed"), result_record(status="partial"),
                               {"status": "succeeded", "target_diagram_desc0_base64_jpg": "Error"}])
    assert counts == {"success": 1, "failed": 2, "incomplete": 1}


def test_atomic_json_unique_paths_and_no_credentials(tmp_path):
    value = [{"api_keys": {"openai_api_key": "fixture-secret"},
              "note": "provider sk-abcdefghijklmnopqrstuvwxyz01234567", "status": "failed"}]
    first, second = save_results_json(value, tmp_path), save_results_json(value, tmp_path)
    assert first != second
    stored = json.loads(first.read_text())
    assert "api_keys" not in stored[0] and "sk-" not in stored[0]["note"]
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob(".*.tmp"))


def evaluation_sample(tmp_path):
    reference = tmp_path / "data" / "PaperBananaBench" / "diagram" / "images" / "gt.png"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(pixels())
    sample = result_record()
    sample.update(content="The true source.", visual_intent="Exact labels", path_to_gt_image="images/gt.png")
    return sample


@pytest.mark.parametrize("provider", ["gemini", "openai", "anthropic", "openrouter"])
def test_eval_uses_nonempty_selected_model_and_real_mime(monkeypatch, tmp_path, provider):
    candidates = list_model_options("text", provider=provider) if provider != "openrouter" else []
    model = candidates[0] if candidates else "openrouter/google/" + get_default_model("text")
    calls = []
    async def router(**kwargs):
        calls.append(kwargs)
        return ['{"winner":"Model", "comparison_reasoning":"Fixture visual comparison."}']
    monkeypatch.setattr(evaluation, "call_model_with_retry_async", router)
    sample = evaluation_sample(tmp_path)
    actual = asyncio.run(evaluation.get_score_for_image_referenced(sample, model_name=model, work_dir=tmp_path))
    assert actual["evaluation_status"] == "completed" and actual["overall_outcome"] == "Model"
    assert actual["scientific_verification_status"] == "unverified"
    assert len(calls) == 4 and all(call["model_name"] == model for call in calls)
    images = [part for part in calls[0]["contents"] if part["type"] == "image"]
    assert images[0]["source"]["media_type"] == "image/png"
    assert images[1]["source"]["media_type"] == "image/png"


@pytest.mark.parametrize("winner", ["A", "Tie", 42, None, ["Model"]])
def test_malformed_evaluation_cannot_be_a_pass(monkeypatch, tmp_path, winner):
    async def router(**kwargs):
        return [json.dumps({"winner": winner, "comparison_reasoning": "Invalid fixture"})]
    monkeypatch.setattr(evaluation, "call_model_with_retry_async", router)
    sample = evaluation_sample(tmp_path)
    actual = asyncio.run(evaluation.get_score_for_image_referenced(sample, work_dir=tmp_path))
    assert actual["evaluation_status"] == "failed" and actual["overall_outcome"] == "Error"
    assert actual["evaluation_model_name"]


def test_evaluation_provider_failure_is_safe_and_not_a_tie(monkeypatch, tmp_path):
    async def router(**kwargs):
        raise RuntimeError("Fixture private details sk-abcdefghijklmnopqrstuvwxyz01234567")
    monkeypatch.setattr(evaluation, "call_model_with_retry_async", router)
    actual = asyncio.run(evaluation.get_score_for_image_referenced(evaluation_sample(tmp_path), work_dir=tmp_path))
    assert actual["evaluation_status"] == "failed" and actual["overall_outcome"] == "Error"
    assert "sk-" not in json.dumps(actual)


@pytest.mark.parametrize("reason", ["no_gt", "failed_generation", "missing_pixels", "missing_gt_file"])
def test_evaluation_skips_or_fails_without_fabricated_winner(monkeypatch, tmp_path, reason):
    async def router(**kwargs):
        raise AssertionError("No usable comparison should not make an API request")
    monkeypatch.setattr(evaluation, "call_model_with_retry_async", router)
    sample = evaluation_sample(tmp_path)
    if reason == "no_gt": sample.pop("path_to_gt_image")
    if reason == "failed_generation": sample["status"] = "failed"
    if reason == "missing_pixels": sample = {"task_name": "diagram", "path_to_gt_image": "images/gt.png"}
    if reason == "missing_gt_file": sample["path_to_gt_image"] = "images/missing.png"
    actual = asyncio.run(evaluation.get_score_for_image_referenced(sample, work_dir=tmp_path))
    assert actual["evaluation_status"] in {"skipped", "failed"}
    assert actual["overall_outcome"] not in {"Human", "Model", "Tie"}


def test_style_generator_import_does_not_create_an_api_client(monkeypatch):
    from google import genai
    def forbidden(*args, **kwargs):
        raise AssertionError("Style-guide import must be offline")
    monkeypatch.setattr(genai, "Client", forbidden)
    module = importlib.import_module("style_guides.generate_category_style_guide")
    importlib.reload(module)
    assert not hasattr(module, "client") and module.MODEL_NAME is None


def test_style_generator_routes_real_png_and_skips_failed_batches(monkeypatch, tmp_path):
    import style_guides.generate_category_style_guide as style
    path = tmp_path / "misnamed.jpg"
    path.write_bytes(pixels())
    calls = []
    async def router(**kwargs):
        calls.append(kwargs)
        return ["Aesthetic observations only."]
    monkeypatch.setattr(generation, "call_model_with_retry_async", router)
    monkeypatch.setattr(style, "BATCH_OUTPUT_DIR", tmp_path / "batches")
    report = asyncio.run(style.analyze_batch(asyncio.Semaphore(1), 0, [path], get_default_model("text")))
    assert "observations" in report
    assert calls[0]["contents"][0]["source"]["media_type"] == "image/png"
    async def failed(**kwargs): raise RuntimeError("fixture")
    monkeypatch.setattr(generation, "call_model_with_retry_async", failed)
    assert asyncio.run(style.analyze_batch(asyncio.Semaphore(1), 1, [path], get_default_model("text"))) is None
    assert not (tmp_path / "batches" / "batch_1.txt").exists()


def test_notebook_code_compiles_and_config_uses_current_parameters():
    notebook = json.loads(next((ROOT / "notebooks").glob("*.ipynb")).read_text())
    allowed = {field.name for field in fields(ExpConfig)}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code": continue
        text = "".join(cell["source"])
        text = "\n".join("pass" if line.lstrip().startswith(("!", "%")) else line for line in text.splitlines())
        compile(text, f"notebook:{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "ExpConfig":
                assert {keyword.arg for keyword in node.keywords} <= allowed
    selection_cell = "".join(notebook["cells"][5]["source"])
    assert "'main_model_name': MODEL_NAME" in selection_cell
    assert "'image_gen_model_name': IMAGE_MODEL_NAME" in selection_cell
    assert "list_model_options" in selection_cell and "genai.Client" not in selection_cell
    assert "STORE_API_KEY_IN_CONFIG" not in selection_cell
    stages_cell = "".join(notebook["cells"][15]["source"])
    assert "build_evolution_stages" in stages_cell and "Critic Round 2" not in stages_cell
    generation_cell = "".join(notebook["cells"][14]["source"])
    assert "NOTEBOOK_TEXT_PRICING_USD_PER_MTOK" in generation_cell
    assert "call_gemini_with_retry_async" not in generation_cell
    assert "processor.aclose()" in generation_cell


def test_legacy_openai_edit_with_supported_1k_options(monkeypatch):
    source, output, calls = pixels(), pixels(), []
    async def edit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(b64_json=encoded(output))], usage=None)
    monkeypatch.setattr(generation, "_client", lambda profile: SimpleNamespace(images=SimpleNamespace(edit=edit)))
    actual, message = asyncio.run(refine_image(source, "Keep every label.", model_name="gpt-image-1.5",
                                               aspect_ratio="3:2", image_size="1K"))
    assert actual == output and calls[0]["size"] == "1536x1024"


def test_unsupported_edit_options_fail_explicitly_before_request(monkeypatch):
    monkeypatch.setattr(generation, "_client", lambda profile: pytest.fail("Unsupported options must fail locally"))
    actual, message = asyncio.run(refine_image(pixels(), "Keep labels.", model_name="gpt-image-1.5",
                                               aspect_ratio="16:9", image_size="4K"))
    assert actual is None and "configuration error" in message and "2K/4K" in message


def test_gradio_real_build_callbacks_no_save_and_true_failure_counts(monkeypatch, tmp_path):
    import app
    application = app.build_app()
    try:
        passwords = [block for block in application.blocks.values() if getattr(block, "type", None) == "password"]
        assert len(passwords) == 4 and all(block.value == "" for block in passwords)
        options = [block for block in application.blocks.values() if getattr(block, "label", None) == "Image Editing Model"]
        assert options and options[0].value
        async def process(*args, **kwargs):
            return [result_record(), result_record(status="failed"), {"status": "failed", "task_name": "diagram"}]
        monkeypatch.setattr(app, "process_parallel_candidates", process)
        monkeypatch.setattr(app, "__file__", str(tmp_path / "app.py"))
        handler = next(function.fn for function in application.fns.values() if function.name == "run_generate")
        output = handler("True method.", "Exact caption.", "demo_full", "diagram", "none", 3, "16:9", 5,
                         get_default_model("text"), get_default_model("image"), "7-9cm", "No",
                         progress=lambda *args, **kwargs: None)
        assert "1 succeeded, 2 failed" in output[3] and output[2] is None
        assert not (tmp_path / "results").exists()
        assert output[0][0][0].mode == "RGBA"
    finally:
        application.close()


def test_gradio_refine_callback_preserves_rgba_source(monkeypatch, tmp_path):
    import app
    application = app.build_app()
    try:
        raw = pixels()
        image_path = tmp_path / "input.png"
        image_path.write_bytes(raw)
        from gradio.components.image import ImageData
        upload = next(block for block in application.blocks.values() if getattr(block, "label", None) == "Upload Image")
        preprocessed_path = upload.preprocess(ImageData(path=str(image_path), orig_name="input.png"))
        assert Path(preprocessed_path).read_bytes() == raw
        assert Image.open(preprocessed_path).mode == "RGBA"
        received = []
        async def refine(image_bytes, edit_prompt, **kwargs):
            received.append((image_bytes, kwargs))
            return raw, "Fixture edited image."
        monkeypatch.setattr(app, "refine_image_with_nanoviz", refine)
        monkeypatch.setattr(app, "__file__", str(tmp_path / "app.py"))
        handler = next(function.fn for function in application.fns.values() if function.name == "run_refine")
        before, after, download, _ = handler(preprocessed_path, "Keep facts.", "2K", "16:9", get_default_model("image"))
        assert received[0][0] == raw and received[0][1]["model_name"] == get_default_model("image")
        assert before.mode == after.mode == "RGBA"
        assert Image.open(download).mode == "RGBA"
    finally:
        application.close()


def test_viewers_use_dynamic_shared_results_and_eval_hot_reload(monkeypatch, tmp_path):
    import visualize.show_pipeline_evolution as pipeline_viewer
    import visualize.show_referenced_eval as eval_viewer
    sample = evaluation_sample(tmp_path)
    calls = []
    async def router(**kwargs):
        calls.append(kwargs)
        return ['{"winner":"Model", "comparison_reasoning":"Fixture comparison."}']
    monkeypatch.setattr(generation, "call_model_with_retry_async", router)
    model = get_default_model("text")
    actual = asyncio.run(eval_viewer.run_eval_on_sample(sample, model_name=model, work_dir=tmp_path))
    assert actual["eval_image_field"] == "target_diagram_critic_desc5_base64_jpg"
    assert actual["evaluation_status"] == "completed" and len(calls) == 4
    assert pipeline_viewer.detect_task_type({"task_name": "plot", "status": "failed"}) == "plot"
    assert eval_viewer.detect_task_type([{"task_name": "plot", "status": "failed"}]) == "plot"
    for viewer in (pipeline_viewer, eval_viewer):
        source = Path(viewer.__file__).read_text()
        assert "resolve_gt_image_path" in source
        assert "range(3)" not in source and "prompts.plots_eval_prompts" not in source


def test_ui_pipeline_closes_processor_on_success_and_exception(monkeypatch):
    import app
    closed = []
    class Agent:
        def __init__(self, **kwargs): pass
    class Processor:
        def __init__(self, **kwargs): pass
        async def process_queries_batch(self, *args, **kwargs):
            assert kwargs["max_concurrent"] > 0
            yield result_record()
            if fail[0]: raise RuntimeError("fixture")
        async def aclose(self): closed.append(True)
    for name in ("VanillaAgent", "PlannerAgent", "StylistAgent", "VisualizerAgent", "CriticAgent", "RetrieverAgent", "PolishAgent"):
        monkeypatch.setattr(app, name, Agent)
    monkeypatch.setattr(app, "PaperVizProcessor", Processor)
    fail = [False]
    assert asyncio.run(app.process_parallel_candidates([{}], retrieval_setting="none"))
    fail[0] = True
    with pytest.raises(RuntimeError):
        asyncio.run(app.process_parallel_candidates([{}], retrieval_setting="none"))
    assert closed == [True, True]


@pytest.mark.parametrize("module_name", ["app", "demo"])
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_ui_selected_model_overrides_yaml_provider(monkeypatch, tmp_path, module_name, provider):
    module = importlib.import_module(module_name)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    import yaml
    (config_dir / "model_config.yaml").write_text(yaml.safe_dump({"defaults": {
        "main_model_provider": "gemini", "image_gen_provider": "gemini", "retrieval_setting": "none"}}))
    monkeypatch.setattr(module, "__file__", str(tmp_path / (module_name + ".py")))
    text_model = list_model_options("text", provider=provider)[0]
    image_model = get_default_model("image", "openai")
    captured = []
    class Agent:
        def __init__(self, **kwargs): pass
    class Processor:
        def __init__(self, exp_config, **kwargs): captured.append(exp_config)
        async def process_queries_batch(self, *args, **kwargs):
            yield result_record()
        async def aclose(self): pass
    for name in ("VanillaAgent", "PlannerAgent", "StylistAgent", "VisualizerAgent", "CriticAgent", "RetrieverAgent", "PolishAgent"):
        monkeypatch.setattr(module, name, Agent)
    monkeypatch.setattr(module, "PaperVizProcessor", Processor)
    asyncio.run(module.process_parallel_candidates([{}], retrieval_setting="none", main_model_name=text_model,
                                                  image_gen_model_name=image_model))
    assert captured[0].main_model_provider == provider
    assert captured[0].image_gen_provider == "openai"


@pytest.mark.parametrize("mode", ["simple", "advanced"])
def test_notebook_generation_cell_runs_offline_with_result_fixture(monkeypatch, tmp_path, mode):
    """Execute Python generation/export logic; model results and display use fixtures.

    This is not a claim that Colab mounting, installing, or live API cells ran.
    """
    import utils.paperviz_processor as processor_module
    imported_agents = ("vanilla", "planner", "visualizer", "stylist", "critic", "retriever", "polish")
    class Agent:
        def __init__(self, **kwargs): pass
    for name in imported_agents:
        module = importlib.import_module(f"agents.{name}_agent")
        monkeypatch.setattr(module, name.title() + "Agent", Agent)
    closed = []
    class Processor:
        def __init__(self, exp_config, **kwargs):
            assert exp_config.main_model_provider == "gemini"
        async def process_queries_batch(self, samples, **kwargs):
            for index, sample in enumerate(samples):
                yield {**sample, **result_record(), "candidate_id": index}
        async def aclose(self): closed.append(True)
    monkeypatch.setattr(processor_module, "PaperVizProcessor", Processor)
    # Generation reload is irrelevant to fixture transport and would replace
    # monkeypatches owned by other offline tests; preserve every other reload.
    original_reload = importlib.reload
    monkeypatch.setattr(importlib, "reload", lambda module: module if module is generation else original_reload(module))
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    notebook = json.loads(next((ROOT / "notebooks").glob("*.ipynb")).read_text())
    source = "".join(notebook["cells"][14]["source"])
    # Colab has IPython by default. The CI venv has no notebook frontend; only
    # its display sink is supplied here, leaving execution and exports real.
    source = source.replace("from IPython.display import display, Markdown", "")
    namespace = dict(
        REPO_DIR=tmp_path, OUTPUT_DIR=output_dir, MODEL_NAME=get_default_model("text"),
        IMAGE_MODEL_NAME=get_default_model("image"), UI_MODE=mode, TASK_NAME="diagram",
        PIPELINE_MODE="demo_full", RETRIEVAL_SETTING="none", MAX_CRITIC_ROUNDS=5,
        MAX_CONCURRENT=2, NUM_CANDIDATES=2, SELECTED_CANDIDATE_INDEX=0,
        INPUT_MODE="paste_text_manual", method_text="True source.", visual_intent_text="Exact intent.",
        figure_caption_text="Exact caption.", resolved_input_path="", ASPECT_RATIO="16:9", RUN_REFINEMENT=False,
        REFINE_RESOLUTION="2K", REFINE_ASPECT_RATIO="16:9", REFINE_SOURCE="selected_result",
        EXTERNAL_REFINE_IMAGE_PATH="", REFINE_PROMPT="Keep facts.", EXPORT_4K=True, ZIP_EXPORT=False,
        AUTO_SELECT_BEST=False, display=lambda *args, **kwargs: None, Markdown=lambda value: value,
    )
    code = compile(source, "notebook-generation-fixture", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    asyncio.run(eval(code, namespace))
    assert closed == [True]
    assert namespace["selected_result"]["eval_image_field"] == "target_diagram_critic_desc5_base64_jpg"
    final_path = namespace.get("selected_final_deliverable_path") or namespace.get("final_deliverable_path")
    image = Image.open(final_path)
    assert image.mode == "RGBA" and image.size == (3840, 2160)
    plt.close("all")


@pytest.mark.parametrize("module_name", ["app", "demo"])
def test_ui_inputs_reach_real_pipeline_with_official_sdk_transport(monkeypatch, module_name):
    """No fake processor or agents: UI input reaches the actual legacy pipeline."""
    import httpx
    from openai import AsyncOpenAI
    module = importlib.import_module(module_name)
    source_pixels, requests = pixels(), []
    def transport(request):
        requests.append(request)
        assert request.url.path.endswith("/images/generations")
        payload = json.loads(request.content)
        assert payload["model"] == get_default_model("image", "openai")
        return httpx.Response(200, json={"created": 1, "data": [{"b64_json": encoded(source_pixels)}]})
    samples = module.create_sample_inputs("Exact true method.", "Keep the labels.", num_copies=1,
                                         max_critic_rounds=0, aspect_ratio="16:9", figure_size="7-9cm")
    assert isinstance(samples[0]["candidate_id"], str)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http_client:
            async with AsyncOpenAI(api_key="offline-fixture-key", base_url="https://fixture.invalid/v1",
                                   http_client=http_client, max_retries=0) as sdk:
                monkeypatch.setattr(generation, "_client", lambda profile: sdk)
                return await module.process_parallel_candidates(samples, exp_mode="vanilla", retrieval_setting="none",
                    main_model_name="gpt-4.1", image_gen_model_name=get_default_model("image", "openai"))
    results = asyncio.run(run())
    assert len(results) == 1 and results[0]["status"] == "succeeded", results
    assert final_image_bytes(results[0]) == source_pixels
    assert len(requests) == 1


def test_eval_does_not_transmit_path_traversal_or_symlink_escape(monkeypatch, tmp_path):
    outside = tmp_path / "private.png"
    outside.write_bytes(pixels())
    sample = evaluation_sample(tmp_path)
    async def forbidden(**kwargs):
        raise AssertionError("Outside-root pixels must never be sent to a provider")
    monkeypatch.setattr(evaluation, "call_model_with_retry_async", forbidden)
    for raw_path in (str(outside), "../../../private.png"):
        sample["path_to_gt_image"] = raw_path
        actual = asyncio.run(evaluation.get_score_for_image_referenced(sample, work_dir=tmp_path))
        assert actual["evaluation_status"] == "failed"
    link = tmp_path / "data" / "PaperBananaBench" / "diagram" / "images" / "escape.png"
    link.symlink_to(outside)
    sample["path_to_gt_image"] = "images/escape.png"
    actual = asyncio.run(evaluation.get_score_for_image_referenced(sample, work_dir=tmp_path))
    assert actual["evaluation_status"] == "failed" and "outside" in actual["evaluation_error"]


@pytest.mark.parametrize("module_name", ["app", "demo"])
def test_repeated_ui_runs_close_loop_bound_clients(monkeypatch, module_name):
    import httpx
    from openai import AsyncOpenAI
    module = importlib.import_module(module_name)
    clients, loops = [], []
    def handler(request):
        loops.append(asyncio.get_running_loop())
        return httpx.Response(200, json={"created": 1, "data": [{"b64_json": encoded(pixels())}]})
    def initialize():
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        sdk = AsyncOpenAI(api_key="offline-fixture-key", base_url="https://fixture.invalid/v1",
                          http_client=http_client, max_retries=0)
        clients.append(sdk)
        generation.openai_client = sdk
        return ["OpenAI"]
    monkeypatch.setattr(generation, "reinitialize_clients", initialize)
    samples = module.create_sample_inputs("Source", "Intent", num_copies=1, max_critic_rounds=0,
                                         aspect_ratio="16:9", figure_size="7-9cm")
    for _ in range(2):
        results = asyncio.run(module.process_parallel_candidates(samples, exp_mode="vanilla", retrieval_setting="none",
            main_model_name="gpt-4.1", image_gen_model_name=get_default_model("image", "openai")))
        assert results[0]["status"] == "succeeded"
        assert generation.openai_client is None
    assert len(clients) == 2 and all(client.is_closed() for client in clients)
    assert loops[0] is not loops[1]
