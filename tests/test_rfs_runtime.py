"""Offline fixtures for runtime behavior. These are not live-generation tests."""
import asyncio
import base64
import copy
from io import BytesIO
import json
import os
from pathlib import Path

import pytest
from PIL import Image

from utils.config import ExpConfig
from utils.dataset import ensure_dataset
from utils.paperviz_processor import PaperVizProcessor
from utils.plot_execution import PlotExecutionTimeout, execute_plot_code_async
from utils.runtime import CandidateCheckpoints, atomic_write_json, generation_binding, prepare_candidate


def image_b64():
    buffer = BytesIO()
    Image.new("RGB", (12, 8), "white").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def config(tmp_path, **kwargs):
    options = dict(dataset_name="Demo", task_name="diagram", exp_mode="demo_full", retrieval_setting="none",
                   work_dir=tmp_path, main_model_name="fixture-main", image_gen_model_name="fixture-image")
    options.update(kwargs)
    return ExpConfig(**options)


class FixtureAgent:
    def __init__(self, stage, events, *, fail=None, delay=0):
        self.stage, self.events, self.fail, self.delay = stage, events, fail, delay
        self.calls = 0
        self.closed = False

    async def process(self, data, **kwargs):
        self.calls += 1
        self.events.append((data.get("candidate_id"), self.stage, data.get("content")))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail and self.fail(data):
            raise RuntimeError("fixture failure")
        task = data.get("task_name", "diagram")
        if self.stage == "retriever":
            data.update(top10_references=[str(data["content"])], retrieved_examples=[{"nested": []}])
        elif self.stage == "planner":
            data[f"target_{task}_desc0"] = "Fixture design"
        elif self.stage == "stylist":
            data[f"target_{task}_stylist_desc0"] = "Fixture styled design"
        elif self.stage == "critic":
            index = data["current_critic_round"]
            data[f"target_{task}_critic_suggestions{index}"] = "Revise spacing."
            data[f"target_{task}_critic_desc{index}"] = "Fixture revised design"
        elif self.stage in ("visualizer", "vanilla", "polish"):
            if self.stage == "vanilla":
                data[f"vanilla_{task}_base64_jpg"] = image_b64()
            elif self.stage == "polish":
                data[f"polished_{task}_base64_jpg"] = image_b64()
            else:
                for key in list(data):
                    if key.startswith(f"target_{task}_") and "desc" in key and not key.endswith(("jpg", "code")):
                        data[key + "_base64_jpg"] = image_b64()
        return data

    async def aclose(self):
        self.closed = True


def processor(conf, **overrides):
    events = []
    agents = {name + "_agent": FixtureAgent(name, events) for name in
              ("vanilla", "planner", "visualizer", "stylist", "critic", "retriever", "polish")}
    agents.update(overrides)
    return PaperVizProcessor(conf, **agents), events


def source(content="source", **kwargs):
    return {"content": content, "visual_intent": "Fixture intent", **kwargs}


def collect(proc, records, **kwargs):
    async def run():
        return [item async for item in proc.process_queries_batch(records, do_eval=False, **kwargs)]
    return asyncio.run(run())


def test_config_models_precedence_and_timezone(tmp_path, monkeypatch):
    folder = tmp_path / "configs"
    folder.mkdir()
    (folder / "model_config.yaml").write_text("defaults:\n  main_model_name: yaml-main\n  image_gen_model_name: yaml-image\n")
    monkeypatch.setenv("TZ", "Etc/UTC")
    monkeypatch.setenv("MAIN_MODEL_NAME", "env-main")
    monkeypatch.setenv("IMAGE_GEN_MODEL_NAME", "env-image")
    assert config(tmp_path, main_model_name="", image_gen_model_name="").main_model_name == "env-main"
    assert config(tmp_path, main_model_name="explicit").main_model_name == "explicit"
    assert os.environ["TZ"] == "Etc/UTC"
    monkeypatch.delenv("MAIN_MODEL_NAME")
    assert config(tmp_path, main_model_name="").main_model_name == "yaml-main"
    assert not (tmp_path / "results").exists()


def test_numeric_config_precedence(tmp_path, monkeypatch):
    folder = tmp_path / "configs"
    folder.mkdir()
    (folder / "model_config.yaml").write_text("defaults:\n  max_concurrent: 7\n  max_critic_rounds: 5\n")
    assert config(tmp_path).max_concurrent == 7
    monkeypatch.setenv("PAPERBANANA_MAX_CONCURRENT", "3")
    assert config(tmp_path).max_concurrent == 3
    assert config(tmp_path, max_concurrent=2).max_concurrent == 2


@pytest.mark.parametrize("field,value", [("max_concurrent", 0), ("max_concurrent", 33), ("max_concurrent", True),
    ("max_critic_rounds", -1), ("max_critic_rounds", 21), ("candidate_timeout_seconds", float("inf")),
    ("plot_timeout_seconds", 0), ("temperature", float("nan")), ("render_mode", "unknown")])
def test_invalid_config(tmp_path, field, value):
    with pytest.raises(ValueError):
        config(tmp_path, **{field: value})


def test_explicit_config_path_aliases_and_invalid_yaml(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("defaults:\n  model_name: alias-main\n  image_model_name: alias-image\n")
    configured = config(tmp_path, main_model_name="", image_gen_model_name="", config_path=path)
    assert (configured.main_model_name, configured.image_gen_model_name) == ("alias-main", "alias-image")
    path.write_text("defaults: [wrong]")
    with pytest.raises(ValueError):
        config(tmp_path, config_path=path)
    path.write_text("api_keys: { secret: unterminated")
    with pytest.raises(ValueError, match="Invalid YAML configuration"):
        config(tmp_path, config_path=path)


def test_dataset_none_and_offline_never_download(tmp_path):
    calls = []
    downloader = lambda *args, **kwargs: calls.append((args, kwargs))
    assert not ensure_dataset("plot", work_dir=tmp_path, retrieval_setting="none", downloader=downloader)
    assert not ensure_dataset("diagram", work_dir=tmp_path, offline=True, downloader=downloader)
    assert not ensure_dataset("diagram", work_dir=tmp_path, policy="never", downloader=downloader)
    assert calls == []


def test_dataset_cache_and_explicit_failure(tmp_path):
    task = tmp_path / "data" / "PaperBananaBench" / "plot"
    task.mkdir(parents=True)
    (task / "images").mkdir()
    (task / "ref.json").write_text("[]")
    assert ensure_dataset("plot", work_dir=tmp_path, downloader=lambda *args, **kwargs: pytest.fail("downloaded cache"))
    def failed(*args, **kwargs):
        raise RuntimeError("secret must not be exported")
    with pytest.raises(RuntimeError, match="Reference dataset download failed") as error:
        ensure_dataset("diagram", work_dir=tmp_path, policy="download", downloader=failed)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("rounds", [0, 1, 3, 5])
def test_five_agent_order_and_dynamic_rounds(tmp_path, rounds):
    proc, events = processor(config(tmp_path, max_critic_rounds=rounds))
    result = collect(proc, [source()])[0]
    sequence = [stage for _, stage, _ in events]
    assert sequence == ["retriever", "planner", "stylist", "visualizer"] + ["critic", "visualizer"] * rounds
    assert result["status"] == "succeeded"
    expected = f"target_diagram_critic_desc{rounds - 1}_base64_jpg" if rounds else "target_diagram_stylist_desc0_base64_jpg"
    assert result["eval_image_field"] == expected


@pytest.mark.parametrize("mode", ["vanilla", "dev_planner", "dev_planner_stylist", "dev_planner_critic", "demo_planner_critic", "dev_full", "demo_full", "dev_polish", "dev_retriever"])
def test_all_supported_modes(tmp_path, mode):
    proc, _ = processor(config(tmp_path, exp_mode=mode, max_critic_rounds=0))
    assert collect(proc, [source()])[0]["status"] == "succeeded"


def test_grouped_retrieval_isolation_and_input_immutability(tmp_path):
    proc, _ = processor(config(tmp_path, max_critic_rounds=0))
    records = [source("paper A"), source("paper A"), source("paper B")]
    original = copy.deepcopy(records)
    output = sorted(collect(proc, records), key=lambda item: item["candidate_index"])
    assert proc.retriever_agent.calls == 2
    assert output[0]["top10_references"] == output[1]["top10_references"] == ["paper A"]
    assert output[2]["top10_references"] == ["paper B"]
    output[0]["retrieved_examples"][0]["nested"].append("mutation")
    assert output[1]["retrieved_examples"][0]["nested"] == []
    assert records == original


def test_failed_candidate_does_not_abort_batch(tmp_path):
    events = []
    proc, _ = processor(config(tmp_path, max_critic_rounds=0),
                         planner_agent=FixtureAgent("planner", events, fail=lambda data: data["content"] == "bad"))
    output = sorted(collect(proc, [source("bad"), source("good")]), key=lambda item: item["candidate_index"])
    assert [item["status"] for item in output] == ["failed", "succeeded"]
    assert output[0]["error"]["stage"] == "planner"
    assert [item["candidate_index"] for item in output] == [0, 1]


def test_critic_failure_retains_last_valid_image(tmp_path):
    proc, _ = processor(config(tmp_path), critic_agent=FixtureAgent("critic", [], fail=lambda data: True))
    output = collect(proc, [source()])[0]
    assert output["status"] == "failed"
    assert output["review_status"] == "failed"
    assert output[output["eval_image_field"]] == image_b64()


def test_candidate_timeout_and_no_leaked_tasks(tmp_path):
    async def run():
        proc, _ = processor(config(tmp_path, candidate_timeout_seconds=0.02), planner_agent=FixtureAgent("planner", [], delay=1))
        output = [item async for item in proc.process_queries_batch([source(), source("other")], do_eval=False)]
        assert all(item["status"] == "failed" and item["error"]["type"] == "TimeoutError" for item in output)
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and task.get_name().startswith("paperbanana-")]
    asyncio.run(run())


def test_generator_close_cancels_worker_pool(tmp_path):
    async def run():
        proc, _ = processor(config(tmp_path, max_critic_rounds=0, max_concurrent=2))
        iterator = proc.process_queries_batch([source(str(i)) for i in range(12)], do_eval=False)
        await anext(iterator)
        await iterator.aclose()
        assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and task.get_name().startswith("paperbanana-")]
        await proc.aclose()
        assert proc.planner_agent.closed
    asyncio.run(run())


def test_worker_concurrency_is_bounded(tmp_path):
    class TrackingAgent(FixtureAgent):
        active = maximum = 0
        async def process(self, data, **kwargs):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            try:
                await asyncio.sleep(0.02)
                return await super().process(data, **kwargs)
            finally:
                self.active -= 1
    agent = TrackingAgent("planner", [])
    proc, _ = processor(config(tmp_path, max_concurrent=2, max_critic_rounds=0), planner_agent=agent)
    assert len(collect(proc, [source(str(i)) for i in range(8)])) == 8
    assert agent.maximum == 2


def test_checkpoint_resume_and_input_configuration_binding(tmp_path):
    checkpoint = tmp_path / "checkpoints"
    conf = config(tmp_path, checkpoint_dir=checkpoint, max_critic_rounds=0)
    proc, _ = processor(conf)
    first = collect(proc, [source()])[0]
    resumed, _ = processor(config(tmp_path, checkpoint_dir=checkpoint, resume=True, max_critic_rounds=0))
    second = collect(resumed, [source()])[0]
    assert second["resumed"] is True
    assert resumed.planner_agent.calls == 0
    assert first["candidate_id"] == second["candidate_id"]
    changed, _ = processor(config(tmp_path, checkpoint_dir=checkpoint, resume=True, max_critic_rounds=1))
    assert "resumed" not in collect(changed, [source()])[0]
    assert changed.planner_agent.calls == 1
    modified, _ = processor(config(tmp_path, checkpoint_dir=checkpoint, resume=True, max_critic_rounds=0))
    assert "resumed" not in collect(modified, [source("different")])[0]


def test_fingerprint_binds_registry_schema_profiles_and_reference_images(tmp_path):
    conf = config(tmp_path)
    registry = tmp_path / "configs" / "model_registry.yaml"
    registry.parent.mkdir()
    registry.write_text("models: {}")
    schema = tmp_path / "rfs" / "schemas" / "figure.schema.json"
    schema.parent.mkdir(parents=True)
    schema.write_text("{}")
    image = tmp_path / "data" / "PaperBananaBench" / "diagram" / "images" / "ref.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"first")
    previous = generation_binding(conf)
    for path, value in [(registry, "models: {changed: {}}"), (schema, '{"changed":true}'), (image, b"second")]:
        path.write_bytes(value) if isinstance(value, bytes) else path.write_text(value)
        current = generation_binding(conf)
        assert current != previous
        previous = current
    conf.model_profiles = {"fixture-main": {"provider": "gemini"}}
    assert generation_binding(conf) != previous
    assert "api_keys" not in json.dumps(generation_binding(conf))


def test_corrupt_and_failed_checkpoint_not_reused(tmp_path):
    candidate = prepare_candidate(source(candidate_id="../untrusted"), 0, {})
    checkpoints = CandidateCheckpoints(tmp_path)
    assert checkpoints.path_for(candidate["candidate_id"]).parent == tmp_path
    checkpoints.save(candidate, {"candidate_id": candidate["candidate_id"], "status": "failed"})
    assert checkpoints.load(candidate) is None
    checkpoints.path_for(candidate["candidate_id"]).write_text("broken")
    assert checkpoints.load(candidate) is None


def test_atomic_write_keeps_previous_on_json_failure(tmp_path):
    path = tmp_path / "result.json"
    atomic_write_json(path, {"old": True})
    with pytest.raises(ValueError):
        atomic_write_json(path, {"new": float("nan")})
    assert json.loads(path.read_text()) == {"old": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_plot_execution_requires_permission():
    with pytest.raises(PermissionError):
        asyncio.run(execute_plot_code_async("value = 42"))


def test_bounded_plot_execution_and_hang_timeout(tmp_path):
    async def run():
        rendered = await execute_plot_code_async("import matplotlib.pyplot as plt\nplt.plot([1,2],[3,4])", dpi=40, timeout=15, allow_execution=True)
        assert rendered
        with Image.open(BytesIO(base64.b64decode(rendered))) as image:
            assert image.format == "JPEG"
        with pytest.raises(PlotExecutionTimeout):
            await execute_plot_code_async("while True: pass", timeout=0.1, allow_execution=True)
    asyncio.run(run())


def test_invalid_refinement_rolls_back_and_marks_incomplete(tmp_path):
    class InvalidRefinement(FixtureAgent):
        async def process(self, data, **kwargs):
            if "current_critic_round" in data:
                data[f"target_diagram_critic_desc{data['current_critic_round']}_base64_jpg"] = "Error"
                return data
            return await super().process(data, **kwargs)
    proc, _ = processor(config(tmp_path), visualizer_agent=InvalidRefinement("visualizer", []))
    output = collect(proc, [source()])[0]
    assert output["status"] == "failed"
    assert output["eval_image_field"] == "target_diagram_stylist_desc0_base64_jpg"
    assert output[output["eval_image_field"]] == image_b64()


def reference_pool(tmp_path):
    root = tmp_path / "data" / "PaperBananaBench" / "diagram"
    (root / "images").mkdir(parents=True)
    (root / "images" / "one.jpg").write_bytes(base64.b64decode(image_b64()))
    (root / "images" / "two.jpg").write_bytes(base64.b64decode(image_b64()))
    records = [{"id": "one", "visual_intent": "caption", "content": "source", "path_to_gt_image": "images/one.jpg"},
               {"id": "two", "visual_intent": "caption", "content": "source", "path_to_gt_image": "images/two.jpg"},
               {"id": "missing", "visual_intent": "caption", "content": "source", "path_to_gt_image": "images/missing.jpg"},
               {"id": "escape", "visual_intent": "caption", "content": "source", "path_to_gt_image": "../../outside.jpg"}]
    (root / "ref.json").write_text(json.dumps(records))
    return root


def test_retriever_validates_pool_filters_ids_and_propagates_provider(tmp_path, monkeypatch):
    from agents.retriever_agent import RetrieverAgent
    from unittest.mock import AsyncMock
    reference_pool(tmp_path)
    conf = config(tmp_path, main_model_provider="gemini", model_profiles={"fixture-main": {"base_profile": "gemini_text"}})
    agent = RetrieverAgent(exp_config=conf)
    generate = AsyncMock(return_value=['{"top10_diagrams":["one","missing","one","invented","two"]}'])
    monkeypatch.setattr("utils.generation_utils.call_model_with_retry_async", generate)
    result = asyncio.run(agent.process(source(), retrieval_setting="auto"))
    assert result["top10_references"] == ["one", "two"]
    assert len(result["retrieved_examples"]) == 2
    assert result["retrieval_status"]["omitted_count"] == 2
    assert generate.call_args.kwargs["provider"] == "gemini"
    assert generate.call_args.kwargs["overrides"] == conf.model_profiles


@pytest.mark.parametrize("response", ['[]', '{}', '{"top10_diagrams":"one"}', '{"top10_diagrams":[1]}'])
def test_retriever_invalid_response_fails_closed(tmp_path, response):
    from agents.retriever_agent import RetrieverAgent
    agent = RetrieverAgent(exp_config=config(tmp_path))
    with pytest.raises(ValueError):
        agent._parse_retrieval_result(response, "diagram")


def test_retrieval_skip_is_explicit_and_does_not_generate(tmp_path, monkeypatch):
    from agents.retriever_agent import RetrieverAgent
    from unittest.mock import AsyncMock
    generate = AsyncMock()
    monkeypatch.setattr("utils.generation_utils.call_model_with_retry_async", generate)
    agent = RetrieverAgent(exp_config=config(tmp_path))
    result = asyncio.run(agent.process(source(), retrieval_setting="auto"))
    assert result["retrieval_status"] == {"status": "skipped", "requested": "auto", "reason": "reference_dataset_unavailable"}
    assert result["top10_references"] == []
    generate.assert_not_called()


def test_plot_cancellation_kills_and_joins_child(monkeypatch):
    import utils.plot_execution as plotting
    processes = []
    original = plotting.asyncio.create_subprocess_exec
    async def capture(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(plotting.asyncio, "create_subprocess_exec", capture)
    async def run():
        task = asyncio.create_task(execute_plot_code_async("while True: pass", timeout=20, allow_execution=True))
        while not processes:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert processes[0].returncode is not None
    asyncio.run(run())


def test_failed_checkpoint_retries_then_success_resumes(tmp_path):
    checkpoint = tmp_path / "checkpoints"
    failed, _ = processor(config(tmp_path, checkpoint_dir=checkpoint, max_critic_rounds=0),
                          planner_agent=FixtureAgent("planner", [], fail=lambda data: True))
    assert collect(failed, [source(candidate_id="stable")])[0]["status"] == "failed"
    retried, _ = processor(config(tmp_path, checkpoint_dir=checkpoint, resume=True, max_critic_rounds=0))
    result = collect(retried, [source(candidate_id="stable")])[0]
    assert result["status"] == "succeeded" and not result.get("resumed")
    assert retried.planner_agent.calls == 1
    resumed, _ = processor(config(tmp_path, checkpoint_dir=checkpoint, resume=True, max_critic_rounds=0))
    assert collect(resumed, [source(candidate_id="stable")])[0]["resumed"]


def test_explicit_model_overrides_template_provider_without_wrong_routing(tmp_path, monkeypatch):
    from utils.model_registry import resolve_model
    folder = tmp_path / "configs"
    folder.mkdir()
    source_template = Path(__file__).resolve().parents[1] / "configs" / "model_config.template.yaml"
    (folder / "model_config.yaml").write_text(source_template.read_text())
    selected = config(tmp_path, main_model_name="gpt-6.1-sol", image_gen_model_name="gpt-image-2")
    assert selected.main_model_provider == selected.image_gen_provider == ""
    assert resolve_model(selected.main_model_name, selected.main_model_provider or None).provider == "openai"
    assert resolve_model(selected.image_gen_model_name, selected.image_gen_provider or None, "image").provider == "openai"
    monkeypatch.setenv("MAIN_MODEL_NAME", "gpt-6.1-sol")
    selected = config(tmp_path, main_model_name="")
    assert selected.main_model_provider == ""
    assert resolve_model(selected.main_model_name).provider == "openai"
    conflict = config(tmp_path, main_model_name="gpt-6.1-sol", main_model_provider="gemini")
    with pytest.raises(ValueError, match="belongs"):
        resolve_model(conflict.main_model_name, conflict.main_model_provider)


def test_rfs_library_plot_requires_explicit_permission_before_agents(tmp_path):
    conf = config(tmp_path, task_name="plot", render_mode="rfs")
    assert conf.allow_generated_plot_code is False
    assert config(tmp_path, task_name="plot").allow_generated_plot_code is True
    proc, events = processor(conf)
    result = collect(proc, [source()])[0]
    assert result["status"] == "failed"
    assert result["error"]["type"] == "PermissionError"
    assert events == []
    with pytest.raises(PermissionError):
        asyncio.run(proc.process_single_query(source(), do_eval=False))
    assert config(tmp_path, task_name="plot", render_mode="rfs", allow_generated_plot_code=True).allow_generated_plot_code is True


def test_legacy_default_permission_cannot_be_inherited_by_rfs_data(tmp_path):
    proc, events = processor(config(tmp_path, task_name="plot"))
    result = collect(proc, [source(render_mode="rfs")])[0]
    assert result["status"] == "failed" and result["error"]["type"] == "PermissionError"
    assert events == []


def test_raised_refinement_preserves_selection_and_failure_status(tmp_path):
    class FailedRefinement(FixtureAgent):
        async def process(self, data, **kwargs):
            if "current_critic_round" in data:
                raise RuntimeError("fixture rendering error")
            return await super().process(data, **kwargs)
    proc, _ = processor(config(tmp_path), visualizer_agent=FailedRefinement("visualizer", []))
    output = collect(proc, [source()])[0]
    assert output["status"] == "failed"
    assert output["eval_image_field"] == "target_diagram_stylist_desc0_base64_jpg"
    assert output[output["eval_image_field"]] == image_b64()
    assert output["error"]["stage"] == "visualizer"


def test_corrupted_success_checkpoint_causes_real_retry(tmp_path):
    checkpoint = tmp_path / "checkpoints"
    original, _ = processor(config(tmp_path, checkpoint_dir=checkpoint, max_critic_rounds=0))
    result = collect(original, [source(candidate_id="stable")])[0]
    path = CandidateCheckpoints(checkpoint).path_for("stable")
    record = json.loads(path.read_text())
    record["result"][result["eval_image_field"]] = "Error"
    path.write_text(json.dumps(record))
    retry, _ = processor(config(tmp_path, checkpoint_dir=checkpoint, resume=True, max_critic_rounds=0))
    output = collect(retry, [source(candidate_id="stable")])[0]
    assert retry.planner_agent.calls == 1
    assert output["status"] == "succeeded" and not output.get("resumed")


def test_valid_checksum_cannot_make_invalid_output_a_successful_checkpoint(tmp_path):
    from utils.runtime import stable_digest
    checkpoint = tmp_path / "checkpoints"
    proc, _ = processor(config(tmp_path, checkpoint_dir=checkpoint, max_critic_rounds=0))
    output = collect(proc, [source(candidate_id="stable")])[0]
    path = CandidateCheckpoints(checkpoint).path_for("stable")
    record = json.loads(path.read_text())
    record["result"][output["eval_image_field"]] = "Error"
    record["result_sha256"] = stable_digest(record["result"])
    path.write_text(json.dumps(record))
    retry, _ = processor(config(tmp_path, checkpoint_dir=checkpoint, resume=True, max_critic_rounds=0))
    assert not collect(retry, [source(candidate_id="stable")])[0].get("resumed")
    assert retry.planner_agent.calls == 1


@pytest.mark.parametrize("identifier", [0, 1, "stable"])
def test_candidate_identity_compatibility_no_private_fields(tmp_path, identifier):
    proc, _ = processor(config(tmp_path, max_critic_rounds=0))
    result = collect(proc, [source(candidate_id=identifier)])[0]
    assert result["candidate_id"] == str(identifier)
    assert result["candidate_index"] == 0
    assert not any(key.startswith("_runtime_") for key in result)


def test_agents_cannot_reassign_candidate_identity(tmp_path):
    class WrongIdentity(FixtureAgent):
        async def process(self, data, **kwargs):
            data["candidate_id"] = "wrong"
            data["candidate_index"] = 999
            return await super().process(data, **kwargs)
    proc, _ = processor(config(tmp_path, max_critic_rounds=0), planner_agent=WrongIdentity("planner", []))
    result = collect(proc, [source(candidate_id="right")])[0]
    assert result["candidate_id"] == "right" and result["candidate_index"] == 0


def test_timed_out_retrieval_calls_remain_bounded_and_joined(tmp_path):
    class SlowRetriever:
        active = maximum = 0
        async def process(self, data, **kwargs):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            try:
                await asyncio.sleep(5)
                return data
            finally:
                self.active -= 1
    agent = SlowRetriever()
    proc, _ = processor(config(tmp_path, max_concurrent=2, candidate_timeout_seconds=0.02), retriever_agent=agent)
    output = collect(proc, [source(f"distinct {index}") for index in range(12)])
    assert len(output) == 12 and all(item["status"] == "failed" for item in output)
    assert agent.maximum <= 2 and agent.active == 0


def test_shared_retrieval_cancellation_keeps_remaining_waiters(tmp_path):
    class SlowRetriever(FixtureAgent):
        async def process(self, data, **kwargs):
            await asyncio.sleep(0.015)
            return await super().process(data, **kwargs)
    agent = SlowRetriever("retriever", [])
    proc, _ = processor(config(tmp_path, max_concurrent=2, candidate_timeout_seconds=0.1, max_critic_rounds=0), retriever_agent=agent)
    output = collect(proc, [source("same") for _ in range(4)])
    assert agent.calls == 1
    assert all(item["status"] == "succeeded" for item in output)


@pytest.mark.parametrize("task,kind", [("plot", "diagram"), ("diagram", "plot")])
def test_rfs_task_spec_mismatch_fails_before_retrieval(tmp_path, task, kind):
    proc, events = processor(config(tmp_path, task_name=task, render_mode="rfs", allow_generated_plot_code=True))
    output = collect(proc, [source(figure_spec={"kind": kind})])[0]
    assert output["status"] == "failed" and output["error"]["type"] == "ValueError"
    assert events == []
