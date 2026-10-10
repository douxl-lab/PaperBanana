"""CLI integration with local fixtures only, never paid model calls."""
import asyncio
import base64
import copy
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest
from PIL import Image

from skill import run as cli
import main as benchmark
from utils.runtime import write_or_reuse_rfs_bundle

ROOT = Path(__file__).resolve().parents[1]


def image_b64(color="white", mode="RGB"):
    stream = BytesIO()
    Image.new(mode, (16, 10), color).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


class FixtureProcessor:
    def __init__(self, results, capture=None):
        self.results = results
        self.capture = capture if capture is not None else {}
        self.closed = False
    async def process_queries_batch(self, records, **kwargs):
        self.capture.update(records=copy.deepcopy(records), batch_options=kwargs)
        for result in self.results:
            yield copy.deepcopy(result)
    async def aclose(self):
        self.closed = True


def args(tmp_path, *extra):
    return cli.build_parser().parse_args(["--content", "Fixture source", "--caption", "Fixture figure", "--num-candidates", "1",
        "--retrieval-setting", "none", "--output", str(tmp_path / "output.png"), *extra])


def success(index=0, task="diagram", color="white"):
    key = f"target_{task}_stylist_desc0_base64_jpg"
    return {"candidate_id": f"fixture-{index}", "candidate_index": index, "status": "succeeded",
            "task_name": task, "eval_image_field": key, key: image_b64(color)}


def patched_pipeline(monkeypatch, records):
    capture = {}
    fixture = FixtureProcessor(records, capture)
    def build(config):
        capture["config"] = config
        return fixture
    monkeypatch.setattr(cli, "build_processor", build)
    monkeypatch.setattr(cli, "validate_generation_configuration", lambda config: None)
    return fixture, capture


def test_help_has_no_key_or_dataset_requirement():
    result = subprocess.run([sys.executable, str(ROOT / "skill" / "run.py"), "--help"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "--render-mode" in result.stdout
    assert "--offline" in result.stdout
    assert "--resume" in result.stdout
    assert "--allow-generated-plot-code" in result.stdout


@pytest.mark.parametrize("extras", [["--num-candidates", "0"], ["--max-concurrent", "0"], ["--max-critic-rounds", "-1"],
    ["--candidate-timeout-seconds", "0"], ["--rfs-file", "missing.json"], ["--offline", "--dataset-policy", "download"]])
def test_invalid_arguments_precede_download_and_model_preflight(tmp_path, monkeypatch, extras):
    downloader, preflight = Mock(), Mock()
    monkeypatch.setattr(cli, "ensure_dataset", downloader)
    monkeypatch.setattr(cli, "validate_generation_configuration", preflight)
    with pytest.raises(ValueError):
        asyncio.run(cli.run(args(tmp_path, "--retrieval-setting", "auto", *extras)))
    downloader.assert_not_called()
    preflight.assert_not_called()


def test_missing_content_precedes_download(tmp_path, monkeypatch):
    arguments = args(tmp_path)
    arguments.content = ""
    downloader = Mock()
    monkeypatch.setattr(cli, "ensure_dataset", downloader)
    with pytest.raises(ValueError, match="content"):
        asyncio.run(cli.run(arguments))
    downloader.assert_not_called()


def test_model_configuration_precedes_dataset_download(tmp_path, monkeypatch):
    downloader = Mock()
    def unavailable(config):
        raise ValueError("Missing provider configuration")
    monkeypatch.setattr(cli, "ensure_dataset", downloader)
    monkeypatch.setattr(cli, "validate_generation_configuration", unavailable)
    with pytest.raises(ValueError, match="provider"):
        asyncio.run(cli.run(args(tmp_path, "--retrieval-setting", "auto")))
    downloader.assert_not_called()


def test_plot_task_content_rounds_concurrency_propagate(tmp_path, monkeypatch):
    fixture, capture = patched_pipeline(monkeypatch, [success(task="plot")])
    arguments = args(tmp_path, "--task", "plot", "--max-critic-rounds", "5", "--max-concurrent", "2")
    arguments.content = '[{"group":"A","score":1},{"group":"B","score":2}]'
    assert asyncio.run(cli.run(arguments)) == 0
    assert capture["config"].task_name == "plot"
    assert capture["config"].max_critic_rounds == 5
    assert capture["records"][0]["content"][1]["score"] == 2
    assert capture["batch_options"] == {"max_concurrent": 2, "do_eval": False}
    assert fixture.closed
    assert (tmp_path / "output.png").is_file()


def test_retrieval_none_does_not_call_dataset(tmp_path, monkeypatch):
    patched_pipeline(monkeypatch, [success()])
    downloader = Mock(side_effect=AssertionError("must not download"))
    monkeypatch.setattr(cli, "ensure_dataset", downloader)
    assert asyncio.run(cli.run(args(tmp_path, "--offline"))) == 0
    downloader.assert_not_called()


def test_offline_missing_required_data_blocks(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "validate_generation_configuration", lambda config: None)
    monkeypatch.setattr(cli, "ensure_dataset", lambda *a, **kw: False)
    constructor = Mock()
    monkeypatch.setattr(cli, "build_processor", constructor)
    with pytest.raises(RuntimeError, match="retrieval-setting none"):
        asyncio.run(cli.run(args(tmp_path, "--retrieval-setting", "auto", "--offline")))
    constructor.assert_not_called()


def test_output_names_use_input_candidate_order(tmp_path, monkeypatch):
    patched_pipeline(monkeypatch, [success(1, color="blue"), success(0, color="red")])
    assert asyncio.run(cli.run(args(tmp_path, "--num-candidates", "2"))) == 0
    with Image.open(tmp_path / "output_0.png") as image:
        assert image.getpixel((0, 0)) == (255, 0, 0)
    with Image.open(tmp_path / "output_1.png") as image:
        assert image.getpixel((0, 0)) == (0, 0, 255)
    records = json.loads((tmp_path / "output.png.results.json").read_text())
    assert [record["candidate_index"] for record in records] == [0, 1]


def test_partial_failure_returns_nonzero_and_retains_success(tmp_path, monkeypatch):
    patched_pipeline(monkeypatch, [{"candidate_id": "fixture-0", "candidate_index": 0, "status": "failed", "error": {"type": "FixtureFailure"}}, success(1)])
    assert asyncio.run(cli.run(args(tmp_path, "--num-candidates", "2"))) == 2
    assert not (tmp_path / "output_0.png").exists()
    assert (tmp_path / "output_1.png").exists()
    records = json.loads((tmp_path / "output.png.results.json").read_text())
    assert [record["status"] for record in records] == ["failed", "succeeded"]


@pytest.mark.parametrize("results", [[], [{"candidate_id": "fixture-0", "candidate_index": 0, "status": "failed"}],
    [{"candidate_id": "fixture-0", "candidate_index": 0, "status": "succeeded", "target_diagram_desc0_base64_jpg": "Error"}]])
def test_empty_or_invalid_output_returns_nonzero(tmp_path, monkeypatch, results):
    fixture, _ = patched_pipeline(monkeypatch, results)
    assert asyncio.run(cli.run(args(tmp_path))) == 2
    assert not (tmp_path / "output.png").exists()
    assert fixture.closed


def test_final_selection_respects_task_eval_and_arbitrary_rounds():
    record = {"task_name": "plot", "eval_image_field": "target_plot_critic_desc5_base64_jpg",
              "target_plot_critic_desc5_base64_jpg": "selected", "target_plot_critic_desc8_base64_jpg": "later",
              "target_diagram_critic_desc99_base64_jpg": "wrong task"}
    assert cli.extract_final_image_b64(record, "demo_full") == "selected"
    record.pop("eval_image_field")
    assert cli.extract_final_image_b64(record, "demo_full", "plot") == "later"


def test_main_default_mode_is_supported():
    arguments = benchmark.build_parser().parse_args([])
    assert arguments.exp_mode == "dev_full"
    assert arguments.max_concurrent is None
    assert benchmark.build_parser().parse_args(["--exp_mode", "dev"]).exp_mode == "dev"


def test_rfs_plot_requires_trust_before_model_or_download(tmp_path, monkeypatch):
    preflight, downloader = Mock(), Mock()
    monkeypatch.setattr(cli, "validate_generation_configuration", preflight)
    monkeypatch.setattr(cli, "ensure_dataset", downloader)
    arguments = args(tmp_path, "--task", "plot", "--render-mode", "rfs")
    arguments.content = '[{"score":1}]'
    with pytest.raises(ValueError, match="allow-generated-plot-code"):
        asyncio.run(cli.run(arguments))
    preflight.assert_not_called()
    downloader.assert_not_called()


def rfs_result():
    from rfs.contract import canonical_json, sha256_text
    spec = json.loads((ROOT / "rfs" / "examples" / "encoder-pipeline.figure-spec.json").read_text())
    output = success()
    key = "target_diagram_stylist_desc0"
    output.update(figure_spec=spec, rfs_design_description="Fixture design", rfs_artifacts={key: {
        "candidate_id": "fixture-0", "stage": "stylist", "provider": "fixture", "model": "fixture-image",
        "prompt": "Actual fixture prompt", "system_prompt": "Actual fixture system prompt", "design_description": "Fixture design",
        "figure_spec": spec, "figure_spec_sha256": sha256_text(canonical_json(spec)),
        "image_base64": image_b64((0, 0, 0, 0), "RGBA"), "image_mime": "image/png", "options": {}}})
    return output


def test_rfs_cli_exports_lossless_png_and_exchange_files(tmp_path, monkeypatch):
    result = rfs_result()
    patched_pipeline(monkeypatch, [result])
    assert asyncio.run(cli.run(args(tmp_path, "--render-mode", "rfs"))) == 0
    bundle = tmp_path / "output_rfs" / "fixture-0"
    from rfs import verify_bundle
    manifest = verify_bundle(bundle)
    assert manifest["status"] == "succeeded"
    assert {"image", "prompt", "design", "figure_spec"} <= manifest["files"].keys()
    assert (bundle / "prompt.txt").read_text() == "Actual fixture prompt"
    with Image.open(tmp_path / "output.png") as image:
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0))[3] == 0


def test_rfs_resume_reuses_only_matching_verified_bundle(tmp_path):
    result = rfs_result()
    write_or_reuse_rfs_bundle(result, tmp_path / "bundle")
    result["resumed"] = True
    assert write_or_reuse_rfs_bundle(result, tmp_path / "bundle")["candidate_id"] == "fixture-0"
    changed = copy.deepcopy(result)
    changed["rfs_artifacts"]["target_diagram_stylist_desc0"]["prompt"] = "Changed prompt"
    with pytest.raises(ValueError, match="prompt"):
        write_or_reuse_rfs_bundle(changed, tmp_path / "bundle")
    (tmp_path / "bundle" / "image.png").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        write_or_reuse_rfs_bundle(result, tmp_path / "bundle")


def test_main_missing_input_precedes_preflight(tmp_path, monkeypatch):
    import utils.generation_utils
    preflight = Mock()
    monkeypatch.setattr(utils.generation_utils, "validate_generation_configuration", preflight)
    with pytest.raises(OSError):
        asyncio.run(benchmark.main(["--input-file", str(tmp_path / "missing.json")]))
    preflight.assert_not_called()


def test_cli_default_flags_honor_yaml_and_environment_runtime_config(tmp_path, monkeypatch):
    path = tmp_path / "configuration.yaml"
    path.write_text("defaults:\n  max_concurrent: 7\n  max_critic_rounds: 5\n  candidate_timeout_seconds: 17\n")
    fixture, capture = patched_pipeline(monkeypatch, [success()])
    monkeypatch.setenv("PAPERBANANA_MAX_CONCURRENT", "2")
    assert asyncio.run(cli.run(args(tmp_path, "--config", str(path)))) == 0
    assert capture["config"].max_concurrent == 2
    assert capture["config"].max_critic_rounds == 5
    assert capture["config"].candidate_timeout_seconds == 17
    assert capture["records"][0]["max_critic_rounds"] == 5
    assert capture["batch_options"]["max_concurrent"] == 2


def test_main_invalid_record_precedes_preflight(tmp_path, monkeypatch):
    import utils.generation_utils
    source = tmp_path / "source.json"
    source.write_text('[{"content":"source","visual_intent":"caption","max_critic_rounds":-1}]')
    preflight = Mock()
    monkeypatch.setattr(utils.generation_utils, "validate_generation_configuration", preflight)
    with pytest.raises(ValueError, match="critic rounds"):
        asyncio.run(benchmark.main(["--input-file", str(source)]))
    preflight.assert_not_called()


@pytest.mark.parametrize("identifier", ["../escaped", "/absolute", "C:\\escape", "..\\escape", "", "../../nested/path"])
def test_candidate_bundle_path_is_encoded_inside_root(tmp_path, identifier):
    from utils.runtime import candidate_artifact_directory
    root = tmp_path / "bundles"
    directory = candidate_artifact_directory(root, identifier)
    assert directory.parent == root
    assert directory.resolve().is_relative_to(root.resolve())
    assert "/" not in directory.name and "\\" not in directory.name


@pytest.mark.parametrize("content", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}', '[{"score":}]', "score=3"])
def test_rfs_plot_rejects_repairs_and_nonfinite_values_before_api(tmp_path, monkeypatch, content):
    preflight, downloader = Mock(), Mock()
    monkeypatch.setattr(cli, "validate_generation_configuration", preflight)
    monkeypatch.setattr(cli, "ensure_dataset", downloader)
    arguments = args(tmp_path, "--task", "plot", "--render-mode", "rfs", "--allow-generated-plot-code")
    arguments.content = content
    with pytest.raises(ValueError):
        asyncio.run(cli.run(arguments))
    preflight.assert_not_called()
    downloader.assert_not_called()


def test_rfs_plot_strict_parse_preserves_exact_values_and_original_text():
    from utils.runtime import normalize_scientific_content
    text = '[{"label":"A","score":0.123456789},{"label":"B","score":-2}]'
    parsed = normalize_scientific_content(text, "plot", "rfs")
    assert parsed == json.loads(text)
    assert parsed[0]["score"] == 0.123456789
    assert normalize_scientific_content(text, "diagram", "rfs") == text


def test_benchmark_rfs_rejects_duplicate_input_keys_before_preflight(tmp_path, monkeypatch):
    import utils.generation_utils
    source = tmp_path / "source.json"
    source.write_text('[{"content":"first","content":"second","visual_intent":"caption"}]')
    preflight = Mock()
    monkeypatch.setattr(utils.generation_utils, "validate_generation_configuration", preflight)
    with pytest.raises(ValueError, match="duplicate"):
        asyncio.run(benchmark.main(["--input-file", str(source), "--render-mode", "rfs"]))
    preflight.assert_not_called()
