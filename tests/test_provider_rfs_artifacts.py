"""Provider-bound RFS rendering and artifact fidelity, entirely offline."""
import asyncio
import base64
import copy
import io
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from agents.visualizer_agent import VisualizerAgent
from utils.config import ExpConfig
from utils import generation_utils as gu


ROOT = Path(__file__).resolve().parents[1]


def fixture_image(format="PNG"):
    output = io.BytesIO()
    Image.new("RGBA" if format == "PNG" else "RGB", (128, 96),
              (20, 40, 60, 90) if format == "PNG" else (20, 40, 60)).save(output, format=format)
    return base64.b64encode(output.getvalue()).decode("ascii")


def load_spec(name="encoder-pipeline"):
    return json.loads((ROOT / "rfs" / "examples" / (name + ".figure-spec.json")).read_text())


def result_for(encoded, contents, config, **kwargs):
    meta = gu.image_metadata(encoded)
    return gu.GenerationResult([gu.GenerationSample(image_base64=encoded, image_mime=meta["mime_type"],
                              width=meta["width"], height=meta["height"])],
                              "openai", "gpt-image-1.5", effective_prompt=contents[0]["text"],
                              effective_system_prompt=config["system_prompt"] or None, options={"size": "1536x1024"})


def config(tmp_path, *, plot=False, rfs=True):
    return ExpConfig(dataset_name="PaperBananaBench", task_name="plot" if plot else "diagram",
                     render_mode="rfs" if rfs else "legacy", retrieval_setting="none", work_dir=tmp_path,
                     main_model_name="gpt-4.1", image_gen_model_name="gpt-image-1.5", max_critic_rounds=3,
                     main_model_provider="openai", image_gen_provider="openai", allow_generated_plot_code=True)


def test_rfs_stage_snapshot_original_png_alpha_and_exact_prompt(tmp_path, monkeypatch):
    original_spec = load_spec()
    current_spec = copy.deepcopy(original_spec)
    current_spec["design"]["layout"] = "A newer stylist layout must not overwrite the planner stage."
    encoded = fixture_image()
    async def image_call(model, contents, options, **kwargs):
        return result_for(encoded, contents, options, **kwargs)
    call = AsyncMock(side_effect=image_call)
    monkeypatch.setattr(gu, "generate_images", call)
    data = {"render_mode": "rfs", "candidate_id": 4, "figure_spec": current_spec,
            "target_diagram_desc0": "Three boxes. Render each scientific label exactly.",
            "rfs_stage_specs": {"target_diagram_desc0": original_spec},
            "additional_info": {"rounded_ratio": "3:2"}}
    output = asyncio.run(VisualizerAgent(exp_config=config(tmp_path)).process(data))
    record = output["rfs_artifacts"]["target_diagram_desc0"]
    assert record["figure_spec"] == original_spec
    assert record["candidate_id"] == "4"
    assert record["prompt"] == call.call_args.args[1][0]["text"]
    assert record["system_prompt"] is None
    assert record["image_base64"] == record["png_base64"] == encoded
    assert record["image_mime"] == "image/png"
    assert record["provenance"]["scientific_verification"] == "unverified"
    assert record["provenance"]["external_consumer_verified"] is False
    with Image.open(io.BytesIO(base64.b64decode(record["png_base64"]))) as image:
        assert image.mode == "RGBA" and image.getpixel((0, 0))[3] == 90
    with Image.open(io.BytesIO(base64.b64decode(output["target_diagram_desc0_base64_jpg"]))) as preview:
        assert preview.format == "JPEG"


def test_local_repair_passes_previous_original_not_lossy_preview(tmp_path, monkeypatch):
    spec = load_spec()
    original = fixture_image()
    seen = []
    async def image_call(model, contents, options, **kwargs):
        seen.append((contents, kwargs))
        return result_for(original, contents, options, **kwargs)
    monkeypatch.setattr(gu, "generate_images", image_call)
    identifiers = [item["id"] for key in ("nodes", "edges", "groups", "labels", "datasets") for item in spec[key]]
    repair = {"instruction": "Improve input spacing only.", "target_ids": ["input"], "allowed_changes": ["spacing"],
              "protected_ids": [value for value in identifiers if value != "input"], "protected_regions": []}
    data = {"render_mode": "rfs", "figure_spec": spec, "candidate_id": "2",
            "target_diagram_desc0": "Original diagram.", "target_diagram_critic_desc0": "Keep every scientific label. Adjust input spacing only.",
            "target_diagram_critic_suggestions0": "Improve input spacing.", "rfs_repair": repair,
            "additional_info": {"rounded_ratio": "3:2"}}
    output = asyncio.run(VisualizerAgent(exp_config=config(tmp_path)).process(data))
    assert len(seen) == 2
    repair_contents, kwargs = seen[1]
    assert kwargs["operation"] == "image_edit"
    assert repair_contents[1]["data"] == original
    assert repair_contents[1]["data"] != output["target_diagram_desc0_base64_jpg"]
    record = output["rfs_artifacts"]["target_diagram_critic_desc0"]
    assert record["stage"] == "local_repair"
    assert record["provenance"]["reference_stage"] == "target_diagram_desc0"


def test_local_repair_missing_original_never_regenerates(tmp_path, monkeypatch):
    spec = load_spec()
    identifiers = [item["id"] for key in ("nodes", "edges", "groups", "labels", "datasets") for item in spec[key]]
    repair = {"instruction": "Improve input spacing only.", "target_ids": ["input"], "allowed_changes": ["spacing"],
              "protected_ids": [value for value in identifiers if value != "input"], "protected_regions": []}
    call = AsyncMock()
    monkeypatch.setattr(gu, "generate_images", call)
    data = {"render_mode": "rfs", "figure_spec": spec, "target_diagram_desc0_base64_jpg": fixture_image("JPEG"),
            "target_diagram_critic_desc0": "Only improve spacing.", "rfs_repair": repair}
    with pytest.raises(gu.ModelConfigurationError, match="previous original"):
        asyncio.run(VisualizerAgent(exp_config=config(tmp_path)).process(data))
    assert call.await_count == 0


def test_existing_rounds_beyond_config_default_are_discovered(tmp_path, monkeypatch):
    encoded = fixture_image()
    call = AsyncMock(return_value=[encoded])
    monkeypatch.setattr(gu, "call_image_with_retry_async", call)
    data = {"target_diagram_critic_desc3": "A fifth-round description.", "additional_info": {"rounded_ratio": "3:2"}}
    result = asyncio.run(VisualizerAgent(exp_config=config(tmp_path, rfs=False)).process(data))
    assert result["target_diagram_critic_desc3_base64"] == encoded
    assert call.await_count == 1


def test_plot_code_repair_receives_exact_dataset_and_previous_code(tmp_path, monkeypatch):
    spec = load_spec("accuracy-comparison")
    previous_code = "import matplotlib.pyplot as plt\nplt.bar(['Baseline','Proposed'], [0.61,0.78])"
    revised_code = previous_code + "\nplt.tight_layout()"
    encoded = fixture_image("JPEG")
    call = AsyncMock(return_value=[revised_code])
    execution = AsyncMock(return_value=encoded)
    monkeypatch.setattr(gu, "call_model_with_retry_async", call)
    monkeypatch.setattr("agents.visualizer_agent.execute_plot_code_async", execution)
    data = {"render_mode": "rfs", "figure_spec": spec, "candidate_id": "0",
            "target_plot_desc0_base64_jpg": encoded, "target_plot_desc0_code": previous_code,
            "target_plot_critic_desc0": "Improve axis label spacing; preserve exact measurements.",
            "rfs_repair": {"instruction": "Improve axis label spacing.", "target_ids": ["input-data"],
                           "allowed_changes": ["axis label spacing"], "protected_ids": [], "protected_regions": []}}
    result = asyncio.run(VisualizerAgent(exp_config=config(tmp_path, plot=True)).process(data))
    actual_prompt = call.call_args.args[1][0]["text"]
    assert previous_code in actual_prompt
    assert "0.61" in actual_prompt and "0.78" in actual_prompt
    record = result["rfs_artifacts"]["target_plot_critic_desc0"]
    assert record["prompt"] == actual_prompt
    assert record["code"] == revised_code
    assert record["stage"] == "code_repair"
    assert record["image_mime"] == "image/jpeg"
    assert record["provenance"]["render_method"] == "generated_matplotlib"
    assert execution.call_args.kwargs["allow_execution"] is True
