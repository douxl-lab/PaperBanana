"""Offline cross-layer acceptance with real agents, processor and SDK request shapes.

Only the SDK-client boundary is controlled. Images are protocol fixtures, not
an assertion of live-provider quality or scientific correctness.
"""
import asyncio
import base64
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from agents.critic_agent import CriticAgent
from agents.planner_agent import PlannerAgent
from agents.polish_agent import PolishAgent
from agents.retriever_agent import RetrieverAgent
from agents.stylist_agent import StylistAgent
from agents.vanilla_agent import VanillaAgent
from agents.visualizer_agent import VisualizerAgent
from rfs import export_bundle, verify_bundle
from rfs.contract import scientific_fingerprint
from utils.config import ExpConfig
from utils.paperviz_processor import PaperVizProcessor

ROOT = Path(__file__).parents[1]


def png_fixture(color, *, repair=False):
    image = Image.new("RGBA", (64, 48), color)
    if repair:
        # The requested target is left of the explicitly protected right-half.
        image.paste((180, 200, 240, 190), (0, 0, 24, 24))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class ControlledGeminiSDK:
    def __init__(self, texts, images):
        self.texts = list(texts)
        self.images = list(images)
        self.requests = []
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self.generate_content))

    async def generate_content(self, *, model, contents, config):
        is_image = "IMAGE" in [str(getattr(value, "value", value)) for value in config.response_modalities or []]
        request = {"model": model, "parts": list(contents), "config": config, "is_image": is_image}
        self.requests.append(request)
        if is_image:
            assert self.images, "Unexpected paid-image request would have occurred"
            part = SimpleNamespace(text=None, inline_data=SimpleNamespace(data=self.images.pop(0), mime_type="image/png"), thought=False)
        else:
            assert self.texts, "Unexpected text request would have occurred"
            part = SimpleNamespace(text=self.texts.pop(0), inline_data=None, thought=False)
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]), finish_reason="STOP")],
                               usage_metadata=None, response_id=f"controlled-{len(self.requests)}")


def create_processor(tmp_path, *, rounds=2):
    reference_dir = tmp_path / "data/PaperBananaBench/diagram"
    reference_dir.mkdir(parents=True)
    reference = png_fixture((200, 210, 220, 255))
    (reference_dir / "reference.png").write_bytes(reference)
    (reference_dir / "agent_selected_12.json").write_text(json.dumps([{
        "id": "local-reference", "content": "An illustrative local reference.",
        "visual_intent": "Use readable labeled boxes.", "path_to_gt_image": "reference.png",
    }]))
    config = ExpConfig(dataset_name="PaperBananaBench", task_name="diagram", exp_mode="demo_full",
                       work_dir=tmp_path, retrieval_setting="manual", render_mode="rfs", max_critic_rounds=rounds,
                       main_model_name="gemini-3.8-flash", image_gen_model_name="gemini-nano-banana-2.1",
                       main_model_provider="gemini", image_gen_provider="gemini", candidate_timeout_seconds=30,
                       max_concurrent=1, style_guide_path=ROOT / "style_guides/neurips2025_diagram_style_guide.md")
    processor = PaperVizProcessor(config, VanillaAgent(exp_config=config), PlannerAgent(exp_config=config),
                                 VisualizerAgent(exp_config=config), StylistAgent(exp_config=config), CriticAgent(exp_config=config),
                                 RetrieverAgent(exp_config=config), PolishAgent(exp_config=config))
    return processor, reference


def specification():
    return json.loads((ROOT / "rfs/examples/encoder-pipeline.figure-spec.json").read_text())


def query(spec):
    return {"candidate_id": "offline-candidate", "content": spec["sources"][0]["content"],
            "visual_intent": "Show the encoder pipeline.", "figure_spec": copy.deepcopy(spec),
            "additional_info": {"rounded_ratio": "4:3", "image_size": "1k"}}


def envelopes(spec):
    styled = copy.deepcopy(spec)
    styled["design"]["typography"] = "Precise sans-serif labels with extra whitespace."
    return [json.dumps({"figure_spec": spec, "design_description": "Detailed planned left-to-right diagram with exact scientific labels."}),
            json.dumps({"figure_spec": styled, "design_description": "Detailed polished left-to-right design with clear hierarchy and exact labels."})]


async def run_processor(processor, sdk, data):
    with patch("utils.generation_utils._client", return_value=sdk):
        try:
            return [result async for result in processor.process_queries_batch([data], do_eval=False)][0]
        finally:
            await processor.aclose()


def test_real_five_agent_pipeline_local_edit_exports_exact_final_contract(tmp_path):
    spec = specification()
    ids = [item["id"] for key in ("nodes", "edges", "groups", "labels") for item in spec[key]]
    repair = {"instruction": "Improve whitespace at the input only.", "target_ids": ["input"],
              "allowed_changes": ["spacing"], "protected_ids": [identifier for identifier in ids if identifier != "input"],
              "protected_regions": [[0.5, 0, 0.5, 1]]}
    text_responses = envelopes(spec) + [
        json.dumps({"critic_suggestions": "Improve spacing at input.", "revised_description": "Preserve the whole scientific diagram; adjust input whitespace only.", "repair": repair}),
        json.dumps({"critic_suggestions": "No changes needed.", "revised_description": "No changes needed.", "repair": None}),
    ]
    draft = png_fixture((40, 80, 120, 180))
    styled = png_fixture((45, 85, 125, 190))
    final = png_fixture((45, 85, 125, 190), repair=True)
    sdk = ControlledGeminiSDK(text_responses, [draft, styled, final])
    processor, reference = create_processor(tmp_path)
    result = asyncio.run(run_processor(processor, sdk, query(spec)))
    assert result["status"] == "succeeded" and result["review_status"] == "completed"
    assert result["retrieval_status"]["selected_ids"] == ["local-reference"]
    assert result["eval_image_field"] == "target_diagram_critic_desc0_base64_jpg"
    assert scientific_fingerprint(result["figure_spec"]) == scientific_fingerprint(spec)
    assert len(result["rfs_reviews"]) == 2
    assert not sdk.texts and not sdk.images
    image_requests = [request for request in sdk.requests if request["is_image"]]
    assert len(image_requests) == 3  # Planner+Stylist drafts and one bounded edit.
    edit_request = image_requests[-1]
    request_images = [part.inline_data.data for part in edit_request["parts"] if part.inline_data]
    assert request_images == [styled]  # Original PNG, not the lossy legacy JPG preview.
    planning_request = next(request for request in sdk.requests if not request["is_image"])
    assert any(part.inline_data and part.inline_data.data == reference for part in planning_request["parts"])
    record = result["rfs_artifacts"]["target_diagram_critic_desc0"]
    actual_prompt = "\n".join(part.text for part in edit_request["parts"] if part.text)
    assert record["prompt"] == actual_prompt
    assert record["stage"] == "local_repair"
    target = tmp_path / "exchange"
    manifest = export_bundle(result, target)
    assert manifest == verify_bundle(target)
    assert (target / "image.png").read_bytes() == final
    assert (target / "prompt.txt").read_text() == actual_prompt
    assert json.loads((target / "figure_spec.input.json").read_text()) == spec
    assert manifest["stage"] == "local_repair" and manifest["scientific_fingerprint"] == scientific_fingerprint(spec)
    assert manifest["provenance"]["scientific_verification"] == "unverified"
    with Image.open(io.BytesIO(styled)) as before, Image.open(target / "image.png") as after:
        assert before.crop((32, 0, 64, 48)).tobytes() == after.crop((32, 0, 64, 48)).tobytes()
        assert after.mode == "RGBA"


def test_real_pipeline_malformed_review_retains_prior_image_with_failed_manifest(tmp_path):
    spec = specification()
    draft, styled = png_fixture((40, 80, 120, 180)), png_fixture((45, 85, 125, 190))
    sdk = ControlledGeminiSDK(envelopes(spec) + ["{}"], [draft, styled])
    processor, _reference = create_processor(tmp_path)
    result = asyncio.run(run_processor(processor, sdk, query(spec)))
    assert result["status"] == "failed" and result["review_status"] == "failed"
    assert result["eval_image_field"] == "target_diagram_stylist_desc0_base64_jpg"
    assert result["error"]["stage"] == "critic"
    assert "target_diagram_critic_desc0" not in result
    assert len([request for request in sdk.requests if request["is_image"]]) == 2
    manifest = export_bundle(result, tmp_path / "review-failed")
    assert manifest["status"] == "failed" and manifest["review_status"] == "failed"
    assert (tmp_path / "review-failed/image.png").read_bytes() == styled
    verify_bundle(tmp_path / "review-failed")


def test_real_pipeline_rejects_planner_rewriting_user_scientific_contract(tmp_path):
    spec = specification()
    changed = copy.deepcopy(spec)
    changed["edges"][0]["target"] = "embedding"  # Valid endpoint, scientifically different topology.
    sdk = ControlledGeminiSDK([json.dumps({"figure_spec": changed, "design_description": "Invalid scientifically altered design."})], [])
    processor, _reference = create_processor(tmp_path)
    result = asyncio.run(run_processor(processor, sdk, query(spec)))
    assert result["status"] == "failed" and result["error"]["stage"] == "planner"
    assert result["figure_spec"] == spec and result["rfs_input_figure_spec"] == spec
    assert not any(request["is_image"] for request in sdk.requests)
    manifest = export_bundle(result, tmp_path / "science-failed")
    assert manifest["status"] == "failed" and "image" not in manifest["files"]
    assert json.loads((tmp_path / "science-failed/figure_spec.json").read_text()) == spec
