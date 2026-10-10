import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agents.critic_agent import CriticAgent
from agents.planner_agent import PlannerAgent
from agents.stylist_agent import StylistAgent
from rfs.agent_contract import parse_critic_response
from rfs.contract import RFSValidationError, scientific_fingerprint

ROOT = Path(__file__).parents[1]


def config(mode="rfs"):
    return SimpleNamespace(main_model_name="test-main", task_name="diagram", planner_metaphor=False,
                           render_mode=mode, temperature=1, work_dir=ROOT)


@pytest.fixture
def spec():
    return json.loads((ROOT / "rfs/examples/encoder-pipeline.figure-spec.json").read_text())


def data_for(spec):
    return {"candidate_id": "0", "content": spec["sources"][0]["content"], "visual_intent": "Show the encoder pipeline.",
            "figure_spec": copy.deepcopy(spec), "retrieved_examples": []}


def test_opt_in_planner_and_stylist_preserve_legacy_keys_and_science(spec):
    data = data_for(spec)
    original = copy.deepcopy(spec)
    response = json.dumps({"figure_spec": spec, "design_description": "Full detailed design with labeled boxes and a framed encoder."})
    with patch("utils.generation_utils.call_model_with_retry_async", AsyncMock(return_value=[response])) as call:
        planned = asyncio.run(PlannerAgent(exp_config=config()).process(data))
    assert planned["target_diagram_desc0"] == planned["rfs_design_description"]
    assert planned["rfs_input_figure_spec"] == original
    assert "RFS exchange mode" in call.call_args.kwargs["config"].system_instruction
    refined_spec = copy.deepcopy(spec)
    refined_spec["design"]["palette"] = ["#112233", "#FFFFFF"]
    refined = json.dumps({"figure_spec": refined_spec, "design_description": "Refined typography and whitespace."})
    with patch("utils.generation_utils.call_model_with_retry_async", AsyncMock(return_value=[refined])):
        styled = asyncio.run(StylistAgent(exp_config=config()).process(planned))
    assert styled["target_diagram_stylist_desc0"] == "Refined typography and whitespace."
    assert scientific_fingerprint(styled["figure_spec"]) == scientific_fingerprint(original)
    assert styled["rfs_stage_specs"]["target_diagram_desc0"]["design"] == original["design"]


def test_planner_rejects_changed_user_scientific_labels(spec):
    data = data_for(spec)
    spec["nodes"][0]["label"] = "Changed feature"
    spec["constraints"]["required_labels"][0] = "Changed feature"
    response = json.dumps({"figure_spec": spec, "design_description": "Detailed but invalid design."})
    with patch("utils.generation_utils.call_model_with_retry_async", AsyncMock(return_value=[response])):
        with pytest.raises(RFSValidationError, match="locked scientific"):
            asyncio.run(PlannerAgent(exp_config=config()).process(data))
    assert "target_diagram_desc0" not in data


def test_automatic_planner_discloses_generated_unverified_origin(spec):
    data = data_for(spec)
    del data["figure_spec"]
    spec["provenance"]["origin"] = "model_generated"
    response = json.dumps({"figure_spec": spec, "design_description": "Traceable but unverified design."})
    with patch("utils.generation_utils.call_model_with_retry_async", AsyncMock(return_value=[response])):
        planned = asyncio.run(PlannerAgent(exp_config=config()).process(data))
    assert planned["figure_spec"]["provenance"]["origin"] == "model_generated"
    assert planned["figure_spec"]["provenance"]["scientific_verification"] == "unverified"


def test_legacy_planner_still_emits_plain_description(spec):
    data = data_for(spec)
    del data["figure_spec"]
    with patch("utils.generation_utils.call_model_with_retry_async", AsyncMock(return_value=["Legacy plain description"])) as call:
        planned = asyncio.run(PlannerAgent(exp_config=config("legacy")).process(data))
    assert planned["target_diagram_desc0"] == "Legacy plain description"
    assert "figure_spec" not in planned
    assert "RFS exchange mode" not in call.call_args.kwargs["config"].system_instruction


@pytest.mark.parametrize("response", ["Error", "{}", "[]", "not json", '{"critic_suggestions":null,"revised_description":null}', '{"critic_suggestions":"No changes needed.","revised_description":5}'])
def test_legacy_critic_malformed_output_fails_closed(spec, response):
    data = data_for(spec)
    data.update(target_diagram_stylist_desc0="Description", target_diagram_stylist_desc0_base64_jpg="x" * 120)
    with patch("utils.generation_utils.call_model_with_retry_async", AsyncMock(return_value=[response])):
        with pytest.raises(RFSValidationError):
            asyncio.run(CriticAgent(exp_config=config("legacy")).process(data))
    assert "target_diagram_critic_suggestions0" not in data


def test_critic_no_changes_requires_image_and_consistent_decision():
    text = json.dumps({"critic_suggestions": "No changes needed.", "revised_description": "No changes needed."})
    assert parse_critic_response(text)["critic_suggestions"] == "No changes needed."
    with pytest.raises(RFSValidationError, match="missing image"):
        parse_critic_response(text, image_available=False)
    with pytest.raises(RFSValidationError, match="disagree"):
        parse_critic_response(json.dumps({"critic_suggestions": "Fix spacing", "revised_description": "No changes needed."}))


def test_rfs_critic_records_valid_review_without_certifying_facts(spec):
    data = data_for(spec)
    data.update(target_diagram_stylist_desc0="Description", target_diagram_stylist_desc0_base64_jpg="x" * 120)
    response = json.dumps({"critic_suggestions": "No changes needed.", "revised_description": "No changes needed.", "repair": None})
    original = copy.deepcopy(spec)
    with patch("utils.generation_utils.call_model_with_retry_async", AsyncMock(return_value=[response])):
        reviewed = asyncio.run(CriticAgent(exp_config=config()).process(data))
    assert reviewed["target_diagram_critic_desc0"] == "Description"
    assert reviewed["figure_spec"] == original
    assert reviewed["rfs_reviews"][0]["scientific_verification"] == "unverified"
    assert reviewed["rfs_reviews"][0]["review_kind"] == "model_visual_review"


def test_rfs_critic_must_return_bounded_repair(spec):
    response = json.dumps({"critic_suggestions": "Improve spacing", "revised_description": "Improved description", "repair": None})
    with pytest.raises(RFSValidationError):
        parse_critic_response(response, spec=spec)


def test_three_language_agents_forward_explicit_provider_and_profile(spec):
    cfg = config("legacy")
    cfg.main_model_provider = "openrouter"
    cfg.model_profiles = {"test-main": {"provider": "openrouter"}}
    cfg.candidate_timeout_seconds = 45
    for cls, response in ((PlannerAgent, "Legacy description"), (StylistAgent, "Styled description"),
                          (CriticAgent, json.dumps({"critic_suggestions": "No changes needed.", "revised_description": "No changes needed."}))):
        data = data_for(spec)
        data.update(target_diagram_desc0="Description", target_diagram_stylist_desc0="Styled description",
                    target_diagram_stylist_desc0_base64_jpg="x" * 120)
        with patch("utils.generation_utils.call_model_with_retry_async", AsyncMock(return_value=[response])) as call:
            asyncio.run(cls(exp_config=cfg).process(data))
        assert call.call_args.kwargs["provider"] == "openrouter"
        assert call.call_args.kwargs["overrides"] == cfg.model_profiles
        assert call.call_args.kwargs["timeout"] == call.call_args.kwargs["total_timeout"] == 45


def test_planner_does_not_open_escaping_reference_path(tmp_path, spec):
    data = data_for(spec)
    del data["figure_spec"]
    data["retrieved_examples"] = [{"content": "Example", "visual_intent": "Example", "path_to_gt_image": "../../../../outside.png"}]
    cfg = config("legacy")
    cfg.work_dir = tmp_path
    with patch("utils.generation_utils.call_model_with_retry_async", AsyncMock(return_value=["Legacy description"])) as call:
        output = asyncio.run(PlannerAgent(exp_config=cfg).process(data))
    assert output["omitted_references"][0]["reason"] == "RFSValidationError"
    assert all(item["type"] != "image" for item in call.call_args.kwargs["contents"])


def test_planner_missing_reference_metadata_is_recorded(tmp_path, spec):
    data = data_for(spec)
    del data["figure_spec"]
    data["top10_references"] = ["missing"]
    cfg = config("legacy")
    cfg.work_dir = tmp_path
    with patch("utils.generation_utils.call_model_with_retry_async", AsyncMock(return_value=["Legacy description"])):
        output = asyncio.run(PlannerAgent(exp_config=cfg).process(data))
    assert output["omitted_references"][0]["stage"] == "reference_metadata"
    assert output["target_diagram_desc0"] == "Legacy description"


def test_stylist_uses_selected_guide(tmp_path, spec):
    data = data_for(spec)
    data["target_diagram_desc0"] = "Description"
    cfg = config("legacy")
    guide = tmp_path / "style.md"
    guide.write_text("Selected custom guide with generous whitespace.")
    cfg.style_guide_path = guide
    with patch("utils.generation_utils.call_model_with_retry_async", AsyncMock(return_value=["Styled description"])) as call:
        asyncio.run(StylistAgent(exp_config=cfg).process(data))
    assert "Selected custom guide" in call.call_args.kwargs["contents"][0]["text"]


def test_planner_rejects_user_contract_task_mismatch_before_model_call():
    spec = json.loads((ROOT / "rfs/examples/accuracy-comparison.figure-spec.json").read_text())
    data = {"content": spec["sources"][0]["content"], "visual_intent": "Quantitative comparison", "figure_spec": spec}
    with patch("utils.generation_utils.call_model_with_retry_async", AsyncMock()) as call:
        with pytest.raises(RFSValidationError, match="task backend"):
            asyncio.run(PlannerAgent(exp_config=config()).process(data))
    call.assert_not_awaited()
