import copy
import json
from pathlib import Path

import pytest

from rfs.contract import (RFSValidationError, assert_scientific_invariants, canonical_json,
                          dataset_from_content, input_source, parse_planner_response,
                          scientific_fingerprint, sha256_text, strict_json_response,
                          validate_figure_spec, validate_relative_path, validate_repair)
from rfs.prompt_compiler import compile_comparison_prompt, compile_prompt

EXAMPLES = Path(__file__).parents[1] / "rfs" / "examples"


@pytest.fixture
def spec():
    return json.loads((EXAMPLES / "encoder-pipeline.figure-spec.json").read_text())


@pytest.fixture
def capabilities():
    return {"provider": "mock", "model": "explicit-test-model", "supports_system_prompt": True,
            "supports_reference_image": True, "supports_editing": True, "prompt_notes": ["Preserve exact text."]}


def test_public_fixture_and_copy_round_trip(spec):
    result = validate_figure_spec(json.loads(json.dumps(spec)))
    assert result == spec
    result["nodes"][0]["label"] = "mutated"
    assert spec["nodes"][0]["label"] == "Input features"


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(schema_version="2.0.0"),
    lambda s: s.update(unknown=True),
    lambda s: s.update(nodes=[]),
    lambda s: s["edges"][0].update(source="missing"),
    lambda s: s["groups"][0].update(node_ids=["missing"]),
    lambda s: s["nodes"][0].update(id=s["nodes"][1]["id"]),
    lambda s: s["nodes"][0]["evidence"][0].update(quote="invented evidence"),
    lambda s: s["sources"][0].update(sha256="0" * 64),
    lambda s: s["constraints"].update(required_labels=["missing"]),
    lambda s: s["constraints"].update(locked=False),
    lambda s: s["provenance"].update(external_consumer_verified=True),
])
def test_invalid_contract_fails_closed(spec, mutation):
    mutation(spec)
    with pytest.raises(RFSValidationError):
        validate_figure_spec(spec)


@pytest.mark.parametrize("path", ["../escape", "/absolute", "C:/outside", "safe/../unsafe", "a\\b", "https://remote", "safe/./file", ""])
def test_source_paths_are_relative_and_safe(path):
    with pytest.raises(RFSValidationError):
        validate_relative_path(path)


def test_relative_paths_accepted():
    assert validate_relative_path("sources/raw-data.csv") == "sources/raw-data.csv"


@pytest.mark.parametrize("response", ["", "Error", "{}", "[]", '{"a":1,"a":2}', '{"a":NaN}', '{"a":1,}', "prose {\"a\": 1}"])
def test_strict_parser_never_repairs_missing_facts(response):
    with pytest.raises(RFSValidationError):
        if response == "{}":
            parse_planner_response(response, input_content="source")
        else:
            strict_json_response(response)


def test_fenced_complete_json_is_accepted():
    assert strict_json_response('```json\n{"a": 1}\n```') == {"a": 1}


def test_scientific_contract_locked_but_design_can_change(spec):
    proposed = copy.deepcopy(spec)
    proposed["design"]["palette"] = ["#112233"]
    assert_scientific_invariants(spec, proposed)
    assert scientific_fingerprint(spec) == scientific_fingerprint(proposed)
    proposed["nodes"][0]["label"] = "Invented feature"
    with pytest.raises(RFSValidationError):
        assert_scientific_invariants(spec, proposed)


def test_generated_structure_is_honestly_unverified(spec):
    spec["provenance"]["origin"] = "model_generated"
    content = spec["sources"][0]["content"]
    envelope = json.dumps({"figure_spec": spec, "design_description": "Detailed left-to-right design."})
    result, description = parse_planner_response(envelope, input_content=content)
    assert result["provenance"]["scientific_verification"] == "unverified"
    assert description
    spec["provenance"]["scientific_verification"] = "user_verified"
    with pytest.raises(RFSValidationError):
        validate_figure_spec(spec)


def test_model_cannot_forge_sources_or_numbers(spec):
    spec["provenance"]["origin"] = "model_generated"
    with pytest.raises(RFSValidationError):
        validate_figure_spec(spec, input_content="different source")
    spec["nodes"][0]["label"] = "98 inputs"
    spec["constraints"]["required_labels"][0] = "98 inputs"
    with pytest.raises(RFSValidationError, match="numerical"):
        validate_figure_spec(spec)


def test_real_table_values_cannot_change():
    spec = json.loads((EXAMPLES / "accuracy-comparison.figure-spec.json").read_text())
    assert validate_figure_spec(spec) == spec
    spec["datasets"][0]["rows"][0][1] = 0.99
    table = {key: spec["datasets"][0][key] for key in ("columns", "rows")}
    spec["datasets"][0]["sha256"] = sha256_text(canonical_json(table))
    with pytest.raises(RFSValidationError, match="original supplied table"):
        validate_figure_spec(spec)


def test_map_and_record_input_preserve_order():
    for content in ({"z": 2, "a": 1}, [{"z": 2, "a": 1}, {"z": 4, "a": 3}]):
        source = input_source(content)
        dataset = dataset_from_content(content)
        assert dataset["columns"][0] in ("label", "z")
        assert json.loads(source["content"]) == content


def test_numeric_boolean_distinction_is_preserved():
    spec = json.loads((EXAMPLES / "accuracy-comparison.figure-spec.json").read_text())
    spec["datasets"][0]["rows"][0][1] = True
    with pytest.raises(RFSValidationError):
        validate_figure_spec(spec)


def test_compiler_is_pure_deterministic_and_provider_adapted(spec, capabilities):
    original = copy.deepcopy(spec)
    result = compile_prompt(spec, "Use clear labeled boxes.", capabilities, candidate_id=0, stage="design")
    repeated = compile_prompt(spec, "Use clear labeled boxes.", capabilities, candidate_id=0, stage="design")
    assert result == repeated
    assert spec == original
    assert result.system_prompt
    for section in ("SCIENTIFIC CONTRACT", "LAYOUT", "VISUAL STYLE", "TEXT AND FORMULAS", "NEGATIVE CONSTRAINTS", "VERIFICATION BOUNDARY", "PROVIDER ADAPTATION"):
        assert section in result.prompt
    assert '"label":"Input features"' in result.prompt
    adapted = dict(capabilities, supports_system_prompt=False)
    result = compile_prompt(spec, "Description", adapted, candidate_id=0)
    assert result.system_prompt is None
    assert result.prompt.startswith("INSTRUCTIONS\n")


def test_candidates_have_explicit_presentation_difference(spec, capabilities):
    a = compile_prompt(spec, "Description", capabilities, candidate_id=0, stage="candidate")
    b = compile_prompt(spec, "Description", capabilities, candidate_id=1, stage="candidate")
    assert a.metadata["candidate_variation"] != b.metadata["candidate_variation"]
    assert a.metadata["scientific_fingerprint"] == b.metadata["scientific_fingerprint"]


def repair_for(spec):
    ids = [item["id"] for key in ("nodes", "edges", "groups", "labels") for item in spec[key]]
    return {"instruction": "Increase whitespace around Input features.", "target_ids": ["input"],
            "allowed_changes": ["spacing"], "protected_ids": [identifier for identifier in ids if identifier != "input"],
            "protected_regions": [[0.5, 0, 0.5, 1]]}


def test_bounded_local_repair_and_protected_regions(spec, capabilities):
    repair = repair_for(spec)
    result = compile_prompt(spec, "Description", capabilities, candidate_id=0, stage="local_repair", repair=repair)
    assert result.metadata["requires_reference_image"] is True
    assert "LOCAL REPAIR" in result.prompt and "protected_regions" in result.prompt
    with pytest.raises(RFSValidationError, match="cannot preserve"):
        compile_prompt(spec, "Description", dict(capabilities, supports_editing=False), candidate_id=0, stage="local_repair", repair=repair)
    repair["protected_ids"] = []
    with pytest.raises(RFSValidationError, match="protect every"):
        validate_repair(repair, spec)


def test_comparison_preserves_scientific_contract(spec, capabilities):
    result = compile_comparison_prompt(spec, [{"candidate_id": "a"}, {"candidate_id": "b"}], capabilities)
    assert result.metadata["stage"] == "comparison"
    assert "scientific fidelity separately" in result.prompt
    with pytest.raises(RFSValidationError):
        compile_comparison_prompt(spec, [{"candidate_id": "a"}, {"candidate_id": "a"}], capabilities)


@pytest.mark.parametrize("capability", ["provider", "model", "supports_system_prompt", "supports_reference_image", "supports_editing"])
def test_capabilities_must_be_explicit(spec, capabilities, capability):
    del capabilities[capability]
    with pytest.raises(RFSValidationError):
        compile_prompt(spec, "Description", capabilities, candidate_id=0)


def test_plot_code_repair_is_not_image_editing(capabilities):
    spec = json.loads((EXAMPLES / "accuracy-comparison.figure-spec.json").read_text())
    repair = {"instruction": "Increase axis label spacing.", "target_ids": ["input-data"],
              "allowed_changes": ["axis label placement"], "protected_ids": []}
    result = compile_prompt(spec, "Preserve all supplied accuracy values.", dict(capabilities, supports_editing=False),
                            candidate_id=0, stage="code_repair", repair=repair)
    assert result.metadata["requires_source_code"] is True
    assert result.metadata["requires_reference_image"] is False
    assert "code repair, not image editing" in result.prompt


@pytest.mark.parametrize("text", ['{"value":NaN}', '{"value":1,"value":2}', '{"columns":["v"],"rows":[[1,]]}'])
def test_structured_scientific_input_is_not_repaired(text):
    from rfs.contract import strict_json_input, table_from_content
    with pytest.raises(RFSValidationError):
        strict_json_input(text)
    assert table_from_content(text) is None
