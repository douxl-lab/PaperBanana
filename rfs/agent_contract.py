"""Optional structured instructions used by the existing three language agents."""
from __future__ import annotations

import copy
import json
from typing import Any

from .contract import (SCHEMA_PATH, RFSValidationError, canonical_json, dataset_from_content,
                       input_source, parse_planner_response, strict_json_response,
                       validate_figure_spec, validate_repair, validate_spec_task)

STRUCTURED_SYSTEM_SUPPLEMENT = """
RFS exchange mode: Return one strict JSON object, without prose or markdown.
The scientific FigureSpec is immutable after planning: never change labels, formulas,
nodes, edge directions/types, groups, real data, sources, constraints, or verification.
Aesthetic design choices may vary only within these constraints. Do not invent paper
facts or numerical results. Exact quotes establish traceability, not scientific truth.
Generated scientific structure must be model_generated and unverified. Never claim
external RFS consumer validation, native editable PPTX output, or scientific acceptance.
"""


def planner_instructions(data: dict, *, task_name: str = "diagram") -> tuple[str, dict | None]:
    existing = data.get("figure_spec")
    locked = validate_figure_spec(existing) if existing is not None else None
    if locked is not None:
        validate_spec_task(locked, task_name)
        data["rfs_input_figure_spec"] = copy.deepcopy(locked)
        context = "USER SCIENTIFIC FIGURESPEC (preserve every field except design exactly):\n" + canonical_json(locked)
    else:
        source = input_source(data["content"])
        dataset = dataset_from_content(data["content"])
        context = ("Create FigureSpec using this public schema:\n" + SCHEMA_PATH.read_text(encoding="utf-8") +
                   "\nUse exactly these sources without changes:\n" + canonical_json([source]) +
                   "\nUse model_generated origin, unverified scientific_verification, and external_consumer_verified=false. "
                   "All scientific entities need exact supplied-source quote evidence. "
                   "Unknown facts remain absent/unverified; do not invent values. "
                   "At least one node is required for a diagram. The constraints.locked field must be true.")
        if dataset is not None:
            context += "\nPreserve this complete real-data dataset verbatim (encoding may name existing columns):\n" + canonical_json(dataset)
        else:
            context += "\nNo explicit tabular data was provided: datasets must be empty."
    context += f"\nSelected task backend is {task_name}; FigureSpec kind must be {task_name} or hybrid."
    context += ("\nReturn exactly {\"figure_spec\": <complete contract>, \"design_description\": <detailed design text>}. "
                "Describe composition, spacing, typography, palette, connector routing, groups, exact text, "
                "and how all scientific entities correspond to the illustration. Preserve all user constraints.")
    return context, locked


def apply_planner_response(data: dict, text: str, *, locked_spec: dict | None, desc_key: str) -> None:
    spec, description = parse_planner_response(text, input_content=data["content"], locked_spec=locked_spec)
    validate_spec_task(spec, "plot" if desc_key.startswith("target_plot_") else "diagram")
    if locked_spec is None:
        expected_data = dataset_from_content(data["content"])
        if expected_data is None and spec["datasets"]:
            raise RFSValidationError("Planner cannot invent datasets without supplied table data")
        if expected_data is not None:
            tables = [{"columns": d["columns"], "rows": d["rows"], "source_id": d["source_id"]} for d in spec["datasets"]]
            expected = {"columns": expected_data["columns"], "rows": expected_data["rows"], "source_id": expected_data["source_id"]}
            if expected not in tables:
                raise RFSValidationError("Planner omitted or changed the supplied real dataset")
    data["figure_spec"] = spec
    data["rfs_locked_spec"] = copy.deepcopy(locked_spec or spec)
    data["rfs_design_description"] = description
    data[desc_key] = description
    data.setdefault("rfs_stage_specs", {})[desc_key] = copy.deepcopy(spec)


def stylist_instructions(data: dict) -> str:
    spec = validate_figure_spec(data["figure_spec"])
    return ("RFS SCIENTIFIC CONTRACT (only design may change):\n" + canonical_json(spec) +
            '\nReturn exactly {"figure_spec": <complete preserved contract with aesthetic design refinement>, '
            '"design_description": <complete detailed polished design>}. Preserve scientific facts and all constraints.')


def apply_stylist_response(data: dict, text: str, *, desc_key: str) -> None:
    spec, description = parse_planner_response(text, input_content=data["content"], locked_spec=data["rfs_locked_spec"])
    data["figure_spec"] = spec
    data["rfs_design_description"] = description
    data[desc_key] = description
    data.setdefault("rfs_stage_specs", {})[desc_key] = copy.deepcopy(spec)


def critic_instructions(data: dict) -> str:
    return ("RFS SCIENTIFIC CONTRACT (immutable, do not return a changed contract):\n" + canonical_json(data["figure_spec"]) +
            '\nReturn exactly {"critic_suggestions": <text>, "revised_description": <complete text or "No changes needed.">, '
            '"repair": null or {"instruction": <bounded aesthetic instruction>, "target_ids": [<known ids>], '
            '"allowed_changes": [<presentation changes>], "protected_ids": [<every other node/edge/group/label/dataset id>], '
            '"protected_regions": [<optional normalized x,y,width,height rectangles>]}}. '
            'If no changes are needed, both text fields must be "No changes needed." and repair null. '
            'If changes are needed, provide a local repair object; never change scientific structure or data. '
            'If the image is unavailable, do not declare visual acceptance. State the missing-image limitation.')


def parse_critic_response(text: str, *, spec: dict | None = None, image_available: bool = True) -> dict:
    result = strict_json_response(text)
    for key in ("critic_suggestions", "revised_description"):
        if not isinstance(result.get(key), str) or not result[key].strip() or result[key].strip() == "Error":
            raise RFSValidationError(f"Critic response requires a nonempty string {key}")
        result[key] = result[key].strip()
    no_suggestions = result["critic_suggestions"] == "No changes needed."
    no_revision = result["revised_description"] == "No changes needed."
    if no_suggestions != no_revision:
        raise RFSValidationError("Critic no-change decision and revised description disagree")
    if no_suggestions and not image_available:
        raise RFSValidationError("Critic cannot accept a missing image")
    if spec is not None:
        if set(result) != {"critic_suggestions", "revised_description", "repair"}:
            raise RFSValidationError("RFS critic must return suggestions, revised_description, and repair only")
        if no_suggestions:
            if result["repair"] is not None:
                raise RFSValidationError("No-change review cannot contain a local repair")
        else:
            result["repair"] = validate_repair(result["repair"], validate_figure_spec(spec))
    return result
