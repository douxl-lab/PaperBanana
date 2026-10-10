"""Versioned scientific exchange contract, independent of model/network clients.

Structural/evidence checks establish traceability, not scientific correctness.
External RFS consumer interoperability has not been accepted or tested.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_VERSION = "1.0.0"
SCHEMA_PATH = Path(__file__).parent / "schemas" / "figure-spec-v1.schema.json"
SCIENTIFIC_KEYS = ("schema_version", "figure_id", "kind", "nodes", "edges", "groups", "labels", "datasets", "sources", "constraints", "provenance")


class RFSValidationError(ValueError):
    """A malformed, unsupported, or scientifically inconsistent exchange record."""


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise RFSValidationError("Contract must contain finite JSON values") from exc


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def strict_json_response(text: str) -> dict:
    """Accept a single JSON object, optionally fenced; never repair missing facts."""
    if not isinstance(text, str) or not text.strip() or text.strip() == "Error":
        raise RFSValidationError("Model returned no usable structured response")
    if len(text) > 2_000_000:
        raise RFSValidationError("Structured response exceeds 2 MB limit")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.S)
        if not match:
            raise RFSValidationError("Response must be one JSON object")
        cleaned = match.group(1).strip()

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RFSValidationError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(cleaned, object_pairs_hook=pairs,
                           parse_constant=lambda token: (_ for _ in ()).throw(RFSValidationError(f"Nonfinite JSON: {token}")))
    except (json.JSONDecodeError, RecursionError) as exc:
        raise RFSValidationError("Malformed model JSON; no fallback specification was created") from exc
    if not isinstance(value, dict):
        raise RFSValidationError("Response must be a JSON object")
    return value


def strict_json_input(text: str) -> dict | list:
    """Parse user data without repair, duplicate-key loss, or nonfinite numbers."""
    if not isinstance(text, str) or not text.strip():
        raise RFSValidationError("Structured scientific input is required")
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise RFSValidationError(f"Duplicate scientific input key: {key}")
            value[key] = item
        return value
    try:
        value = json.loads(text, object_pairs_hook=pairs,
                           parse_constant=lambda token: (_ for _ in ()).throw(RFSValidationError(f"Nonfinite scientific data: {token}")))
    except (json.JSONDecodeError, RecursionError) as exc:
        raise RFSValidationError("Scientific data must be strict JSON; values are never repaired") from exc
    if not isinstance(value, (dict, list)):
        raise RFSValidationError("Structured scientific input must be an object or array")
    canonical_json(value)
    return value


def validate_relative_path(value: str) -> str:
    """Validate exchange paths lexically; this does not grant filesystem access."""
    if (not isinstance(value, str) or not value or "\x00" in value or "\\" in value
            or PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute()
            or PureWindowsPath(value).drive or any(part in ("..", ".") for part in value.split("/"))
            or ":" in value):
        raise RFSValidationError("Artifact/source path must be a safe relative POSIX path")
    return value


def input_source(content: Any) -> dict:
    if isinstance(content, (dict, list)):
        canonical_json(content)  # Finite JSON guard without reordering source columns/maps.
        text = json.dumps(content, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    else:
        text = content
    if not isinstance(text, str) or not text.strip():
        raise RFSValidationError("Scientific input content is required")
    return {"id": "input-source", "kind": "user_data" if isinstance(content, (dict, list)) or table_from_content(content) is not None else "user_text",
            "content": text, "sha256": sha256_text(text)}


def table_from_content(content: Any) -> dict | None:
    """Normalize only explicit tables/records/scalar maps, without invented values."""
    if isinstance(content, str):
        try:
            content = strict_json_input(content)
        except (ValueError, TypeError):
            return None
    if isinstance(content, dict) and set(("columns", "rows")) <= set(content):
        columns, rows = content["columns"], content["rows"]
    elif isinstance(content, list) and content and all(isinstance(row, dict) for row in content):
        columns = list(content[0])
        if any(set(row) != set(columns) for row in content):
            return None
        rows = [[row[column] for column in columns] for row in content]
    elif isinstance(content, dict) and content and all(isinstance(v, (str, int, float, bool)) or v is None for v in content.values()):
        columns, rows = ["label", "value"], [[key, value] for key, value in content.items()]
    else:
        return None
    if (not isinstance(columns, list) or not columns or any(not isinstance(c, str) or not c for c in columns)
            or len(set(columns)) != len(columns) or not isinstance(rows, list) or not rows
            or any(not isinstance(row, list) or len(row) != len(columns) for row in rows)
            or any(isinstance(v, (dict, list)) for row in rows for v in row)):
        return None
    return {"columns": copy.deepcopy(columns), "rows": copy.deepcopy(rows)}


def dataset_from_content(content: Any, *, dataset_id: str = "input-data", name: str = "User supplied data") -> dict | None:
    table = table_from_content(content)
    if table is None:
        return None
    return {"id": dataset_id, "name": name, "source_id": "input-source", **table,
            "sha256": sha256_text(canonical_json(table))}


def validate_figure_spec(value: Any, *, input_content: Any = None) -> dict:
    """Return an isolated validated copy; preserve labels, topology and exact data."""
    canonical_json(value)  # Reject nonfinite numbers before JSON Schema inspection.
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: str(list(e.path)))
    if errors:
        error = errors[0]
        raise RFSValidationError(f"FigureSpec {'.'.join(map(str, error.path)) or '<root>'}: {error.message}")
    spec = copy.deepcopy(value)
    collections = ("nodes", "edges", "groups", "labels", "datasets")
    entity_ids = [item["id"] for key in collections for item in spec[key]]
    if len(entity_ids) != len(set(entity_ids)):
        raise RFSValidationError("Scientific entity IDs must be unique across collections")
    source_ids = [source["id"] for source in spec["sources"]]
    if len(source_ids) != len(set(source_ids)):
        raise RFSValidationError("Source IDs must be unique")
    sources = {source["id"]: source for source in spec["sources"]}
    nodes = {node["id"] for node in spec["nodes"]}
    datasets = {dataset["id"] for dataset in spec["datasets"]}
    if spec["kind"] in ("diagram", "hybrid") and not nodes:
        raise RFSValidationError("A scientific diagram requires at least one node")
    if spec["kind"] in ("plot", "hybrid") and not datasets:
        raise RFSValidationError("Quantitative figures require supplied datasets")
    for source in sources.values():
        if source["kind"] == "user_data":
            strict_json_input(source["content"])
        if source["sha256"] != sha256_text(source["content"]):
            raise RFSValidationError(f"Source checksum mismatch: {source['id']}")
        if "relative_path" in source:
            validate_relative_path(source["relative_path"])
    for key in ("nodes", "edges", "groups", "labels"):
        for item in spec[key]:
            for evidence in item["evidence"]:
                source = sources.get(evidence["source_id"])
                if source is None or evidence["quote"] not in source["content"]:
                    raise RFSValidationError(f"Evidence for {item['id']} must quote a supplied source exactly")
    for node in spec["nodes"]:
        if not set(node.get("dataset_ids", [])) <= datasets:
            raise RFSValidationError(f"Unknown node dataset reference: {node['id']}")
    for edge in spec["edges"]:
        if edge["source"] not in nodes or edge["target"] not in nodes:
            raise RFSValidationError(f"Unknown edge endpoint: {edge['id']}")
    for group in spec["groups"]:
        if not group["node_ids"] or not set(group["node_ids"]) <= nodes:
            raise RFSValidationError(f"Group must reference known nodes: {group['id']}")
    for label in spec["labels"]:
        if not set(label["target_ids"]) <= set(entity_ids):
            raise RFSValidationError(f"Unknown label target: {label['id']}")
    for dataset in spec["datasets"]:
        source = sources.get(dataset["source_id"])
        table = {"columns": dataset["columns"], "rows": dataset["rows"]}
        if source is None or source["kind"] != "user_data" or canonical_json(table_from_content(source["content"])) != canonical_json(table):
            raise RFSValidationError(f"Dataset {dataset['id']} must exactly retain its original supplied table")
        if dataset["sha256"] != sha256_text(canonical_json(table)):
            raise RFSValidationError(f"Dataset checksum mismatch: {dataset['id']}")
        if any(len(row) != len(dataset["columns"]) for row in dataset["rows"]):
            raise RFSValidationError(f"Dataset row width mismatch: {dataset['id']}")
        if not set(dataset.get("units", {})) <= set(dataset["columns"]):
            raise RFSValidationError(f"Unknown unit column: {dataset['id']}")
        for channel, column in dataset.get("encoding", {}).items():
            if channel != "chart_type" and column not in dataset["columns"]:
                raise RFSValidationError(f"Unknown encoded column: {column}")
    text_labels = {item["label"] for key in ("nodes", "groups") for item in spec[key]}
    text_labels |= {item["text"] for item in spec["labels"]}
    text_labels |= {edge["label"] for edge in spec["edges"] if "label" in edge}
    if not set(spec["constraints"]["required_labels"]) <= text_labels:
        raise RFSValidationError("Required labels must be explicit scientific labels")
    if spec["provenance"]["origin"] == "model_generated":
        if spec["provenance"]["scientific_verification"] != "unverified":
            raise RFSValidationError("A model cannot mark its generated scientific claims verified")
        if input_content is not None and spec["sources"] != [input_source(input_content)]:
            raise RFSValidationError("Model-generated sources must match the original scientific input exactly")
        # This guards invented numeric claims, not semantic scientific truth.
        for key in ("nodes", "edges", "groups", "labels"):
            for item in spec[key]:
                evidence_text = " ".join(e["quote"] for e in item["evidence"])
                numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", " ".join(str(item.get(k, "")) for k in ("label", "text", "formula")))
                if any(number not in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", evidence_text) for number in numbers):
                    raise RFSValidationError(f"Unsupported numerical claim in {item['id']}")
    return spec


def validate_spec_task(spec: dict, task_name: str) -> None:
    """Prevent routing a quantitative contract through a different task backend."""
    if task_name not in ("diagram", "plot") or spec.get("kind") not in (task_name, "hybrid"):
        raise RFSValidationError("FigureSpec kind conflicts with the selected task backend")


def scientific_fingerprint(spec: dict) -> str:
    validated = validate_figure_spec(spec)
    return sha256_text(canonical_json({key: validated[key] for key in SCIENTIFIC_KEYS}))


def assert_scientific_invariants(original: dict, proposed: dict) -> None:
    if scientific_fingerprint(original) != scientific_fingerprint(proposed):
        raise RFSValidationError("Model changed locked scientific labels, topology, data, sources, constraints, or verification status")


def parse_planner_response(text: str, *, input_content: Any, locked_spec: dict | None = None) -> tuple[dict, str]:
    result = strict_json_response(text)
    if set(result) != {"figure_spec", "design_description"}:
        raise RFSValidationError("Planner must return figure_spec and design_description only")
    description = result["design_description"]
    if not isinstance(description, str) or not description.strip() or description.strip() == "Error":
        raise RFSValidationError("Planner must provide a nonempty detailed design description")
    spec = validate_figure_spec(result["figure_spec"], input_content=None if locked_spec is not None else input_content)
    if locked_spec is not None:
        assert_scientific_invariants(locked_spec, spec)
    elif spec["provenance"]["origin"] != "model_generated":
        raise RFSValidationError("Automatically planned structure must disclose model_generated origin")
    return spec, description.strip()


def is_rfs_enabled(data: dict, config: Any = None) -> bool:
    return data.get("render_mode", getattr(config, "render_mode", "legacy")) == "rfs"


def validate_repair(value: Any, spec: dict) -> dict:
    if not isinstance(value, dict) or set(value) - {"instruction", "target_ids", "allowed_changes", "protected_ids", "protected_regions"}:
        raise RFSValidationError("Local repair must be a structured, bounded edit request")
    for key in ("instruction", "target_ids", "allowed_changes", "protected_ids"):
        if key not in value:
            raise RFSValidationError(f"Local repair missing {key}")
    if not isinstance(value["instruction"], str) or not value["instruction"].strip():
        raise RFSValidationError("Local repair instruction is required")
    for key in ("target_ids", "allowed_changes", "protected_ids"):
        if not isinstance(value[key], list) or any(not isinstance(v, str) or not v for v in value[key]):
            raise RFSValidationError(f"Local repair {key} must be a string list")
    ids = {item["id"] for key in ("nodes", "edges", "groups", "labels", "datasets") for item in spec[key]}
    targets, protected = set(value["target_ids"]), set(value["protected_ids"])
    if not targets or not value["allowed_changes"] or not targets <= ids or not protected <= ids or targets & protected:
        raise RFSValidationError("Repair needs known targets, allowed changes and disjoint protected IDs")
    if ids - targets != protected:
        raise RFSValidationError("Local repair must explicitly protect every non-target scientific entity")
    regions = value.get("protected_regions", [])
    if not isinstance(regions, list):
        raise RFSValidationError("protected_regions must be normalized [x,y,width,height] rectangles")
    for region in regions:
        if (not isinstance(region, list) or len(region) != 4 or any(type(n) not in (int, float) or not math.isfinite(n) for n in region)
                or any(n < 0 or n > 1 for n in region) or region[2] <= 0 or region[3] <= 0 or region[0] + region[2] > 1 or region[1] + region[3] > 1):
            raise RFSValidationError("Invalid protected region rectangle")
    return copy.deepcopy(value)
