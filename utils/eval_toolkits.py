# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Evaluation toolkits for PaperVizAgent
"""

import json_repair
import json
import asyncio
import base64
import re

from prompts import diagram_eval_prompts, plot_eval_prompts
from utils.generation_utils import call_model_with_retry_async
from utils.model_registry import get_default_model
from utils.legacy_ui_results import infer_task_name, resolve_final_output, resolve_gt_image_path
from utils.legacy_ui_refinement import image_mime_type, final_image_bytes


# Prompt mapping: task_name -> eval_dim -> system_prompt
PROMPT_MAP = {
    "diagram": {
        "faithfulness": diagram_eval_prompts.DIAGRAM_REFERENCED_COMPARISON_FAITHFULNESS_SYSTEM_PROMPT,
        "conciseness": diagram_eval_prompts.DIAGRAM_REFERENCED_COMPARISON_CONCISENESS_SYSTEM_PROMPT,
        "readability": diagram_eval_prompts.DIAGRAM_REFERENCED_COMPARISON_READABILITY_SYSTEM_PROMPT,
        "aesthetics": diagram_eval_prompts.DIAGRAM_REFERENCED_COMPARISON_AESTHETICS_SYSTEM_PROMPT,
    },
    "plot": {
        "faithfulness": plot_eval_prompts.PLOT_REFERENCED_COMPARISON_FAITHFULNESS_SYSTEM_PROMPT,
        "conciseness": plot_eval_prompts.PLOT_REFERENCED_COMPARISON_CONCISENESS_SYSTEM_PROMPT,
        "readability": plot_eval_prompts.PLOT_REFERENCED_COMPARISON_READABILITY_SYSTEM_PROMPT,
        "aesthetics": plot_eval_prompts.PLOT_REFERENCED_COMPARISON_AESTHETICS_SYSTEM_PROMPT,
    },
}

# Task configuration: task_name -> field labels
TASK_CONFIG = {
    "diagram": {
        "visual_intent_label": "Diagram Caption",
        "raw_content_label": "Methodology Section",
        "human_label": "Human-Drawn Diagram (Human)",
        "model_label": "Model-Generated Diagram (Model)",
    },
    "plot": {
        "visual_intent_label": "Visual Intent of the Desired Plot",
        "raw_content_label": "Raw Data",
        "human_label": "Human-Drawn Plot (Human)",
        "model_label": "Model-Generated Plot (Model)",
    },
}


def _try_regex_extract_winner(text: str) -> str | None:
    """Try to extract winner field using regex as a fallback."""
    patterns = [
        r'"winner"\s*:\s*"([^"]+)"',  # Standard JSON: "winner": "value"
        r'\*\*winner\*\*\s*:\s*"([^"]+)"',  # Markdown bold: **winner**: "value" or **winner**:"value"
        r'\*\*winner\*\*\s*:\s*([A-Za-z][A-Za-z\s]+?)(?:,|\n|$)',  # Markdown bold without quotes: **winner**: value (capture until comma, newline, or end)
        r'"winner"\s*:\s*([A-Za-z][A-Za-z\s]+?)(?:,|\n|$)',  # Mixed format: "winner": value (no quotes on value, capture until comma, newline, or end)
        r'(?:\*\*|")winner(?:\*\*|")\s*:\s*(?:\*\*|")?([A-Za-z][A-Za-z\s]+?)(?:\*\*|"|,|\n|$)',  # Very flexible: any winner marker followed by colon and value
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            value = value.rstrip('*"').strip()
            return value
    
    return None


def _extract_winner_with_fallback(clean_json: str, eval_dim: str, valid_winners: list[str]) -> str:
    """Try regex extraction and return winner or 'Error'."""
    extracted = _try_regex_extract_winner(clean_json)
    if extracted and extracted in valid_winners:
        print(f"⚠️  {eval_dim}: regex extracted '{extracted}'")
        return extracted
    print(f"⚠️  {eval_dim}: failed to extract valid winner")
    return "Error"


def _determine_tier_outcome(dim1_outcome: str, dim2_outcome: str) -> str:
    """Determine the outcome for a tier given two dimension outcomes."""
    valid = {"Human", "Model", "Both are good", "Both are bad"}
    if not isinstance(dim1_outcome, str) or not isinstance(dim2_outcome, str):
        return "Error"
    o1, o2 = dim1_outcome.strip(), dim2_outcome.strip()
    if o1 not in valid or o2 not in valid:
        return "Error"

    # Both agree on a clear winner
    if o1 == o2:
        if o1 in ["Both are good", "Both are bad"]:
            return "Tie"
        return o1
    
    # One Model, one neutral (Both are good/bad)
    if (o1 == "Model" and o2 in ["Both are good", "Both are bad"]) or \
       (o2 == "Model" and o1 in ["Both are good", "Both are bad"]):
        return "Model"
    
    # One Human, one neutral (Both are good/bad)
    if (o1 == "Human" and o2 in ["Both are good", "Both are bad"]) or \
       (o2 == "Human" and o1 in ["Both are good", "Both are bad"]):
        return "Human"
    
    # All other cases (conflicting winners, etc.) -> Tie
    return "Tie"


async def _run_single_eval_ref(
    task_name: str, eval_dim: str, raw_content: str, visual_intent: str,
    gt_image_base64: str, model_image_base64: str, model_name: str,
    provider: str | None = None, overrides: dict | None = None,
) -> tuple[str, dict]:
    """Run one visual comparison through the same explicit provider router as generation."""
    if task_name not in TASK_CONFIG:
        raise ValueError(f"Invalid task name: {task_name}")
    labels = TASK_CONFIG[task_name]
    sys_prompt = PROMPT_MAP[task_name][eval_dim]
    input_text = f"{labels['visual_intent_label']}: {visual_intent}\n{labels['human_label']}: "
    if eval_dim not in {"readability", "aesthetics"}:
        input_text = f"{labels['raw_content_label']}: {raw_content}\n" + input_text
    valid_winners = ["Human", "Model", "Both are good", "Both are bad"]
    try:
        images = []
        for encoded in (gt_image_base64, model_image_base64):
            encoded = encoded.split(",", 1)[-1] if encoded.startswith("data:") else encoded
            mime = image_mime_type(base64.b64decode(encoded, validate=True))
            images.append({"type": "image", "source": {
                "type": "base64", "media_type": mime, "data": encoded,
            }})
        content_list = [
            {"type": "text", "text": input_text}, images[0],
            {"type": "text", "text": f"\n{labels['model_label']}: "}, images[1],
        ]
        responses = await call_model_with_retry_async(
            model_name=model_name, provider=provider, overrides=overrides, contents=content_list,
            config={"system_prompt": sys_prompt, "temperature": 1.0,
                    "candidate_num": 1, "max_output_tokens": 10000},
            max_attempts=3, retry_delay=1, error_context=f"evaluation_{eval_dim}",
        )
        if not responses or not isinstance(responses[0], str) or not responses[0].strip() or responses[0] == "Error":
            raise ValueError("Evaluation model returned no usable response")
        clean_json = responses[0].replace("```json", "").replace("```", "").strip()
        res_obj = json_repair.loads(clean_json)
        if not isinstance(res_obj, dict):
            res_obj = {"comparison_reasoning": clean_json,
                       "winner": _extract_winner_with_fallback(clean_json, eval_dim, valid_winners)}
        elif "winner" not in res_obj:
            res_obj["winner"] = _extract_winner_with_fallback(clean_json, eval_dim, valid_winners)
        if res_obj.get("winner") not in valid_winners:
            raise ValueError("Evaluation response has an invalid winner")
        reasoning = res_obj.get("comparison_reasoning", clean_json)
        if not isinstance(reasoning, str):
            raise ValueError("Evaluation reasoning must be text")
        return eval_dim, {"comparison_reasoning": reasoning,
                          "winner": res_obj["winner"], "status": "completed"}
    except Exception as exc:
        # Provider failures cannot turn into a tie, a Human win, or a scientific pass.
        return eval_dim, {"comparison_reasoning": f"Evaluation failed ({type(exc).__name__}).",
                          "winner": "Error", "status": "failed",
                          "error_type": type(exc).__name__}


_DIMENSIONS = ("faithfulness", "conciseness", "readability", "aesthetics")


def _mark_not_evaluated(sample_data: dict, status: str, outcome: str, reason: str) -> dict:
    sample_data["evaluation_status"] = status
    sample_data["evaluation_error"] = reason
    sample_data.setdefault("scientific_verification_status", "unverified")
    for dimension in (*_DIMENSIONS, "overall"):
        sample_data[f"{dimension}_outcome"] = outcome
        sample_data[f"{dimension}_reasoning"] = reason
    return sample_data


async def get_score_for_image_referenced(
    sample_data: dict, task_name: str = "diagram", model_name: str = "", work_dir=None,
    *, provider: str | None = None, overrides: dict | None = None, results_file_path=None,
    allowed_gt_roots=None,
) -> dict:
    """Record a visual benchmark comparison, never proof of scientific correctness.

    Missing output or reference is explicitly failed/skipped. An omitted model
    uses the registry default; callers should pass their experiment's eval model.
    """
    task_name = infer_task_name(sample_data, default=task_name)
    model_name = (model_name or get_default_model("text")).strip()
    if not model_name:
        return _mark_not_evaluated(sample_data, "failed", "Error", "No evaluation model configured.")
    sample_data["evaluation_model_name"] = model_name
    sample_data.setdefault("scientific_verification_status", "unverified")
    if not sample_data.get("path_to_gt_image"):
        return _mark_not_evaluated(sample_data, "skipped", "N/A - No GT", "No ground truth image path provided.")
    if sample_data.get("status") in {"failed", "cancelled"}:
        return _mark_not_evaluated(sample_data, "skipped", "N/A - Generation Failed", "Generation did not succeed.")
    selection = resolve_final_output(sample_data, task_name=task_name)
    if not selection.image_key:
        return _mark_not_evaluated(sample_data, "skipped", "N/A - Generation Failed", "No generated image available.")
    sample_data["eval_image_field"] = selection.image_key
    reference = resolve_gt_image_path(sample_data, task_name=task_name,
                                      results_file_path=results_file_path, repo_root=work_dir)
    if reference is None:
        return _mark_not_evaluated(sample_data, "failed", "Error", "Ground truth image was not found.")
    from pathlib import Path
    root = Path(work_dir) if work_dir else Path(__file__).resolve().parents[1]
    allowed_roots = [root / "data" / "PaperBananaBench" / task_name, root / "results"]
    if results_file_path:
        allowed_roots.append(Path(results_file_path).resolve().parent)
    if allowed_gt_roots is not None:
        allowed_roots.extend(Path(path) for path in allowed_gt_roots)
    try:
        resolved_reference = reference.resolve(strict=True)
        if not any(resolved_reference.is_relative_to(path.resolve()) for path in allowed_roots):
            return _mark_not_evaluated(sample_data, "failed", "Error", "Ground truth image is outside configured dataset/results roots.")
        reference = resolved_reference
    except OSError:
        return _mark_not_evaluated(sample_data, "failed", "Error", "Ground truth image is unreadable.")
    try:
        reference_bytes = reference.read_bytes()
        image_mime_type(reference_bytes)
        gt_image_base64 = base64.b64encode(reference_bytes).decode("ascii")
        model_bytes = final_image_bytes(sample_data)
        if model_bytes is None:
            raise ValueError("Generated image is unreadable")
        model_image_base64 = base64.b64encode(model_bytes).decode("ascii")
    except (OSError, ValueError):
        return _mark_not_evaluated(sample_data, "failed", "Error", "An evaluation image is unreadable or invalid.")
    raw_content = sample_data.get("content", "")
    if not isinstance(raw_content, str):
        raw_content = json.dumps(raw_content, ensure_ascii=False)
    results = await asyncio.gather(*[
        _run_single_eval_ref(task_name, dim, raw_content,
                             sample_data.get("visual_intent", sample_data.get("caption", "")),
                             gt_image_base64, model_image_base64, model_name, provider, overrides)
        for dim in _DIMENSIONS
    ])
    failed_dimensions = []
    for dimension, response in results:
        sample_data[f"{dimension}_reasoning"] = response["comparison_reasoning"]
        sample_data[f"{dimension}_outcome"] = response["winner"]
        if response["status"] != "completed":
            failed_dimensions.append(dimension)
    if failed_dimensions:
        sample_data["evaluation_status"] = "failed"
        sample_data["evaluation_failed_dimensions"] = failed_dimensions
        sample_data["overall_outcome"] = "Error"
        sample_data["overall_reasoning"] = "Incomplete evaluation; no overall comparison outcome."
        return sample_data
    tier1 = _determine_tier_outcome(sample_data["faithfulness_outcome"], sample_data["readability_outcome"])
    tier2 = _determine_tier_outcome(sample_data["conciseness_outcome"], sample_data["aesthetics_outcome"])
    sample_data["overall_outcome"] = tier1 if tier1 in {"Model", "Human"} else tier2
    sample_data["overall_reasoning"] = f"Visual comparison aggregation: Tier 1={tier1}; Tier 2={tier2}. Scientific correctness remains separately unverified."
    sample_data["evaluation_status"] = "completed"
    sample_data.pop("evaluation_error", None)
    sample_data.pop("evaluation_failed_dimensions", None)
    return sample_data
