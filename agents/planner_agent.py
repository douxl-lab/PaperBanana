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
Vanilla Agent - Directly rendering images based on the method section.
"""

import json
from typing import Dict, Any
from google.genai import types
import base64, io, asyncio
from PIL import Image

from utils import generation_utils
from .base_agent import BaseAgent
from rfs.contract import is_rfs_enabled, validate_relative_path, RFSValidationError
from rfs.agent_contract import STRUCTURED_SYSTEM_SUPPLEMENT, planner_instructions, apply_planner_response


class PlannerAgent(BaseAgent):
    """Planner Agent to generate images based on user queries"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.exp_config.main_model_name

        # Task-specific configurations
        if "plot" in self.exp_config.task_name:
            self.system_prompt = build_planner_system_prompt(
                task_name="plot",
                planner_metaphor=self.exp_config.planner_metaphor,
            )
            self.task_config = {
                "task_name": "plot",
                "content_label": "Plot Raw Data",
                "visual_intent_label": "Visual Intent of the Desired Plot",
            }
        else:
            self.system_prompt = build_planner_system_prompt(
                task_name="diagram",
                planner_metaphor=self.exp_config.planner_metaphor,
            )
            self.task_config = {
                "task_name": "diagram",
                "content_label": "Methodology Section",
                "visual_intent_label": "Diagram Caption",
            }

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unified processing method that works for both diagram and plot tasks.
        Uses task_config to determine task-specific parameters.
        Expects data['top10_references'] to be already populated by retriever.
        """
        cfg = self.task_config
        
        raw_content = data["content"]
        content = json.dumps(raw_content) if isinstance(raw_content, (dict, list)) else raw_content
        description = data["visual_intent"]

        content_list = []
        
        # Check if retriever has already provided full examples (e.g., in manual mode)
        examples = data.get("retrieved_examples", [])
        if not examples:
            retrieved_ids = data.get("top10_references", [])
            if retrieved_ids:
                try:
                    with open(self.exp_config.work_dir / f"data/PaperBananaBench/{cfg['task_name']}/ref.json", "r", encoding="utf-8") as f:
                        candidate_pool = json.load(f)
                    if not isinstance(candidate_pool, list):
                        raise ValueError("Reference metadata must be a list")
                    id_to_item = {item["id"]: item for item in candidate_pool
                                  if isinstance(item, dict) and isinstance(item.get("id"), (str, int))}
                    examples = [id_to_item[ref_id] for ref_id in retrieved_ids
                                if isinstance(ref_id, (str, int)) and ref_id in id_to_item]
                except (OSError, ValueError, TypeError) as exc:
                    data.setdefault("omitted_references", []).append({"reason": type(exc).__name__, "stage": "reference_metadata"})
        
        user_prompt = ""
        used_reference_chars = 0
        for idx, item in enumerate(examples):
            # Reference entries are external data, never arbitrary filesystem authority.
            try:
                relative_path = validate_relative_path(item["path_to_gt_image"])
                reference_root = (self.exp_config.work_dir / f"data/PaperBananaBench/{cfg['task_name']}").resolve()
                image_path = (reference_root / relative_path).resolve()
                if not image_path.is_relative_to(reference_root):
                    raise RFSValidationError("Reference path escapes the reference dataset")
                if image_path.stat().st_size > 16 * 1024 * 1024:
                    raise RFSValidationError("Reference image exceeds 16 MB budget")
                with image_path.open("rb") as f:
                    ref_image_base64 = base64.b64encode(f.read()).decode("utf-8")
                item_content = item["content"]
                item_intent = item["visual_intent"]
                reference_chars = len(str(item_content)) + len(str(item_intent))
                budget = getattr(self.exp_config, "reference_prompt_budget_chars", 100_000)
                if used_reference_chars + reference_chars > budget:
                    raise RFSValidationError("Reference text prompt budget exceeded")
                used_reference_chars += reference_chars
            except (KeyError, TypeError, OSError, RFSValidationError) as exc:
                data.setdefault("omitted_references", []).append({"index": idx, "reason": type(exc).__name__})
                continue
            user_prompt += f"Example {idx+1}:\n"
            
            item_content = item["content"]
            if isinstance(item_content, (dict, list)):
                item_content = json.dumps(item_content)
            
            user_prompt += f"{cfg['content_label']}: {item_content}\n"
            user_prompt += f"{cfg['visual_intent_label']}: {item_intent}\nReference {cfg['task_name'].capitalize()}: "
            content_list.append({"type": "text", "text": user_prompt})
            
            content_list.append({"type": "image", "image_base64": ref_image_base64})
            user_prompt = ""

        user_prompt += f"Now, based on the following {cfg['content_label'].lower()} and {cfg['visual_intent_label'].lower()}, provide a detailed description for the figure to be generated.\n"
        user_prompt += f"{cfg['content_label']}: {content}\n{cfg['visual_intent_label']}: {description}\n"
        user_prompt += "Detailed description of the target figure to be generated"
        if cfg["task_name"] == "diagram":
            user_prompt += " (do not include figure titles)"
        user_prompt += ":"

        rfs_mode = is_rfs_enabled(data, self.exp_config)
        locked_spec = None
        if rfs_mode:
            rfs_instructions, locked_spec = planner_instructions(data, task_name=cfg["task_name"])
            user_prompt += "\n" + rfs_instructions
        content_list.append({"type": "text", "text": user_prompt})

        response_list = await generation_utils.call_model_with_retry_async(
            model_name=self.model_name,
            contents=content_list,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt + (STRUCTURED_SYSTEM_SUPPLEMENT if rfs_mode else ""),
                temperature=self.exp_config.temperature,
                candidate_count=1,
                max_output_tokens=50000,
            ),
            max_attempts=5,
            retry_delay=5,
            provider=getattr(self.exp_config, "main_model_provider", None) or None,
            overrides=getattr(self.exp_config, "model_profiles", None),
            timeout=getattr(self.exp_config, "candidate_timeout_seconds", 600),
            total_timeout=getattr(self.exp_config, "candidate_timeout_seconds", 600),
        )
        
        if rfs_mode:
            if not response_list:
                raise RFSValidationError("Planner returned no structured response")
            for idx, response in enumerate(response_list):
                apply_planner_response(data, response, locked_spec=locked_spec, desc_key=f"target_{cfg['task_name']}_desc{idx}")
        else:
            for idx, response in enumerate(response_list):
                data[f"target_{cfg['task_name']}_desc{idx}"] = response.strip()

        return data




def build_planner_system_prompt(task_name: str, planner_metaphor: bool = False) -> str:
    """Return the Planner system prompt, preserving defaults byte-for-byte."""
    if "plot" in task_name:
        return PLOT_PLANNER_AGENT_SYSTEM_PROMPT
    if planner_metaphor:
        return DIAGRAM_PLANNER_AGENT_SYSTEM_PROMPT + DIAGRAM_PLANNER_METAPHOR_SUPPLEMENT
    return DIAGRAM_PLANNER_AGENT_SYSTEM_PROMPT


DIAGRAM_PLANNER_AGENT_SYSTEM_PROMPT = """
I am working on a task: given the 'Methodology' section of a paper, and the caption of the desired figure, automatically generate a corresponding illustrative diagram. I will input the text of the 'Methodology' section, the figure caption, and your output should be a detailed description of an illustrative figure that effectively represents the methods described in the text.

To help you understand the task better, and grasp the principles for generating such figures, I will also provide you with several examples. You should learn from these examples to provide your figure description.

** IMPORTANT: **
Your description should be as detailed as possible. Semantically, clearly describe each element and their connections. Formally, include various details such as background style (typically pure white or very light pastel), colors, line thickness, icon styles, etc. Remember: vague or unclear specifications will only make the generated figure worse, not better.
"""

DIAGRAM_PLANNER_METAPHOR_SUPPLEMENT = """

Planner metaphor mode (diagram-only): Before producing the detailed description, identify a compact visual metaphor that can organize the diagram's layout and relationships, such as a pipeline, map, layered stack, control loop, branching tree, or hub-and-spoke network. Use the metaphor only as a conceptual guide for the diagram composition.

Return the same detailed textual figure-description output requested above. Do not output SVG, code, image markup, a separate sketch, or rendering instructions beyond the detailed description.
"""

PLOT_PLANNER_AGENT_SYSTEM_PROMPT = """
I am working on a task: given the raw data (typically in tabular or json format) and a visual intent of the desired plot, automatically generate a corresponding statistical plot that are both accurate and aesthetically pleasing. I will input the raw data and the plot visual intent, and your output should be a detailed description of an illustrative plot that effectively represents the data.  Note that your description should include all the raw data points to be plotted.

To help you understand the task better, and grasp the principles for generating such plots, I will also provide you with several examples. You should learn from these examples to provide your plot description.

** IMPORTANT: **
Your description should be as detailed as possible. For content, explain the precise mapping of variables to visual channels (x, y, hue) and explicitly enumerate every raw data point's coordinate to be drawn to ensure accuracy. For presentation, specify the exact aesthetic parameters, including specific HEX color codes, font sizes for all labels, line widths, marker dimensions, legend placement, and grid styles. You should learn from the examples' content presentation and aesthetic design (e.g., color schemes).
"""
