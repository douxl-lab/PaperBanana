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

"""Reference-image polishing through capability-checked editing adapters."""
from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any, Dict

from utils import generation_utils, image_utils
from utils.legacy_generation_options import image_size_from_data
from utils.model_registry import resolve_model
from .base_agent import BaseAgent


def _load_image_as_base64(image_path: str) -> str:
    return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")


class PolishAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_gen_model_name = self.exp_config.image_gen_model_name
        self.main_model_name = self.exp_config.main_model_name
        plot = self.exp_config.task_name == "plot"
        self.style_guide_filename = "neurips2025_plot_style_guide.md" if plot else "neurips2025_diagram_style_guide.md"
        self.suggestion_system_prompt = PLOT_SUGGESTION_SYSTEM_PROMPT if plot else DIAGRAM_SUGGESTION_SYSTEM_PROMPT
        self.task_config = {"task_name": "plot" if plot else "diagram"}

    def close(self):
        pass

    async def aclose(self):
        self.close()

    async def _generate_suggestions(self, gt_image_b64, style_guide):
        prompt = f"Here is the style guide:\n{style_guide}\n\nAnalyze the reference image. List up to 10 presentation improvements. Preserve exact labels, scientific topology and data. If no changes are needed, say 'No changes needed'."
        responses = await generation_utils.call_model_with_retry_async(self.main_model_name,
            [{"type": "text", "text": prompt}, {"type": "image", "data": gt_image_b64}],
            {"system_prompt": self.suggestion_system_prompt, "temperature": self.exp_config.temperature,
             "candidate_count": 1, "max_output_tokens": 8192},
            provider=getattr(self.exp_config, "main_model_provider", None) or None,
            overrides=getattr(self.exp_config, "model_profiles", None), max_attempts=3, retry_delay=1,
            timeout=self.exp_config.candidate_timeout_seconds, total_timeout=self.exp_config.candidate_timeout_seconds)
        return responses[0]

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        task = self.task_config["task_name"]
        relative = data.get("path_to_gt_image")
        if not relative:
            raise generation_utils.ModelConfigurationError("Polishing requires path_to_gt_image")
        root = (self.exp_config.work_dir / "data" / "PaperBananaBench" / task).resolve()
        image_path = (root / relative).resolve()
        if not image_path.is_relative_to(root):
            raise generation_utils.ModelConfigurationError("Ground-truth image path escapes the dataset")
        original = _load_image_as_base64(str(image_path))
        meta = generation_utils.image_metadata(original)
        style_path = getattr(self.exp_config, "style_guide_path", None) or self.exp_config.work_dir / "style_guides" / self.style_guide_filename
        style_guide = Path(style_path).read_text(encoding="utf-8")
        suggestions = await self._generate_suggestions(original, style_guide)
        data[f"suggestions_{task}"] = suggestions
        stem = f"polished_{task}"
        if suggestions.strip().rstrip(".") == "No changes needed":
            encoded, mime = original, meta["mime_type"]
            result = None
        else:
            provider = getattr(self.exp_config, "image_gen_provider", None) or None
            overrides = getattr(self.exp_config, "model_profiles", None)
            profile = resolve_model(self.image_gen_model_name, provider, "image_edit", overrides)
            prompt = f"Polish only the described presentation details. Preserve all scientific content and exact labels.\nSuggestions:\n{suggestions}"
            system = "You are a scientific figure editor. Preserve reference image content and data."
            metadata, spec = {}, data.get("figure_spec")
            if getattr(self.exp_config, "rfs_enabled", False):
                from rfs.prompt_compiler import compile_prompt
                if not isinstance(spec, dict):
                    raise generation_utils.ModelConfigurationError("RFS polish requires an explicit FigureSpec")
                compiled = compile_prompt(spec, prompt, profile.compiler_capabilities(), candidate_id=data.get("candidate_id", "0"), stage="candidate")
                prompt, system, metadata = compiled.prompt, compiled.system_prompt, compiled.metadata
            options = {"system_prompt": system or "", "temperature": self.exp_config.temperature,
                       "aspect_ratio": data.get("additional_info", {}).get("rounded_ratio", "1:1"),
                       "image_size": image_size_from_data(data), "output_format": "png", "background": "opaque"}
            result = await generation_utils.generate_images(self.image_gen_model_name,
                [{"type": "text", "text": prompt}, {"type": "image", "data": original, "mime_type": meta["mime_type"]}],
                options, operation="image_edit", provider=provider, overrides=overrides, max_attempts=3, retry_delay=1,
                timeout=self.exp_config.candidate_timeout_seconds, total_timeout=self.exp_config.candidate_timeout_seconds)
            sample = result.samples[-1]
            encoded, mime = sample.image_base64, sample.image_mime
        data[stem + "_base64"] = encoded
        data[stem + "_image_mime"] = mime
        data[stem + "_base64_png"] = generation_utils.lossless_png_base64(encoded)
        preview = await asyncio.to_thread(image_utils.convert_png_b64_to_jpg_b64, encoded)
        if not preview:
            raise generation_utils.GenerationResponseError("Polish preview conversion failed", category="image_decode")
        data[stem + "_base64_jpg"] = preview
        if getattr(self.exp_config, "rfs_enabled", False) and result is not None:
            from .visualizer_agent import VisualizerAgent
            VisualizerAgent._record_artifact(self, data, stem, result, encoded, mime, spec, suggestions, "candidate", metadata,
                                            reference_key="ground_truth")
        return data


DIAGRAM_SUGGESTION_SYSTEM_PROMPT = """
You are a senior art director for NeurIPS 2025. Your task is to critique a diagram against a provided style guide.
Provide up to 10 concise, actionable improvement suggestions. Focus on aesthetics (color, layout, fonts, icons).
Directly list the suggestions. Do not use filler phrases like "Based on the style guide...".
If the diagram is substantially compliant, output "No changes needed".
"""

PLOT_SUGGESTION_SYSTEM_PROMPT = """
You are a senior data visualization expert for NeurIPS 2025. Your task is to critique a plot against a provided style guide.
Provide up to 10 concise, actionable improvement suggestions. Focus on aesthetics (color, layout, fonts).
Directly list the suggestions. Do not use filler phrases like "Based on the style guide...".
If the plot is substantially compliant, output "No changes needed".
"""

DIAGRAM_POLISH_AGENT_SYSTEM_PROMPT = """
## ROLE
You are a professional diagram polishing expert for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are given an existing diagram image and a list of specific improvement suggestions. Your task is to generate a polished version of this diagram by applying these suggestions while preserving the semantic logic and structure of the original diagram.

## OUTPUT
Generate a polished diagram image that maintains the original content while applying the improvement suggestions.
"""

PLOT_POLISH_AGENT_SYSTEM_PROMPT = """
## ROLE
You are a professional plot polishing expert for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You are given an existing statistical plot image and a list of specific improvement suggestions. Your task is to generate a polished version of this plot by applying these suggestions while preserving all the data and quantitative information.

**Important Instructions:**
1. **Preserve Data:** Do NOT alter any data points, values, or quantitative information in the plot.
2. **Apply Suggestions:** Enhance the visual aesthetics according to the provided suggestions (colors, fonts, layout, etc.).
3. **Maintain Accuracy:** Ensure all numerical values and relationships remain accurate.
4. **Professional Quality:** Ensure the output meets publication standards for top-tier conferences.

## OUTPUT
Generate a polished plot image that maintains the original data while applying the improvement suggestions.
"""
