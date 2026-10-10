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

"""Direct image or explicitly trusted generated-plot rendering."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from utils import generation_utils, image_utils
from utils.legacy_generation_options import image_size_from_data
from utils.model_registry import resolve_model
from utils.plot_execution import execute_plot_code_async, LegacyExecutorHandle
from .base_agent import BaseAgent


class VanillaAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        plot = self.exp_config.task_name == "plot"
        self.model_name = self.exp_config.main_model_name if plot else self.exp_config.image_gen_model_name
        self.system_prompt = PLOT_VANILLA_AGENT_SYSTEM_PROMPT if plot else DIAGRAM_VANILLA_AGENT_SYSTEM_PROMPT
        self.process_executor = LegacyExecutorHandle() if plot else None
        self.task_config = {"task_name": "plot" if plot else "diagram", "use_image_generation": not plot,
                            "content_label": "Plot Raw Data" if plot else "Method Section",
                            "visual_intent_label": "Visual Intent of the Desired Plot" if plot else "Diagram Caption"}

    def close(self):
        if self.process_executor is not None:
            self.process_executor.shutdown(wait=False)
            self.process_executor = None

    async def aclose(self):
        self.close()

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.task_config
        plot = not cfg["use_image_generation"]
        raw = data["content"]
        content = json.dumps(raw) if isinstance(raw, (dict, list)) else str(raw)
        prompt = f"{cfg['content_label']}: {content}\n{cfg['visual_intent_label']}: {data['visual_intent']}\n"
        prompt += "Write only Python matplotlib code, preserving all supplied values exactly." if plot else "Render the diagram. Do not render the figure caption or title."
        system = self.system_prompt
        provider = getattr(self.exp_config, "main_model_provider" if plot else "image_gen_provider", None) or None
        overrides = getattr(self.exp_config, "model_profiles", None)
        timeout = getattr(self.exp_config, "candidate_timeout_seconds", 600) or 600
        rfs_enabled = bool(getattr(self.exp_config, "rfs_enabled", False) or data.get("render_mode") == "rfs")
        metadata, spec = {}, None
        if rfs_enabled:
            from rfs.prompt_compiler import compile_prompt
            spec = data.get("figure_spec")
            if not isinstance(spec, dict):
                raise generation_utils.ModelConfigurationError("RFS vanilla mode requires an explicit FigureSpec")
            profile = resolve_model(self.model_name, provider, "text" if plot else "image_generate", overrides)
            compiled = compile_prompt(spec, prompt, profile.compiler_capabilities(), candidate_id=data.get("candidate_id", "0"), stage="candidate")
            prompt, system, metadata = compiled.prompt, compiled.system_prompt, compiled.metadata
            if plot:
                prompt += "\n\nWrite only Python matplotlib code. Use all supplied dataset values exactly; do not invent measurements."
        options = {"system_prompt": system or "", "temperature": self.exp_config.temperature,
                   "candidate_count": 1, "max_output_tokens": 8192}
        stem = f"vanilla_{cfg['task_name']}"
        if plot:
            responses = await generation_utils.call_model_with_retry_async(self.model_name, [{"type": "text", "text": prompt}], options,
                provider=provider, overrides=overrides, timeout=timeout, max_attempts=3, retry_delay=1, total_timeout=timeout)
            data[stem + "_code"] = responses[0]
            encoded = await execute_plot_code_async(responses[0], dpi=300,
                timeout=self.exp_config.plot_timeout_seconds, allow_execution=self.exp_config.allow_generated_plot_code)
            if not encoded:
                raise generation_utils.GenerationResponseError("Generated code produced no plot", category="plot_render")
            data[stem + "_base64_jpg"] = encoded
            mime = "image/jpeg"
            result = None
        else:
            options.update(aspect_ratio=data.get("additional_info", {}).get("rounded_ratio", "1:1"),
                           image_size=image_size_from_data(data), output_format="png", background="opaque")
            if rfs_enabled:
                result = await generation_utils.generate_images(self.model_name, [{"type": "text", "text": prompt}], options,
                    provider=provider, overrides=overrides, timeout=timeout, max_attempts=3, retry_delay=1, total_timeout=timeout)
                sample = result.samples[-1]
                encoded, mime = sample.image_base64, sample.image_mime
            else:
                responses = await generation_utils.call_image_with_retry_async(self.model_name, [{"type": "text", "text": prompt}], options,
                    provider=provider, overrides=overrides, timeout=timeout, max_attempts=3, retry_delay=1, total_timeout=timeout)
                encoded = responses[-1]
                mime = generation_utils.image_metadata(encoded)["mime_type"]
                result = None
            preview = await asyncio.to_thread(image_utils.convert_png_b64_to_jpg_b64, encoded)
            if not preview:
                raise generation_utils.GenerationResponseError("Image preview conversion failed", category="image_decode")
            data[stem + "_base64_jpg"] = preview
        data[stem + "_base64"] = encoded
        data[stem + "_image_mime"] = mime
        data[stem + "_base64_png"] = generation_utils.lossless_png_base64(encoded)
        if rfs_enabled:
            from .visualizer_agent import VisualizerAgent
            if result is None:
                profile = resolve_model(self.model_name, provider, "text", overrides)
                result = generation_utils.GenerationResult([], profile.provider, profile.model_id,
                            effective_prompt=prompt, effective_system_prompt=system, options=options)
            VisualizerAgent._record_artifact(self, data, stem, result, encoded, mime, spec,
                data.get("rfs_design_description", content), "candidate", metadata,
                code=data.get(stem + "_code"))
        return data


DIAGRAM_VANILLA_AGENT_SYSTEM_PROMPT = """
## ROLE
You are a Lead Visual Designer for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You will be provided with a "Method Section" and a "Diagram Caption". Your task is to generate a high-quality scientific diagram that effectively illustrates the method described in the text, as the caption requires, and adhering strictly to modern academic visualization standards.

**CRITICAL INSTRUCTION ON CAPTION:**
The "Diagram Caption" is provided solely to describe the visual content and logic you need to draw. **DO NOT render, write, or include the caption text itself (e.g., "Figure 1: ...") inside the generated image.**

## INPUT DATA
-   **Method Section**: [Content of method section]
-   **Diagram Caption**: [Diagram caption]
## OUTPUT
Generate a single, high-resolution image that visually explains the method and aligns well with the caption. 
"""

PLOT_VANILLA_AGENT_SYSTEM_PROMPT = """
## ROLE
You are an expert statistical plot illustrator for top-tier AI conferences (e.g., NeurIPS 2025).

## TASK
You will be provided with "Plot Raw Data" and a "Visual Intent of the Desired Plot". Your task is to write matplotlib code to generate a high-quality statistical plot that effectively visualizes the data according to the visual intent, adhering strictly to modern academic visualization standards.

## INPUT DATA
-   **Plot Raw Data**: [Raw data to be visualized]
-   **Visual Intent of the Desired Plot**: [Description of what the plot should convey]

## OUTPUT
Write Python matplotlib code to generate the plot. Only provide the code without any explanations.
"""
