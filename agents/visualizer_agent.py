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

"""Render legacy and RFS figures through explicit provider adapters."""
from __future__ import annotations

import asyncio
import base64
import copy
import re
import hashlib
from typing import Any, Dict

from utils import generation_utils, image_utils
from utils.legacy_generation_options import image_size_from_data
from utils.model_registry import resolve_model
from utils.plot_execution import execute_plot_code_async, LegacyExecutorHandle
from .base_agent import BaseAgent


class VisualizerAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        plot = self.exp_config.task_name == "plot"
        self.model_name = self.exp_config.main_model_name if plot else self.exp_config.image_gen_model_name
        self.system_prompt = PLOT_VISUALIZER_AGENT_SYSTEM_PROMPT if plot else DIAGRAM_VISUALIZER_AGENT_SYSTEM_PROMPT
        self.process_executor = LegacyExecutorHandle() if plot else None
        self.task_config = {
            "task_name": "plot" if plot else "diagram", "use_image_generation": not plot,
            "prompt_template": "Use python matplotlib to generate a statistical plot based on the following detailed description: {desc}\nOnly provide the code without explanations. Code:" if plot else
                               "Render an image based on the following detailed description: {desc}\nDo not include figure titles in the image. Diagram:",
        }

    def close(self):
        if self.process_executor is not None:
            self.process_executor.shutdown(wait=False)
            self.process_executor = None

    async def aclose(self):
        self.close()

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        plot = not self.task_config["use_image_generation"]
        task = self.task_config["task_name"]
        rfs_enabled = bool(getattr(self.exp_config, "rfs_enabled", False) or data.get("render_mode") == "rfs")
        provider = getattr(self.exp_config, "main_model_provider" if plot else "image_gen_provider", None) or None
        overrides = getattr(self.exp_config, "model_profiles", None)
        timeout = getattr(self.exp_config, "candidate_timeout_seconds", 600) or 600
        keys = [f"target_{task}_desc0", f"target_{task}_stylist_desc0"]
        critic_pattern = re.compile(rf"^target_{re.escape(task)}_critic_desc([0-9]+)$")
        critic_keys = [(int(match.group(1)), key) for key in data for match in [critic_pattern.fullmatch(key)] if match]
        keys.extend(key for _, key in sorted(critic_keys))
        for desc_key in keys:
            if desc_key not in data or f"{desc_key}_base64_jpg" in data:
                continue
            description = data[desc_key]
            if not isinstance(description, str) or not description.strip():
                raise generation_utils.ModelConfigurationError("A nonempty figure design description is required")
            prompt = self.task_config["prompt_template"].format(desc=description)
            system = self.system_prompt
            metadata = {}
            spec = None
            stage = "candidate" if "stylist" in desc_key else "design"
            operation = "image_generate"
            reference_key = None
            content_list = []
            if "critic_desc" in desc_key:
                round_idx = int(desc_key.rsplit("critic_desc", 1)[1])
                reference_key = f"target_{task}_critic_desc{round_idx - 1}" if round_idx > 0 else f"target_{task}_stylist_desc0"
                if f"{reference_key}_base64_jpg" not in data:
                    reference_key = f"target_{task}_desc0"
                suggestions = data.get(f"target_{task}_critic_suggestions{round_idx}", "")
                if suggestions.strip().rstrip(".") == "No changes needed":
                    for suffix in ("_base64_jpg", "_base64_png", "_base64", "_image_mime", "_code"):
                        if reference_key + suffix in data:
                            data[desc_key + suffix] = data[reference_key + suffix]
                    artifacts = data.get("rfs_artifacts", {})
                    if reference_key in artifacts:
                        artifacts[desc_key] = copy.deepcopy(artifacts[reference_key])
                        artifacts[desc_key]["provenance"]["reused_from_stage"] = reference_key
                    if desc_key + "_base64_jpg" in data:
                        continue
                if rfs_enabled:
                    stage = "code_repair" if plot else "local_repair"
                    operation = "image_generate" if plot else "image_edit"
            if rfs_enabled:
                from rfs.prompt_compiler import compile_prompt
                spec = copy.deepcopy(data.get("rfs_stage_specs", {}).get(desc_key, data.get("figure_spec")))
                if not isinstance(spec, dict):
                    raise generation_utils.ModelConfigurationError("RFS rendering requires a validated FigureSpec")
                profile = resolve_model(self.model_name, provider, "text" if plot else operation, overrides)
                compiled = compile_prompt(spec, description, profile.compiler_capabilities(),
                    candidate_id=data.get("candidate_id", "0"), stage=stage,
                    repair=data.get("rfs_repair") if stage in {"local_repair", "code_repair"} else None)
                prompt, system, metadata = compiled.prompt, compiled.system_prompt, compiled.metadata
                if plot:
                    prompt += "\n\nREPRODUCIBLE PLOT OUTPUT\nWrite only Python matplotlib code. Use every supplied dataset value exactly. Never invent, sample or smooth measurements."
                    if stage == "code_repair" and (not reference_key or not data.get(reference_key + "_code")):
                        raise generation_utils.ModelConfigurationError("Code repair requires the previous plot source code")
                    if reference_key and data.get(reference_key + "_code"):
                        prompt += "\nApply only the described presentation correction to this previous code:\n" + data[reference_key + "_code"]
            content_list.append({"type": "text", "text": prompt})
            if operation == "image_edit":
                artifacts = data.get("rfs_artifacts", {})
                previous = artifacts.get(reference_key, {})
                encoded = previous.get("image_base64") or data.get(reference_key + "_base64")
                if not encoded:
                    raise generation_utils.ModelConfigurationError("Local repair requires the previous original image artifact")
                content_list.append({"type": "image", "data": encoded, "mime_type": previous.get("image_mime", "image/png")})
            options = {"system_prompt": system or "", "temperature": self.exp_config.temperature,
                       "candidate_count": 1, "max_output_tokens": 8192}
            if plot:
                responses = await generation_utils.call_model_with_retry_async(self.model_name, content_list, options,
                    provider=provider, overrides=overrides, timeout=timeout, max_attempts=3, retry_delay=1, total_timeout=timeout)
                code = responses[0]
                data[desc_key + "_code"] = code
                encoded = await execute_plot_code_async(code, dpi=300,
                    timeout=self.exp_config.plot_timeout_seconds, allow_execution=self.exp_config.allow_generated_plot_code)
                if not encoded:
                    raise generation_utils.GenerationResponseError("Generated plot code produced no image", category="plot_render")
                data[desc_key + "_base64_jpg"] = encoded
                data[desc_key + "_base64"] = encoded
                data[desc_key + "_image_mime"] = "image/jpeg"
                data[desc_key + "_base64_png"] = generation_utils.lossless_png_base64(encoded)
                if rfs_enabled:
                    profile = resolve_model(self.model_name, provider, "text", overrides)
                    result = generation_utils.GenerationResult([], profile.provider, profile.model_id,
                                                              effective_prompt=prompt, effective_system_prompt=system, options=options)
                    self._record_artifact(data, desc_key, result, encoded, "image/jpeg", spec, description, stage, metadata, code=code)
            else:
                options.update(aspect_ratio=data.get("additional_info", {}).get("rounded_ratio", "1:1"),
                               image_size=image_size_from_data(data), output_format="png", background="opaque")
                if rfs_enabled:
                    result = await generation_utils.generate_images(self.model_name, content_list, options, provider=provider,
                        operation=operation, overrides=overrides, timeout=timeout, max_attempts=3, retry_delay=1, total_timeout=timeout)
                    sample = result.samples[-1]
                    encoded = sample.image_base64
                    data[desc_key + "_base64"] = encoded
                    data[desc_key + "_image_mime"] = sample.image_mime
                    data[desc_key + "_base64_png"] = generation_utils.lossless_png_base64(encoded)
                    self._record_artifact(data, desc_key, result, encoded, sample.image_mime, spec, description, stage, metadata,
                                          reference_key=reference_key if operation == "image_edit" else None)
                else:
                    responses = await generation_utils.call_image_with_retry_async(self.model_name, content_list, options,
                        provider=provider, overrides=overrides, timeout=timeout, max_attempts=3, retry_delay=1, total_timeout=timeout)
                    encoded = responses[-1]
                    # Original-image fields never replace the legacy JPEG preview key.
                    meta = generation_utils.image_metadata(encoded)
                    data[desc_key + "_base64"] = encoded
                    data[desc_key + "_image_mime"] = meta["mime_type"]
                    data[desc_key + "_base64_png"] = generation_utils.lossless_png_base64(encoded)
                preview = await asyncio.to_thread(image_utils.convert_png_b64_to_jpg_b64, encoded)
                if not preview:
                    raise generation_utils.GenerationResponseError("Image preview conversion failed", category="image_decode")
                data[desc_key + "_base64_jpg"] = preview
        return data

    def _record_artifact(self, data, key, result, encoded, mime, spec, description, stage, metadata,
                         *, reference_key=None, code=None):
        provenance = {"scientific_verification": spec.get("provenance", {}).get("scientific_verification", "unverified"),
                      "external_consumer_verified": False, "render_method": "generated_matplotlib" if code else "provider_image",
                      "original_bytes_preserved": True, "png_conversion": "original" if mime == "image/png" else "lossless_container_from_original"}
        if reference_key:
            provenance["reference_stage"] = reference_key
        record = {"candidate_id": str(data.get("candidate_id", "0")), "stage": stage,
                  "provider": result.provider, "model": result.model,
                  "prompt": result.effective_prompt, "system_prompt": result.effective_system_prompt,
                  "design_description": description, "figure_spec": copy.deepcopy(spec),
                  "figure_spec_sha256": metadata.get("figure_spec_sha256"), "image_base64": encoded,
                  "image_mime": mime, "png_base64": data[key + "_base64_png"], "options": copy.deepcopy(result.options),
                  "provenance": provenance, "compiler_metadata": metadata,
                  "request_id": result.request_id, "usage": result.usage, "attempts": result.attempts}
        record["submitted_prompt_sha256"] = hashlib.sha256((result.effective_prompt or "").encode("utf-8")).hexdigest()
        if code:
            record["code"] = code
        data.setdefault("rfs_artifacts", {})[key] = record


DIAGRAM_VISUALIZER_AGENT_SYSTEM_PROMPT = """You are an expert scientific diagram illustrator. Generate high-quality scientific diagrams based on user requests."""

PLOT_VISUALIZER_AGENT_SYSTEM_PROMPT = """You are an expert statistical plot illustrator. Write code to generate high-quality statistical plots based on user requests."""


# !!! Note: If using image generation models, use the following system prompt instead:

# PLOT_VISUALIZER_AGENT_SYSTEM_PROMPT = """You are an expert statistical plot illustrator. Generate high-quality statistical plots based on user requests. Note that you should not use code, but directly generate the image."""
