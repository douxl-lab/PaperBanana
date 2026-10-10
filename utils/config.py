# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Predictable configuration without process-wide timezone changes."""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.legacy_generation_options import is_plot_task

EXPERIMENT_MODES = (
    "vanilla", "dev_planner", "dev_planner_stylist", "dev_planner_critic",
    "demo_planner_critic", "dev_full", "demo_full", "dev_polish", "dev_retriever",
)
MAX_CONCURRENCY = 32


def _settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import yaml
    try:
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream) or {}
    except yaml.YAMLError as exc:
        raise ValueError("Invalid YAML configuration") from exc
    if not isinstance(value, dict) or not isinstance(value.get("defaults", {}), dict):
        raise ValueError("Configuration must contain a mapping and a defaults mapping")
    # Never expose credentials in errors or carry the api_keys section into results.
    return value.get("defaults", {})


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


@dataclass
class ExpConfig:
    dataset_name: str
    task_name: str = "diagram"
    split_name: str = "test"
    temperature: float | None = None
    exp_mode: str = ""
    retrieval_setting: str = "auto"
    planner_metaphor: bool = False
    max_critic_rounds: int | None = None
    main_model_name: str = ""
    image_gen_model_name: str = ""
    work_dir: Path = Path(__file__).parent.parent
    timestamp: str | None = None
    config_path: Path | str | None = None
    main_model_provider: str = ""
    image_gen_provider: str = ""
    evaluation_model_name: str = ""
    model_profiles: dict = field(default_factory=dict)
    render_mode: str = "legacy"
    max_concurrent: int | None = None
    candidate_timeout_seconds: float | None = None
    plot_timeout_seconds: float | None = None
    # The library preserves legacy trusted-code behavior. RFS callers and the CLI
    # must explicitly opt into it; this is never a security sandbox.
    allow_generated_plot_code: bool | None = None
    dataset_policy: str = "auto"
    offline: bool = False
    checkpoint_dir: Path | str | None = None
    resume: bool = False
    style_guide_path: Path | str | None = None

    def __post_init__(self):
        self.work_dir = Path(self.work_dir)
        self.task_name = "plot" if is_plot_task(self.task_name) else "diagram"
        path = Path(self.config_path) if self.config_path else self.work_dir / "configs" / "model_config.yaml"
        if self.config_path and not path.is_file():
            raise ValueError("The explicit configuration file does not exist")
        defaults = _settings(path)
        self.config_path = path
        from utils.model_registry import get_default_model
        origins = {}
        for field_name, env, aliases in (
            ("main_model_provider", "MAIN_MODEL_PROVIDER", ("main_provider",)),
            ("image_gen_provider", "IMAGE_GEN_PROVIDER", ("image_provider",)),
            ("main_model_name", "MAIN_MODEL_NAME", ("model_name",)),
            ("image_gen_model_name", "IMAGE_GEN_MODEL_NAME", ("image_model_name",)),
            ("evaluation_model_name", "EVALUATION_MODEL_NAME", ()),
        ):
            explicit = getattr(self, field_name)
            if explicit and not isinstance(explicit, str):
                raise ValueError(f"{field_name} must be a string")
            yaml_value = defaults.get(field_name, next((defaults[a] for a in aliases if a in defaults), ""))
            environment = os.environ.get(env, "")
            origins[field_name] = 3 if explicit else 2 if environment else 1 if yaml_value else 0
            value = explicit or environment or yaml_value
            if not value and field_name in ("main_model_name", "image_gen_model_name"):
                operation = "text" if field_name == "main_model_name" else "image"
                provider = self.main_model_provider if operation == "text" else self.image_gen_provider
                value = get_default_model(operation=operation, provider=provider or "gemini")
            value = value or ""
            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string")
            setattr(self, field_name, value.strip())
        for model_field, provider_field in (("main_model_name", "main_model_provider"), ("image_gen_model_name", "image_gen_provider")):
            if origins[model_field] > origins[provider_field]:
                # A higher-priority explicit model is resolved by its exact registry
                # profile; a lower-priority default provider cannot hijack it.
                setattr(self, provider_field, "")
        for field_name, fallback, parser in (
            ("temperature", 1.0, float), ("max_critic_rounds", 3, int), ("max_concurrent", 4, int),
            ("candidate_timeout_seconds", 600.0, float), ("plot_timeout_seconds", 30.0, float),
        ):
            value = getattr(self, field_name)
            if value is None:
                environment = os.environ.get("PAPERBANANA_" + field_name.upper())
                if environment is not None:
                    try:
                        value = parser(environment)
                    except ValueError as exc:
                        raise ValueError(f"Invalid environment value for {field_name}") from exc
                else:
                    value = defaults.get(field_name, fallback)
            setattr(self, field_name, value)
        self.evaluation_model_name = self.evaluation_model_name or self.main_model_name
        if not isinstance(self.model_profiles, dict):
            raise ValueError("model_profiles must be a mapping")
        if not self.model_profiles:
            self.model_profiles = defaults.get("model_profiles", {})
            if not isinstance(self.model_profiles, dict):
                raise ValueError("model_profiles must be a mapping")
        if self.exp_mode == "dev":  # historical CLI alias
            self.exp_mode = "dev_full"
        if self.exp_mode and self.exp_mode not in EXPERIMENT_MODES:
            raise ValueError("Unsupported experiment mode")
        if self.retrieval_setting not in ("auto", "manual", "random", "none"):
            raise ValueError("Unsupported retrieval setting")
        if self.render_mode not in ("legacy", "rfs"):
            raise ValueError("render_mode must be legacy or rfs")
        self.plot_execution_permission_explicit = self.allow_generated_plot_code is not None
        if self.allow_generated_plot_code is None:
            self.allow_generated_plot_code = self.render_mode == "legacy"
        if not isinstance(self.allow_generated_plot_code, bool):
            raise ValueError("allow_generated_plot_code must be a boolean")
        if self.dataset_policy not in ("auto", "never", "download"):
            raise ValueError("dataset_policy must be auto, never, or download")
        if isinstance(self.max_critic_rounds, bool) or not isinstance(self.max_critic_rounds, int) or not 0 <= self.max_critic_rounds <= 20:
            raise ValueError("max_critic_rounds must be an integer between 0 and 20")
        if isinstance(self.max_concurrent, bool) or not isinstance(self.max_concurrent, int) or not 1 <= self.max_concurrent <= MAX_CONCURRENCY:
            raise ValueError(f"max_concurrent must be an integer between 1 and {MAX_CONCURRENCY}")
        self.candidate_timeout_seconds = _positive_number(self.candidate_timeout_seconds, "candidate_timeout_seconds")
        self.plot_timeout_seconds = _positive_number(self.plot_timeout_seconds, "plot_timeout_seconds")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)) or not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.checkpoint_dir is not None:
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if self.resume and self.checkpoint_dir is None:
            raise ValueError("resume requires checkpoint_dir")
        if self.style_guide_path is not None:
            self.style_guide_path = Path(self.style_guide_path)
        self.timestamp = time.strftime("%m%d_%H%M") if self.timestamp is None else self.timestamp
        self.exp_name = f"{self.timestamp}_{self.retrieval_setting}ret_{self.exp_mode}_{self.split_name}"
        self.result_dir = self.work_dir / "results" / f"{self.dataset_name}_{self.task_name}"
        # Keep legacy public attributes, but avoid creating directories in preflight.

    @property
    def rfs_enabled(self) -> bool:
        return self.render_mode == "rfs"
