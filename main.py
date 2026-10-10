# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Benchmark entry point with bounded concurrency and matching-run resume."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from utils.config import EXPERIMENT_MODES, ExpConfig
from utils.runtime import atomic_write_json, strict_scientific_json, normalize_scientific_content, check_figure_task


def build_parser():
    parser = argparse.ArgumentParser(description="PaperBanana benchmark processor")
    parser.add_argument("--dataset_name", "--dataset-name", default="PaperBananaBench")
    parser.add_argument("--task_name", "--task-name", choices=["diagram", "plot"], default="diagram")
    parser.add_argument("--split_name", "--split-name", default="test")
    parser.add_argument("--exp_mode", "--exp-mode", choices=list(EXPERIMENT_MODES) + ["dev"], default="dev_full")
    parser.add_argument("--retrieval_setting", "--retrieval-setting", choices=["auto", "manual", "random", "none"], default="auto")
    parser.add_argument("--planner-metaphor", action="store_true")
    parser.add_argument("--max_critic_rounds", "--max-critic-rounds", type=int, default=None)
    parser.add_argument("--main_model_name", "--main-model-name", default="")
    parser.add_argument("--image_gen_model_name", "--image-gen-model-name", default="")
    parser.add_argument("--main-model-provider", default="")
    parser.add_argument("--image-gen-provider", default="")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument("--candidate-timeout-seconds", type=float, default=None)
    parser.add_argument("--plot-timeout-seconds", type=float, default=None)
    parser.add_argument("--rfs-output-dir", default=None)
    parser.add_argument("--render-mode", choices=["legacy", "rfs"], default="legacy")
    parser.add_argument("--allow-generated-plot-code", action="store_true")
    parser.add_argument("--dataset-policy", choices=["auto", "never", "download"], default="auto")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--no-eval", action="store_true")
    return parser


async def main(argv=None):
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parent
    config = ExpConfig(dataset_name=args.dataset_name, task_name=args.task_name, split_name=args.split_name,
        exp_mode=args.exp_mode, retrieval_setting=args.retrieval_setting, planner_metaphor=args.planner_metaphor,
        max_critic_rounds=args.max_critic_rounds, main_model_name=args.main_model_name,
        image_gen_model_name=args.image_gen_model_name, main_model_provider=args.main_model_provider,
        image_gen_provider=args.image_gen_provider, config_path=args.config, work_dir=root,
        max_concurrent=args.max_concurrent, candidate_timeout_seconds=args.candidate_timeout_seconds,
        plot_timeout_seconds=args.plot_timeout_seconds, render_mode=args.render_mode,
        allow_generated_plot_code=args.allow_generated_plot_code or args.render_mode == "legacy",
        dataset_policy=args.dataset_policy, offline=args.offline, checkpoint_dir=args.checkpoint_dir, resume=args.resume)
    source = Path(args.input_file) if args.input_file else root / "data" / config.dataset_name / config.task_name / f"{config.split_name}.json"
    input_text = source.read_text(encoding="utf-8")
    data_list = strict_scientific_json(input_text) if config.render_mode == "rfs" else json.loads(input_text)
    if not isinstance(data_list, list) or any(not isinstance(data, dict) or "content" not in data or "visual_intent" not in data for data in data_list):
        raise ValueError("Benchmark input must be a list containing content and visual_intent for every record")
    for data in data_list:
        content, intent = data["content"], data["visual_intent"]
        if not isinstance(intent, str) or not intent.strip() or content is None or (isinstance(content, (str, list, dict)) and not content):
            raise ValueError("Every input record requires nonempty scientific content and visual_intent")
        rounds = data.get("max_critic_rounds", config.max_critic_rounds)
        if isinstance(rounds, bool) or not isinstance(rounds, int) or not 0 <= rounds <= 20:
            raise ValueError("Input critic rounds must be an integer between 0 and 20")
        if config.render_mode == "rfs" and "figure_spec" in data:
            from rfs import validate_figure_spec
            data["figure_spec"] = validate_figure_spec(data["figure_spec"], input_content=content)
            check_figure_task(data["figure_spec"], config.task_name)
        if config.render_mode == "rfs" and isinstance(content, str):
            data["source_content_text"] = content
        data["content"] = normalize_scientific_content(content, config.task_name, config.render_mode)
    if config.render_mode == "rfs" and config.task_name == "plot" and not args.allow_generated_plot_code:
        raise ValueError("RFS generated plot code requires --allow-generated-plot-code; this is not a sandbox")
    from utils.generation_utils import validate_generation_configuration
    validate_generation_configuration(config, needs_text=True, needs_image=config.task_name == "diagram")
    if args.offline and args.dataset_policy == "download":
        raise ValueError("--offline conflicts with --dataset-policy download")
    if config.retrieval_setting != "none" and config.exp_mode not in ("vanilla", "dev_polish"):
        from utils.dataset import ensure_dataset
        available = ensure_dataset(config.task_name, work_dir=root, retrieval_setting=config.retrieval_setting,
                                   policy=config.dataset_policy, offline=config.offline)
        if not available:
            if config.offline or config.dataset_policy != "auto":
                raise RuntimeError("Required local references are missing; choose --retrieval-setting none to explicitly skip retrieval")
            print("WARNING: Reference data unavailable; retrieval will be recorded as skipped", file=sys.stderr)
    if config.task_name == "plot":
        print("WARNING: Trusted generated Python executes with filesystem/network access. Timeout is not a sandbox.", file=sys.stderr)
    from agents.planner_agent import PlannerAgent
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    from utils.paperviz_processor import PaperVizProcessor
    processor = PaperVizProcessor(exp_config=config,
        vanilla_agent=VanillaAgent(exp_config=config), planner_agent=PlannerAgent(exp_config=config),
        visualizer_agent=VisualizerAgent(exp_config=config), stylist_agent=StylistAgent(exp_config=config),
        critic_agent=CriticAgent(exp_config=config), retriever_agent=RetrieverAgent(exp_config=config),
        polish_agent=PolishAgent(exp_config=config))
    output = Path(args.output_file) if args.output_file else config.result_dir / f"{config.exp_name}.json"
    results = []
    try:
        async for result in processor.process_queries_batch(data_list, max_concurrent=config.max_concurrent, do_eval=not args.no_eval):
            if config.render_mode == "rfs":
                from utils.runtime import write_or_reuse_rfs_bundle, candidate_artifact_directory
                bundle_root = Path(args.rfs_output_dir) if args.rfs_output_dir else output.parent / f"{output.stem}_rfs"
                try:
                    bundle_dir = candidate_artifact_directory(bundle_root, result["candidate_id"])
                    write_or_reuse_rfs_bundle(result, bundle_dir)
                    result["rfs_bundle_dir"] = str(bundle_dir.resolve())
                except (OSError, ValueError) as exc:
                    result.update(status="failed", error={"type": type(exc).__name__, "stage": "export", "message": "RFS bundle export failed"})
            results.append(result)
            atomic_write_json(output, sorted(results, key=lambda item: item["candidate_index"]))
        if not results:
            atomic_write_json(output, [])
    finally:
        await processor.aclose()
    print(f"Saved {len(results)} candidate records to {output}")
    return 2 if any(result.get("status") != "succeeded" for result in results) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
