# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""PaperBanana CLI: legacy images or a versioned RFS exchange bundle."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import sys
import tempfile
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def ensure_model_config():
    """Explicit legacy helper; normal preflight never writes configuration."""
    configs_dir = PROJECT_ROOT / "configs"
    path = configs_dir / "model_config.yaml"
    template = configs_dir / "model_config.template.yaml"
    if not path.exists() and template.exists():
        shutil.copy2(template, path)


def ensure_dataset(task_name: str, **kwargs):
    from utils.dataset import ensure_dataset as prepare
    return prepare(task_name, work_dir=PROJECT_ROOT, **kwargs)


def extract_final_image_b64(result: dict, exp_mode: str, task_name: str | None = None) -> str | None:
    from utils.legacy_ui_results import infer_task_name, resolve_final_output
    task = task_name or infer_task_name(result)
    selection = resolve_final_output(result, exp_mode=exp_mode, task_name=task)
    return result.get(selection.image_key) if selection.image_key else None


def _save_png(encoded: str, path: Path) -> None:
    from PIL import Image
    payload = base64.b64decode(encoded.split(",", 1)[-1], validate=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream, Image.open(BytesIO(payload)) as image:
            image.load()
            image.save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_parser():
    parser = argparse.ArgumentParser(description="PaperBanana: generate scientific figures and RFS exchange bundles")
    parser.add_argument("--content", default="", help="Method text, or real plot data")
    parser.add_argument("--content-file", default="", help="UTF-8 source file")
    parser.add_argument("--caption", required=True, help="Figure caption / visual intent")
    parser.add_argument("--task", choices=["diagram", "plot"], default="diagram")
    parser.add_argument("--output", default="output.png", help="PNG output path")
    parser.add_argument("--aspect-ratio", choices=["21:9", "16:9", "3:2", "1:1", "9:16"], default="21:9")
    parser.add_argument("--max-critic-rounds", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--max-concurrent", type=int, default=None)
    parser.add_argument("--candidate-timeout-seconds", type=float, default=None)
    parser.add_argument("--plot-timeout-seconds", type=float, default=None)
    parser.add_argument("--retrieval-setting", choices=["auto", "manual", "random", "none"], default="auto")
    parser.add_argument("--dataset-policy", choices=["auto", "never", "download"], default="auto")
    parser.add_argument("--offline", action="store_true", help="Disable dataset downloads. Live generation still requires an API.")
    parser.add_argument("--planner-metaphor", action="store_true")
    parser.add_argument("--main-model-name", default="")
    parser.add_argument("--image-gen-model-name", default="")
    parser.add_argument("--main-model-provider", default="")
    parser.add_argument("--image-gen-provider", default="")
    parser.add_argument("--config", default=None, help="Explicit model configuration YAML")
    parser.add_argument("--exp-mode", choices=["demo_full", "demo_planner_critic", "vanilla", "dev_polish"], default="demo_full")
    parser.add_argument("--render-mode", choices=["legacy", "rfs"], default="legacy")
    parser.add_argument("--rfs-file", default="", help="Scientific FigureSpec JSON; requires the original source input")
    parser.add_argument("--rfs-output-dir", default="", help="New per-candidate exchange directories")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse only successful checkpoints matching source, models, code, and reference data")
    parser.add_argument("--allow-generated-plot-code", action="store_true", help="Trust generated Python code. It has filesystem/network access; this is not a sandbox.")
    return parser


def build_processor(config):
    from agents.planner_agent import PlannerAgent
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    from utils.paperviz_processor import PaperVizProcessor

    return PaperVizProcessor(exp_config=config,
        vanilla_agent=VanillaAgent(exp_config=config), planner_agent=PlannerAgent(exp_config=config),
        visualizer_agent=VisualizerAgent(exp_config=config), stylist_agent=StylistAgent(exp_config=config),
        critic_agent=CriticAgent(exp_config=config), retriever_agent=RetrieverAgent(exp_config=config),
        polish_agent=PolishAgent(exp_config=config))


def validate_generation_configuration(config):
    from utils.generation_utils import validate_generation_configuration as preflight
    preflight(config, needs_text=True, needs_image=config.task_name == "diagram")


async def run(args):
    from utils.config import ExpConfig
    from utils.runtime import atomic_write_json, normalize_scientific_content, check_figure_task

    # Validate every source and runtime parameter before imports, downloads, or API work.
    if args.content and args.content_file:
        raise ValueError("Choose --content or --content-file")
    content = Path(args.content_file).read_text(encoding="utf-8") if args.content_file else args.content
    if not content or not content.strip():
        raise ValueError("--content or --content-file is required")
    if not args.caption.strip():
        raise ValueError("--caption cannot be empty")
    if not 1 <= args.num_candidates <= 1000:
        raise ValueError("--num-candidates must be between 1 and 1000")
    if args.rfs_file and args.render_mode != "rfs":
        raise ValueError("--rfs-file requires --render-mode rfs")
    if args.offline and args.dataset_policy == "download":
        raise ValueError("--offline conflicts with --dataset-policy download")
    source_content_text = content
    content = normalize_scientific_content(content, args.task, args.render_mode)
    config = ExpConfig(dataset_name="Demo", task_name=args.task, split_name="demo", exp_mode=args.exp_mode,
        retrieval_setting=args.retrieval_setting, planner_metaphor=args.planner_metaphor,
        max_critic_rounds=args.max_critic_rounds, main_model_name=args.main_model_name,
        image_gen_model_name=args.image_gen_model_name, main_model_provider=args.main_model_provider,
        image_gen_provider=args.image_gen_provider, config_path=args.config, work_dir=PROJECT_ROOT,
        render_mode=args.render_mode, max_concurrent=args.max_concurrent,
        candidate_timeout_seconds=args.candidate_timeout_seconds, plot_timeout_seconds=args.plot_timeout_seconds,
        allow_generated_plot_code=args.allow_generated_plot_code or args.render_mode == "legacy",
        dataset_policy=args.dataset_policy, offline=args.offline,
        checkpoint_dir=args.checkpoint_dir, resume=args.resume)
    if config.task_name == "plot" and config.render_mode == "rfs" and not args.allow_generated_plot_code:
        raise ValueError("The current plot backend executes generated Python. Use --allow-generated-plot-code only for a trusted single-user installation; this is not a sandbox.")
    spec = None
    if args.rfs_file:
        from rfs import validate_figure_spec
        from utils.runtime import strict_scientific_json
        spec = validate_figure_spec(strict_scientific_json(Path(args.rfs_file).read_text(encoding="utf-8")), input_content=content)
        check_figure_task(spec, config.task_name)
    validate_generation_configuration(config)
    needs_references = config.exp_mode not in ("vanilla", "dev_polish") and config.retrieval_setting != "none"
    if needs_references:
        available = ensure_dataset(config.task_name, retrieval_setting=config.retrieval_setting,
                                   policy=config.dataset_policy, offline=config.offline)
        if not available:
            if config.offline or config.dataset_policy in ("never", "download"):
                raise RuntimeError("Required reference data is unavailable. Supply local data or explicitly use --retrieval-setting none.")
            print("WARNING: Reference dataset unavailable. Retrieval will be marked skipped; this run is not reference-guided.", file=sys.stderr)
    if config.task_name == "plot":
        print("WARNING: Executing trusted generated Python with a timeout. This is not a security sandbox and code can access files or networks.", file=sys.stderr)

    processor = build_processor(config)
    data_list = [{"filename": f"skill_candidate_{index}", "caption": args.caption,
                  "content": content, "visual_intent": args.caption,
                  **({"source_content_text": source_content_text} if args.render_mode == "rfs" else {}),
                  "additional_info": {"rounded_ratio": args.aspect_ratio},
                  "max_critic_rounds": config.max_critic_rounds, "render_mode": config.render_mode,
                  **({"figure_spec": spec} if spec is not None else {})} for index in range(args.num_candidates)]
    output = Path(args.output).resolve()
    results, saved_paths = [], []
    try:
        async for result in processor.process_queries_batch(data_list, max_concurrent=config.max_concurrent, do_eval=False):
            results.append(result)
            index = result.get("candidate_index", len(results) - 1)
            encoded = extract_final_image_b64(result, args.exp_mode, config.task_name)
            if args.render_mode == "rfs":
                from utils.runtime import write_or_reuse_rfs_bundle, candidate_artifact_directory
                bundle_root = Path(args.rfs_output_dir) if args.rfs_output_dir else output.parent / f"{output.stem}_rfs"
                bundle_dir = candidate_artifact_directory(bundle_root, result["candidate_id"])
                try:
                    write_or_reuse_rfs_bundle(result, bundle_dir)
                    result["rfs_bundle_dir"] = str(bundle_dir.resolve())
                except (ValueError, OSError) as exc:
                    result.update(status="failed", error={"type": type(exc).__name__, "stage": "export", "message": "RFS bundle export failed"})
                # Preserve source bytes for the selected generation, avoiding JPEG compatibility copies.
                key = result.get("eval_image_field", "").removesuffix("_base64_jpg")
                artifact = result.get("rfs_artifacts", {}).get(key, {})
                encoded = artifact.get("image_base64") or encoded
            if encoded:
                path = output if args.num_candidates == 1 else output.parent / f"{output.stem}_{index}{output.suffix or '.png'}"
                try:
                    _save_png(encoded, path)
                    saved_paths.append(str(path))
                    result["output_path"] = str(path)
                except (ValueError, OSError) as exc:
                    result.update(status="failed", error={"type": type(exc).__name__, "stage": "save", "message": "The final image could not be decoded or saved"})
            else:
                result.update(status="failed", error=result.get("error") or {"type": "MissingOutput", "stage": "save", "message": "No final image produced"})
            atomic_write_json(output.with_suffix(output.suffix + ".results.json"), sorted(results, key=lambda item: item.get("candidate_index", 0)))
    finally:
        await processor.aclose()
    for path in saved_paths:
        print(path)
    failed = sum(result.get("status") != "succeeded" for result in results)
    if failed or not results:
        print(f"ERROR: {failed or args.num_candidates} candidate(s) failed. Successful files, if any, were retained; see the results JSON.", file=sys.stderr)
        return 2
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(run(args))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
