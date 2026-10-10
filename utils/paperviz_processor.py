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
Processing pipeline of PaperVizAgent
"""

import asyncio
import copy
import inspect
import re
from typing import List, Dict, Any, AsyncGenerator

import numpy as np
from tqdm.asyncio import tqdm

from agents.vanilla_agent import VanillaAgent
from agents.planner_agent import PlannerAgent
from agents.visualizer_agent import VisualizerAgent
from agents.stylist_agent import StylistAgent
from agents.critic_agent import CriticAgent
from agents.retriever_agent import RetrieverAgent
from agents.polish_agent import PolishAgent

from .config import ExpConfig, MAX_CONCURRENCY
from .runtime import CandidateCheckpoints, generation_binding, prepare_candidate, stable_digest, valid_success_result, check_figure_task
from .legacy_ui_results import resolve_final_output
from .eval_toolkits import get_score_for_image_referenced


class PaperVizProcessor:
    """Main class for multimodal document processor"""

    def __init__(
        self,
        exp_config: ExpConfig,
        vanilla_agent: VanillaAgent,
        planner_agent: PlannerAgent,
        visualizer_agent: VisualizerAgent,
        stylist_agent: StylistAgent,
        critic_agent: CriticAgent,
        retriever_agent: RetrieverAgent,
        polish_agent: PolishAgent,
    ):
        self.exp_config = exp_config
        self.vanilla_agent = vanilla_agent
        self.planner_agent = planner_agent
        self.visualizer_agent = visualizer_agent
        self.stylist_agent = stylist_agent
        self.critic_agent = critic_agent
        self.retriever_agent = retriever_agent
        self.polish_agent = polish_agent

    def _check_plot_permission(self, data):
        mode = data.get("render_mode", self.exp_config.render_mode)
        if mode == "rfs" and "figure_spec" in data:
            check_figure_task(data["figure_spec"], self.exp_config.task_name)
        if self.exp_config.task_name == "plot" and mode == "rfs" and (
            not self.exp_config.allow_generated_plot_code or not self.exp_config.plot_execution_permission_explicit
        ):
            raise PermissionError("RFS generated plot execution requires explicit trusted-code permission; this is not a sandbox")

    async def _call_agent(self, stage, data, **kwargs):
        data["_runtime_stage"] = stage
        result = await getattr(self, stage + "_agent").process(data, **kwargs)
        if not isinstance(result, dict):
            raise TypeError("Agent response must be an object")
        for key, value in result.items():
            if key.startswith(("target_diagram_", "target_plot_")) and re.search(r"desc\d+$", key):
                if not isinstance(value, str) or not value.strip() or value.strip() == "Error":
                    raise ValueError("Agent returned no valid design description")
        return result

    async def _run_critic_iterations(self, data: Dict[str, Any], task_name: str, max_rounds: int = 3, source: str = "stylist") -> Dict[str, Any]:
        """
        Run multi-round critic iteration (up to max_rounds).
        Returns the data with critic suggestions and updated eval_image_field.
        
        Args:
            data: Input data dictionary
            task_name: Name of the task (e.g., "diagram", "plot")
            max_rounds: Maximum number of critic iterations
            source: Source of the input for round 0 critique ("stylist" or "planner")
        """
        # Determine initial fallback image key based on source
        if source == "planner":
            current_best_image_key = f"target_{task_name}_desc0_base64_jpg"
        else: # default to stylist
            current_best_image_key = f"target_{task_name}_stylist_desc0_base64_jpg"
            
        if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or not 0 <= max_rounds <= 20:
            raise ValueError("Critic rounds must be an integer between 0 and 20")
        data["review_status"] = "skipped" if max_rounds == 0 else "completed"
        data["eval_image_field"] = current_best_image_key
        for round_idx in range(max_rounds):
            data["current_critic_round"] = round_idx
            try:
                data = await self._call_agent("critic", data, source=source)
                if data.get("critic_review_status") in ("invalid", "failed"):
                    raise ValueError("Critic response was invalid")
            except Exception as exc:
                data["review_status"] = "failed"
                data["error"] = {"type": type(exc).__name__, "stage": "critic", "message": "Critic review failed; last valid image retained"}
                break
            
            critic_suggestions_key = f"target_{task_name}_critic_suggestions{round_idx}"
            critic_suggestions = data.get(critic_suggestions_key, "")
            # Critic output occasionally parses to NaN (float) or None when the
            # model returns empty/malformed JSON on dense content. Coerce to str
            # before calling .strip() to avoid AttributeError mid-run.
            if not isinstance(critic_suggestions, str):
                critic_suggestions = "" if critic_suggestions is None or (
                    isinstance(critic_suggestions, float) and str(critic_suggestions) == "nan"
                ) else str(critic_suggestions)

            if critic_suggestions.strip() == "No changes needed.":
                print(f"[Critic Round {round_idx}] No changes needed. Stopping iteration.")
                break
            
            try:
                data = await self._call_agent("visualizer", data)
            except Exception as exc:
                data["review_status"] = "failed"
                data["error"] = {"type": type(exc).__name__, "stage": "visualizer", "message": "Critic refinement failed; last valid image retained"}
                break

            # Check if visualization validation succeeded
            new_image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
            if self._valid_encoded_image(data.get(new_image_key)):
                current_best_image_key = new_image_key
                data["eval_image_field"] = current_best_image_key
                print(f"[Critic Round {round_idx}] Completed iteration. Visualization SUCCESS.")
            else:
                print(f"[Critic Round {round_idx}] Visualization FAILED (No valid image). Rolling back to previous best: {current_best_image_key}")
                data["review_status"] = "failed"
                data["error"] = {"type": "InvalidImage", "stage": "visualizer", "message": "Critic refinement failed; last valid image retained"}
                break
        
        data["eval_image_field"] = current_best_image_key
        return data

    async def process_single_query(
        self, data: Dict[str, Any], do_eval=True
    ) -> Dict[str, Any]:
        """
        Complete processing pipeline for a single query
        """
        # print(f"[DEBUG] -> Entered process_single_query for candidate {data.get('candidate_id', 'N/A')}")
        self._check_plot_permission(data)
        exp_mode = self.exp_config.exp_mode
        task_name = self.exp_config.task_name.lower()
        retrieval_setting = self.exp_config.retrieval_setting

        # Skip retriever if results were already populated by process_queries_batch
        already_retrieved = "top10_references" in data

        if exp_mode == "vanilla":
            data = await self._call_agent("vanilla", data)
            data["eval_image_field"] = f"vanilla_{task_name}_base64_jpg"

        elif exp_mode == "dev_planner":
            if not already_retrieved:
                data = await self._call_agent("retriever", data, retrieval_setting=retrieval_setting)
            data = await self._call_agent("planner", data)
            data = await self._call_agent("visualizer", data)
            data["eval_image_field"] = f"target_{task_name}_desc0_base64_jpg"

        elif exp_mode == "dev_planner_stylist":
            if not already_retrieved:
                data = await self._call_agent("retriever", data, retrieval_setting=retrieval_setting)
            data = await self._call_agent("planner", data)
            data = await self._call_agent("stylist", data)
            data = await self._call_agent("visualizer", data)
            data["eval_image_field"] = f"target_{task_name}_stylist_desc0_base64_jpg"

        elif exp_mode in ["dev_planner_critic", "demo_planner_critic"]:
            if not already_retrieved:
                data = await self._call_agent("retriever", data, retrieval_setting=retrieval_setting)
            data = await self._call_agent("planner", data)
            data = await self._call_agent("visualizer", data)
            # Use max_critic_rounds from data if available, otherwise default to 3
            max_rounds = data.get("max_critic_rounds", self.exp_config.max_critic_rounds)
            data = await self._run_critic_iterations(data, task_name, max_rounds=max_rounds, source="planner")
            if "demo" in exp_mode: do_eval = False

        elif exp_mode in ["dev_full", "demo_full"]:
            if not already_retrieved:
                data = await self._call_agent("retriever", data, retrieval_setting=retrieval_setting)
            data = await self._call_agent("planner", data)
            data = await self._call_agent("stylist", data)
            data = await self._call_agent("visualizer", data)
            # Use max_critic_rounds from data (if set) or config
            max_rounds = data.get("max_critic_rounds", self.exp_config.max_critic_rounds)
            data = await self._run_critic_iterations(data, task_name, max_rounds=max_rounds, source="stylist")
            if "demo" in exp_mode: do_eval = False
        
        elif exp_mode == "dev_polish":
            data = await self._call_agent("polish", data)
            data["eval_image_field"] = f"polished_{task_name}_base64_jpg"
        
        elif exp_mode == "dev_retriever":
            data = await self._call_agent("retriever", data, retrieval_setting=retrieval_setting)
            do_eval = False

        else:
            raise ValueError(f"Unknown experiment name: {exp_mode}")

        if do_eval:
            data_with_eval = await self.evaluation_function(data, exp_config=self.exp_config)
            return data_with_eval
        else:
            return data

    @staticmethod
    def _valid_encoded_image(encoded):
        from io import BytesIO
        import base64
        from PIL import Image
        if not isinstance(encoded, str) or not encoded:
            return False
        try:
            raw = base64.b64decode(encoded.split(",", 1)[-1], validate=True)
            with Image.open(BytesIO(raw)) as image:
                image.verify()
            return True
        except (ValueError, OSError, Image.DecompressionBombError):
            return False

    def _has_valid_image(self, data):
        selection = resolve_final_output(data, exp_mode=self.exp_config.exp_mode, task_name=self.exp_config.task_name)
        return self._valid_encoded_image(data.get(selection.image_key)) if selection.image_key else False

    async def aclose(self):
        """Explicitly close agent resources; repeated calls are supported."""
        for agent in (self.vanilla_agent, self.planner_agent, self.visualizer_agent,
                      self.stylist_agent, self.critic_agent, self.retriever_agent, self.polish_agent):
            close = getattr(agent, "aclose", None) or getattr(agent, "close", None)
            if close:
                result = close()
                if inspect.isawaitable(result):
                    await result

    async def process_queries_batch(self, data_list: List[Dict[str, Any]],
                                    max_concurrent: int | None = None,
                                    do_eval: bool = True) -> AsyncGenerator[Dict[str, Any], None]:
        """Run a bounded worker pool, isolating every candidate's terminal result.

        Retrieval is shared only for identical source/intent groups. Closing the
        async iterator cancels and joins all workers and retrieval tasks.
        """
        limit = self.exp_config.max_concurrent if max_concurrent is None else max_concurrent
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_CONCURRENCY:
            raise ValueError(f"max_concurrent must be between 1 and {MAX_CONCURRENCY}")
        if not isinstance(data_list, list) or any(not isinstance(data, dict) for data in data_list):
            raise ValueError("Batch input must be a list of objects")
        binding = generation_binding(self.exp_config)
        binding["do_eval"] = do_eval
        candidates = [prepare_candidate(copy.deepcopy(data), index, binding) for index, data in enumerate(data_list)]
        if len({data["candidate_id"] for data in candidates}) != len(candidates):
            raise ValueError("Candidate IDs must be unique within a batch")
        if not candidates:
            return
        checkpoints = CandidateCheckpoints(self.exp_config.checkpoint_dir)
        queue = asyncio.Queue()
        for candidate in candidates:
            queue.put_nowait(candidate)
        completed = asyncio.Queue(maxsize=limit)
        retrieval_tasks = {}
        retrieval_waiters = {}
        all_retrieval_tasks = set()
        retrieval_semaphore = asyncio.Semaphore(limit)
        needs_retrieval = self.exp_config.exp_mode not in ("vanilla", "dev_polish", "dev_retriever")

        async def retrieve(candidate):
            if not needs_retrieval or "top10_references" in candidate:
                return
            group = stable_digest({"content": candidate.get("content"), "intent": candidate.get("visual_intent"),
                                   "caption": candidate.get("caption"), "task": self.exp_config.task_name,
                                   "retrieval": self.exp_config.retrieval_setting})
            if group not in retrieval_tasks:
                async def bounded_retrieval():
                    async with retrieval_semaphore:
                        return await self.retriever_agent.process(copy.deepcopy(candidate),
                            retrieval_setting=self.exp_config.retrieval_setting)
                task = asyncio.create_task(bounded_retrieval(), name="paperbanana-retrieval")
                retrieval_tasks[group] = task
                all_retrieval_tasks.add(task)
            task = retrieval_tasks[group]
            retrieval_waiters[task] = retrieval_waiters.get(task, 0) + 1
            try:
                retrieved = await asyncio.shield(task)
            finally:
                retrieval_waiters[task] -= 1
                if retrieval_waiters[task] == 0 and not task.done():
                    if retrieval_tasks.get(group) is task:
                        del retrieval_tasks[group]
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            for key in ("top10_references", "retrieved_examples", "retrieval_status"):
                if key in retrieved:
                    candidate[key] = copy.deepcopy(retrieved[key])

        async def process(candidate):
            candidate.setdefault("render_mode", self.exp_config.render_mode)
            stage = "retriever"
            identifier, candidate_index = candidate["candidate_id"], candidate["candidate_index"]
            async def pipeline():
                nonlocal stage
                self._check_plot_permission(candidate)
                await retrieve(candidate)
                stage = "pipeline"
                return await self.process_single_query(candidate, do_eval=do_eval)
            try:
                result = await asyncio.wait_for(pipeline(), timeout=self.exp_config.candidate_timeout_seconds)
                if not isinstance(result, dict):
                    raise TypeError("Agent pipeline returned a non-object")
                # IDs cannot be reassigned by generated content or a provider.
                result["candidate_id"], result["candidate_index"] = identifier, candidate_index
                result.pop("_runtime_fingerprint", None)
                result.pop("_runtime_stage", None)
                result["task_name"] = self.exp_config.task_name
                result["exp_mode"] = self.exp_config.exp_mode
                if result.get("review_status") == "failed":
                    result["status"] = "failed"
                elif self.exp_config.exp_mode == "dev_retriever" or self._has_valid_image(result):
                    result["status"] = "succeeded"
                    if result.get("render_mode") == "rfs" and not valid_success_result(result):
                        result["status"] = "failed"
                        result["error"] = {"type": "MissingArtifact", "stage": "visualizer", "message": "The selected RFS generation is missing a complete image, prompt, design, or scientific contract"}
                else:
                    result["status"] = "failed"
                    result["error"] = {"type": "MissingOutput", "stage": "visualizer", "message": "No valid final image was produced"}
                return result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                result = {key: value for key, value in candidate.items() if not key.startswith("_runtime_")}
                result.update(status="failed", candidate_id=identifier, candidate_index=candidate_index, task_name=self.exp_config.task_name,
                              error={"type": type(exc).__name__, "stage": candidate.get("_runtime_stage", stage),
                                     "message": "Candidate timed out" if isinstance(exc, asyncio.TimeoutError) else "Candidate processing failed"})
                return result

        async def worker():
            while True:
                try:
                    candidate = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                fingerprint = candidate["_runtime_fingerprint"]
                restored = checkpoints.load(candidate) if self.exp_config.resume else None
                result = restored if restored is not None else await process(candidate)
                checkpoint_identity = {"candidate_id": result["candidate_id"], "_runtime_fingerprint": fingerprint}
                if restored is None:
                    try:
                        checkpoints.save(checkpoint_identity, result)
                    except (OSError, ValueError, TypeError) as exc:
                        result["checkpoint_status"] = {"status": "failed", "type": type(exc).__name__}
                await completed.put(result)
                queue.task_done()

        workers = [asyncio.create_task(worker(), name=f"paperbanana-worker-{index}") for index in range(min(limit, len(candidates)))]
        try:
            for _ in candidates:
                yield await completed.get()
        finally:
            for task in workers + list(all_retrieval_tasks):
                if not task.done():
                    task.cancel()
            await asyncio.gather(*workers, *all_retrieval_tasks, return_exceptions=True)

    async def evaluation_function(
        self, data: Dict[str, Any], exp_config: ExpConfig
    ) -> Dict[str, Any]:
        """
        Evaluation function - uses referenced setting (GT shown first)
        """
        data = await get_score_for_image_referenced(
            data, task_name=exp_config.task_name, work_dir=exp_config.work_dir,
            model_name=exp_config.evaluation_model_name,
            provider=exp_config.main_model_provider or None, overrides=exp_config.model_profiles
        )
        return data

