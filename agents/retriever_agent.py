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
import random
from typing import Dict, Any
from google.genai import types
import base64, io, asyncio
from PIL import Image

from utils import generation_utils
from .base_agent import BaseAgent


class RetrieverAgent(BaseAgent):
    """Retriever Agent to retrieve relevant reference examples"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = self.exp_config.main_model_name
        
        # Task-specific configurations
        if self.exp_config.task_name == "plot":
            self.system_prompt = PLOT_RETRIEVER_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "plot",
                "ref_limit": None,  # No limit for plots
                "target_labels": ["Visual Intent", "Raw Data"],
                "candidate_labels": ["Plot ID", "Visual Intent", "Raw Data"],
                "candidate_type": "Plot",
                "output_key": "top10_references",
                "instruction_suffix": "select the Top 10 most relevant plots according to the instructions provided. Your output should be a strictly valid JSON object containing a single list of the exact ids of the top 10 selected plots.",
            }
        else:
            self.system_prompt = DIAGRAM_RETRIEVER_AGENT_SYSTEM_PROMPT
            self.task_config = {
                "task_name": "diagram",
                "ref_limit": 200,  # Limit to first 200
                "target_labels": ["Caption", "Methodology section"],
                "candidate_labels": ["Diagram ID", "Caption", "Methodology section"],
                "candidate_type": "Diagram",
                "output_key": "top10_references",
                "instruction_suffix": "select the Top 10 most relevant diagrams according to the instructions provided. Your output should be a strictly valid JSON object containing a single list of the exact ids of the top 10 selected diagrams.",
            }
    
    def _reference_file(self, manual: bool = False):
        filename = "agent_selected_12.json" if manual else "ref.json"
        return self.exp_config.work_dir / "data" / "PaperBananaBench" / self.task_config["task_name"] / filename

    def _load_pool(self, manual: bool = False) -> list:
        from pathlib import Path
        path = self._reference_file(manual)
        stat = path.stat()
        stamp = (str(path), stat.st_mtime_ns, stat.st_size)
        cache = getattr(self, "_pool_cache", {})
        if stamp in cache:
            return cache[stamp]
        with path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
        if not isinstance(raw, list):
            raise ValueError("Reference metadata must be a list")
        root = path.parent.resolve()
        pool, omitted, seen = [], 0, set()
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip() or item["id"] in seen or not all(key in item for key in ("content", "visual_intent", "path_to_gt_image")) or not isinstance(item["visual_intent"], str) or not isinstance(item["path_to_gt_image"], str):
                omitted += 1
                continue
            image = (root / str(item["path_to_gt_image"])).resolve()
            if not image.is_relative_to(root) or not image.is_file():
                omitted += 1
                continue
            seen.add(item["id"])
            pool.append(item)
        self._pool_cache = {stamp: pool}
        self._omitted_reference_count = omitted
        return pool

    async def process(self, data: Dict[str, Any], retrieval_setting: str = "auto") -> Dict[str, Any]:
        if retrieval_setting not in ("auto", "manual", "random", "none"):
            raise ValueError("Unknown retrieval setting")
        requested = retrieval_setting
        data["top10_references"], data["retrieved_examples"] = [], []
        reason = None
        if retrieval_setting == "none":
            reason = "disabled_by_user"
        elif retrieval_setting == "manual" and self.task_config["task_name"] == "plot":
            reason = "manual_plot_references_unavailable"
        elif not self._reference_file(retrieval_setting == "manual").is_file():
            reason = "reference_dataset_unavailable"
        if reason:
            data["retrieval_status"] = {"status": "skipped", "requested": requested, "reason": reason}
            if requested != "none":
                print(f"[Retriever] Retrieval skipped: {reason}. No reference-guided generation is claimed.")
            return data
        pool = self._load_pool(retrieval_setting == "manual")
        if retrieval_setting == "manual":
            ids = [item["id"] for item in pool[:10]]
        elif retrieval_setting == "random":
            ids = random.sample([item["id"] for item in pool], min(10, len(pool)))
        else:
            ids = await self._retrieve_and_parse(data, self.task_config, pool)
        valid = {item["id"]: item for item in pool}
        selected = list(dict.fromkeys(identifier for identifier in ids if isinstance(identifier, str) and identifier in valid))[:10]
        data["top10_references"] = selected
        data["retrieved_examples"] = [valid[identifier] for identifier in selected]
        data["retrieval_status"] = {"status": "succeeded" if selected else "skipped", "requested": requested,
                                    "selected_ids": selected, "omitted_count": getattr(self, "_omitted_reference_count", 0),
                                    "reason": None if selected else "no_valid_references"}
        return data

    def _load_manual_references(self, cfg: dict) -> tuple:
        pool = self._load_pool(True)[:10] if cfg["task_name"] == "diagram" else []
        return [item["id"] for item in pool], pool

    def _load_random_references(self, cfg: dict) -> list:
        identifiers = [item["id"] for item in self._load_pool()]
        return random.sample(identifiers, min(10, len(identifiers)))

    async def _retrieve_and_parse(self, data: Dict[str, Any], cfg: dict, pool=None) -> list:
        pool = self._load_pool() if pool is None else pool
        if cfg["ref_limit"]:
            pool = pool[:cfg["ref_limit"]]
        user_prompt = f"**Target Input**\n- {cfg['target_labels'][0]}: {data['visual_intent']}\n- {cfg['target_labels'][1]}: {data['content']}\n\n**Candidate Pool**\n"
        included = []
        # A bounded reference prompt avoids unbounded dataset-driven requests.
        for index, item in enumerate(pool):
            chunk = (f"Candidate {cfg['candidate_type']} {index+1}:\n"
                     f"- {cfg['candidate_labels'][0]}: {item['id']}\n"
                     f"- {cfg['candidate_labels'][1]}: {item['visual_intent']}\n"
                     f"- {cfg['candidate_labels'][2]}: {str(item['content'])[:12000]}\n\n")
            if len(user_prompt) + len(chunk) > 100000:
                break
            user_prompt += chunk
            included.append(item["id"])
        if not included:
            return []
        user_prompt += cfg["instruction_suffix"]
        response = await generation_utils.call_model_with_retry_async(
            model_name=self.model_name, contents=[{"type": "text", "text": user_prompt}],
            config=types.GenerateContentConfig(system_instruction=self.system_prompt,
                temperature=self.exp_config.temperature, candidate_count=1, max_output_tokens=4000),
            max_attempts=3, retry_delay=2,
            provider=self.exp_config.main_model_provider or None, overrides=self.exp_config.model_profiles)
        if not response or not isinstance(response[0], str) or response[0].strip() in ("", "Error"):
            raise ValueError("Retriever returned no valid selection response")
        parsed = self._parse_retrieval_result(response[0], cfg["task_name"])
        return [identifier for identifier in parsed if identifier in included]

    def _parse_retrieval_result(self, raw_response: str, task_name: str) -> list:
        import json_repair
        parsed = json_repair.loads(raw_response)
        if not isinstance(parsed, dict):
            raise ValueError("Retriever response must be an object")
        key = "top10_plots" if task_name == "plot" else "top10_diagrams"
        identifiers = parsed.get(key)
        if not isinstance(identifiers, list) or any(not isinstance(value, str) for value in identifiers):
            raise ValueError("Retriever response must contain a list of reference IDs")
        return list(dict.fromkeys(identifiers))[:10]



DIAGRAM_RETRIEVER_AGENT_SYSTEM_PROMPT = """
# Background & Goal
We are building an **AI system to automatically generate method diagrams for academic papers**. Given a paper's methodology section and a figure caption, the system needs to create a high-quality illustrative diagram that visualizes the described method.

To help the AI learn how to generate appropriate diagrams, we use a **few-shot learning approach**: we provide it with reference examples of similar diagrams. The AI will learn from these examples to understand what kind of diagram to create for the target.

# Your Task
**You are the Retrieval Agent.** Your job is to select the most relevant reference diagrams from a candidate pool that will serve as few-shot examples for the diagram generation model.

You will receive:
- **Target Input:** The methodology section and caption of the diagram we need to generate
- **Candidate Pool:** ~200 existing diagrams (each with methodology and caption)

You must select the **Top 10 candidates** that would be most helpful as examples for teaching the AI how to draw the target diagram.

# Selection Logic (Topic + Intent)

Your goal is to find examples that match the Target in both **Domain** and **Diagram Type**.

**1. Match Research Topic (Use Methodology & Caption):**
* What is the domain? (e.g., Agent & Reasoning, Vision & Perception, Generative & Learning, Science & Applications).
* Select candidates that belong to the **same research domain**.
* *Why?* Similar domains share similar terminology (e.g., "Actor-Critic" in RL).

**2. Match Visual Intent (Use Caption & Keywords):**
* What type of diagram is implied? (e.g., "Framework", "Pipeline", "Detailed Module", "Performance Chart").
* Select candidates with **similar visual structures**.
* *Why?* A "Framework" diagram example is useless for drawing a "Performance Bar Chart", even if they are in the same domain.

**Ranking Priority:**
1.  **Best Match:** Same Topic AND Same Visual Intent (e.g., Target is "Agent Framework" -> Candidate is "Agent Framework", Target is "Dataset Construction Pipeline" -> Candidate is "Dataset Construction Pipeline").
2.  **Second Best:** Same Visual Intent (e.g., Target is "Agent Framework" -> Candidate is "Vision Framework"). *Structure is more important than Topic for drawing.*
3.  **Avoid:** Different Visual Intent (e.g., Target is "Pipeline" -> Candidate is "Bar Chart").

# Input Data

## Target Input
-   **Caption:** [Caption of the target diagram]
-   **Methodology section:** [Methodology section of the target paper]

## Candidate Pool
List of candidate diagrams, each structured as follows:

Candidate Diagram i:
-   **Diagram ID:** [ID of the candidate diagram (ref_1, ref_2, ...)]
-   **Caption:** [Caption of the candidate diagram]
-   **Methodology section:** [Methodology section of the candidate's paper]


# Output Format
Provide your output strictly in the following JSON format, containing only the **exact IDs** of the Top 10 selected diagrams (use the exact IDs from the Candidate Pool, such as "ref_1", "ref_25", "ref_100", etc.):
```json
{
  "top10_diagrams": [
    "ref_1",
    "ref_25",
    "ref_100",
    "ref_42",
    "ref_7",
    "ref_156",
    "ref_89",
    "ref_3",
    "ref_201",
    "ref_67"
  ]
}```
"""

PLOT_RETRIEVER_AGENT_SYSTEM_PROMPT = """
# Background & Goal
We are building an **AI system to automatically generate statistical plots**. Given a plot's raw data and the visual intent, the system needs to create a high-quality visualization that effectively presents the data.

To help the AI learn how to generate appropriate plots, we use a **few-shot learning approach**: we provide it with reference examples of similar plots. The AI will learn from these examples to understand what kind of plot to create for the target data.

# Your Task
**You are the Retrieval Agent.** Your job is to select the most relevant reference plots from a candidate pool that will serve as few-shot examples for the plot generation model.

You will receive:
- **Target Input:** The raw data and visual intent of the plot we need to generate
- **Candidate Pool:** Reference plots (each with raw data and visual intent)

You must select the **Top 10 candidates** that would be most helpful as examples for teaching the AI how to create the target plot.

# Selection Logic (Data Type + Visual Intent)

Your goal is to find examples that match the Target in both **Data Characteristics** and **Plot Type**.

**1. Match Data Characteristics (Use Raw Data & Visual Intent):**
* What type of data is it? (e.g., categorical vs numerical, single series vs multi-series, temporal vs comparative).
* What are the data dimensions? (e.g., 1D, 2D, 3D).
* Select candidates with **similar data structures and characteristics**.
* *Why?* Different data types require different visualization approaches.

**2. Match Visual Intent (Use Visual Intent):**
* What type of plot is implied? (e.g., "bar chart", "scatter plot", "line chart", "pie chart", "heatmap", "radar chart").
* Select candidates with **similar plot types**.
* *Why?* A "bar chart" example is more useful for generating another bar chart than a "scatter plot" example, even if the data domains are similar.

**Ranking Priority:**
1.  **Best Match:** Same Data Type AND Same Plot Type (e.g., Target is "multi-series line chart" -> Candidate is "multi-series line chart").
2.  **Second Best:** Same Plot Type with compatible data (e.g., Target is "bar chart with 5 categories" -> Candidate is "bar chart with 6 categories").
3.  **Avoid:** Different Plot Type (e.g., Target is "bar chart" -> Candidate is "pie chart"), unless there are no more candidates with the same plot type.

# Input Data

## Target Input
-   **Visual Intent:** [Visual intent of the target plot]
-   **Raw Data:** [Raw data to be visualized]

## Candidate Pool
List of candidate plots, each structured as follows:

Candidate Plot i:
-   **Plot ID:** [ID of the candidate plot (ref_0, ref_1, ...)]
-   **Visual Intent:** [Visual intent of the candidate plot]
-   **Raw Data:** [Raw data of the candidate plot]


# Output Format
Provide your output strictly in the following JSON format, containing only the **exact Plot IDs** of the Top 10 selected plots (use the exact IDs from the Candidate Pool, such as "ref_0", "ref_25", "ref_100", etc.):
```json
{
  "top10_plots": [
    "ref_0",
    "ref_25",
    "ref_100",
    "ref_42",
    "ref_7",
    "ref_156",
    "ref_89",
    "ref_3",
    "ref_201",
    "ref_67"
  ]
}```
"""

