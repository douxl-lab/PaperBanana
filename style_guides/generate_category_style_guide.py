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

import asyncio
import os
import json
import pathlib
import base64
import sys
from tqdm.asyncio import tqdm

# ================= 1. Configuration =================
MODE = "diagram"

# Define work directory and data directory for path resolution
WORK_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = WORK_DIR / f"data/PaperBananaBench/{MODE}"

INPUT_JSON_PATH = WORK_DIR / f"data/PaperBananaBench/{MODE}/ref.json"
OUTPUT_REPORT_PATH = WORK_DIR / f"style_guides/neurips2025_{MODE}_style_guide.md"
BATCH_OUTPUT_DIR = WORK_DIR / "tmp/style_analysis_all"

# Sampling settings
NUM_SAMPLES = 50  # None for all, or a number to limit

MODEL_NAME = None  # Use configured main model / registry default unless explicitly selected.

# Batch processing settings
BATCH_SIZE = 20
CONCURRENCY_LIMIT = 5

# No API client or network request is created when this module is imported.
sys.path.insert(0, str(WORK_DIR))

# ================= 3. Improved Prompt Design =================


DIAGRAM_BATCH_ANALYSIS_PROMPT = """
You are a Lead Information Designer analyzing the visual style of top-tier AI conference papers (NeurIPS 2025).
I have attached a batch of methodology diagrams from the NeurIPS 2025 conference.

**Your Task:**
Summarize a visual design guideline that ignores the specific scientific algorithms. Focus ONLY on the **Aesthetic and Graphic Design** choices.

**Critical:** Do NOT converge each element to a single fixed design choice. Instead, identify what common design choices exist for each element and which ones are more popular or preferred.

Please focus on these specific dimensions:
1.  **Color Palette:** Observe color schemes, saturation levels, etc. Notice aesthetically pleasing combinations and preserve multiple options.
2.  **Shapes & Containers:** Observe shape choices (e.g., rounded vs. sharp rectangles), containers, borders (thickness, color), background fills, shadows, etc.
3.  **Lines & Arrows:** Observe line thickness, colors, arrow styles, dashed line usage.
4.  **Layout & Composition:** Observe layouts, element arrangement patterns, information density, whitespace usage.
5.  **Typography & Icons:** Observe font weights, sizes, colors, usage patterns, and icon usage.

Please note that papers of different domains may have different aesthetic preferences. For example, agent papers will use detailed, cartoon-like illustrative styles more often, while theorectical papers will use more minimalistic styles. When you are summarizing the style, please consider the domain of the paper. You can use "For [domain], common options include: [list]" format to describe the style.

Return a concise bullet-point summary of the visual style diversity observed in this batch.
"""

PLOT_BATCH_ANALYSIS_PROMPT = """
You are a Lead Information Designer analyzing the visual style of top-tier AI conference papers (NeurIPS 2025).
I have attached a batch of statistical plots from the NeurIPS 2025 conference.

**Your Task:**
Summarize a visual design guideline for statistical plots. Focus ONLY on the **Aesthetic and Graphic Design** choices (not the data itself).

**Critical:** Do NOT converge each element to a single fixed design choice. Instead, identify what common design choices exist for each element and which ones are more popular or preferred.

Please focus on these specific dimensions:
1.  **Color Palette:** Observe color schemes for categorical data, sequential gradients for heatmaps, and diverging scales. Identify aesthetically pleasing combinations.
2.  **Axes & Grids:** Observe the styling of x/y axes, tick marks, and grid lines (e.g., light gray, dashed, none). Note the line weights and colors.
3.  **Data Representation (by Type):**
    *   **Bar Chart:** Bar width, spacing, borders, and error bar styles.
    *   **Line Chart:** Line thickness, transparency, marker styles (circles, squares, etc.), and shadow/area fills.
    *   **Tree & Pie Chart:** Node shapes, edge styles, and slice explosion/labeling.
    *   **Scatter Plot:** Marker transparency (alpha), size, and overlap handling.
    *   **Heatmap:** Colormap choices (e.g., Viridis, Magma, custom), cell borders, and aspect ratios.
    *   **Radar Chart:** Grid structure, polygon fill transparency, and axis labeling.
    *   **Miscellaneous:** Observe styles for other specialized types.
4.  **Layout & Composition:** Legend placement, whitespace balance, margins, and subplot arrangements.
5.  **Typography:** Font weights, sizes, and colors for titles, axis labels, and annotations.

Return a concise bullet-point summary of the visual style diversity observed for these plot types in this batch.
"""


PLOT_FINAL_SUMMARY_PROMPT = """
Below are multiple visual analysis reports from a dataset of NeurIPS 2025 statistical plots.
Your goal is to synthesize these into a **"NeurIPS 2025 Statistical Plot Aesthetics Guide"**.

**Target Audience:** A researcher who wants to create plots that look "professional" and "NeurIPS-style".

**Critical Philosophy:** This is NOT about prescribing a single "correct" design. Instead, summarize the **multiple accepted design choices** in this field.

**Output Structure:**
1.  **The "NeurIPS Look" for Plots:** A high-level description of the prevailing aesthetic vibe (e.g., minimalistic, high-contrast, specific color schemes).
2.  **Detailed Style Options:**
    * **Color Palettes:** Common color sets for different data types (categorical, sequential).
    * **Axes & Grids:** Prevailing conventions for grid visibility and axis styling.
    * **Layout & Typography:** Common legend positions and font preferences.
3.  **Type-Specific Guidelines:**
    * Summarize specific aesthetic preferences for: *Bar Chart*, *Line Chart*, *Tree & Pie Chart*, *Scatter Plot*, *Heatmap*, *Radar Chart*, and *Miscellaneous*.
4.  **Common Pitfalls:** What design choices make a plot look "amateur" or "outdated" (e.g., default Excel/old Matplotlib styles)?

**Formatting Guidelines:**
- Use "Common choices include: [Option A], [Option B]" format.
- Frame everything as OBSERVATIONS not PRESCRIPTIONS.
- Focus on aesthetic quality and professional rendering.

**Input Reports:**
{all_reports}
"""



DIAGRAM_FINAL_SUMMARY_PROMPT = """
Below are multiple visual analysis reports from a dataset of NeurIPS 2025 method diagrams.
Your goal is to synthesize these into a **"NeurIPS 2025 Method Diagram Aesthetics Guide"**.

**Target Audience:** A researcher who wants to draw a diagram that looks "professional" and "accepted" by the community.

**Critical Philosophy:** This is NOT about prescribing a single "correct" design. Instead, summarize the **multiple accepted design choices** in this field.

**AVOID These Anti-Patterns:**
1. **DO NOT create rigid semantic bindings** like "Light Blue is standard for encoders" or "LLMs use brain icons". 
2. **DO NOT prescribe icon-to-concept mappings** like "🧠 Brain (LLM/Reasoning Core)".
3. **Present COLOR as aesthetic OPTIONS, not functional rules**.
   - Focus on: "These color combinations look good together" rather than "This component type requires this color"

**Output Structure:**
1.  **The "NeurIPS Look":** A high-level description of the prevailing aesthetic vibe.
2.  **Detailed Style Options:**
    * **Colors:** What aesthetically pleasing color palettes are common? List hex codes and describe combinations, NOT what component types they're "for".
    * **Shapes & Containers:** Common shape choices, border styles, shadow usage patterns.
    * **Lines & Arrows:** Common line styles, arrow types, and dashed line conventions.
    * **Layout & Composition:** Common layout patterns and information density preferences.
    * **Typography & Icons:** Common font choices. For icons: describe what icon OPTIONS are available for different purposes (format: "For [purpose], common options include: [icon1], [icon2]...")
3.  **Common Pitfalls:** What design choices make a diagram look "outdated" or "amateur"?
4.  **Domain-Specific Styles:** What are the common styles used in different domains? For example, agent papers will use detailed, cartoon-like illustrative styles more often, while theorectical papers will use more minimalistic styles.

**Formatting Guidelines for Options:**
- If 80%+ prevalence: "Most papers use [Option A]..."
- If multiple popular options: "Common choices include: [Option A] (~X%), [Option B] (~Y%)..."
- For icons/colors: Use "For representing [concept], observed options include: [list]" format
- Frame everything as OBSERVATIONS not PRESCRIPTIONS
- Emphasize aesthetic quality over semantic rules

**Input Reports:**
{all_reports}
"""





# Selective Prompt Selection
if MODE == "plot":
    BATCH_ANALYSIS_PROMPT = PLOT_BATCH_ANALYSIS_PROMPT
    FINAL_SUMMARY_PROMPT = PLOT_FINAL_SUMMARY_PROMPT
else:
    BATCH_ANALYSIS_PROMPT = DIAGRAM_BATCH_ANALYSIS_PROMPT
    FINAL_SUMMARY_PROMPT = DIAGRAM_FINAL_SUMMARY_PROMPT


# ================= 4. Core Logic =================

async def _generate_report(contents, model_name=None, provider=None, max_output_tokens=16000):
    from utils.config import ExpConfig
    from utils.generation_utils import call_model_with_retry_async
    from utils.model_registry import resolve_model

    settings = ExpConfig(dataset_name="PaperBananaBench", work_dir=WORK_DIR, task_name=MODE,
                         main_model_name=model_name or MODEL_NAME or "")
    model_name = settings.main_model_name
    resolve_model(model_name, provider=provider, operation="text", overrides=settings.model_profiles)
    responses = await call_model_with_retry_async(
        model_name=model_name, provider=provider, overrides=settings.model_profiles, contents=contents,
        config={"system_prompt": "", "temperature": 1.0, "candidate_num": 1,
                "max_output_tokens": max_output_tokens},
        max_attempts=3, retry_delay=1, error_context="style_guide_analysis",
    )
    if not responses or not isinstance(responses[0], str) or not responses[0].strip() or responses[0] == "Error":
        raise ValueError("Style analysis returned no usable text")
    return responses[0]


async def analyze_batch(sem, batch_index, image_paths, model_name=None, provider=None):
    """Analyze valid reference pixels through the shared multimodal text router."""
    from utils.legacy_ui_refinement import image_mime_type, atomic_write
    async with sem:
        images = []
        for path in image_paths:
            try:
                pixels = pathlib.Path(path).read_bytes()
                mime = image_mime_type(pixels)
                images.append({"type": "image", "source": {
                    "type": "base64", "media_type": mime,
                    "data": base64.b64encode(pixels).decode("ascii"),
                }})
            except (OSError, ValueError) as exc:
                print(f"Reference omitted: {path} ({type(exc).__name__})")
        if not images:
            return None
        try:
            report = await _generate_report(
                images + [{"type": "text", "text": BATCH_ANALYSIS_PROMPT}],
                model_name, provider,
            )
            atomic_write(pathlib.Path(BATCH_OUTPUT_DIR) / f"batch_{batch_index}.txt", report.encode("utf-8"))
            return f"=== Batch {batch_index} Analysis ({len(images)} images) ===\n{report}\n"
        except Exception as exc:
            print(f"Batch {batch_index} failed ({type(exc).__name__}); no report used for synthesis.")
            return None


async def main_task(model_name=None, provider=None):
    """Generate an aesthetic guide using only successfully analyzed references."""
    from utils.legacy_ui_refinement import atomic_write
    input_path = pathlib.Path(INPUT_JSON_PATH)
    output_path = pathlib.Path(OUTPUT_REPORT_PATH)
    if not output_path.is_absolute():
        output_path = WORK_DIR / output_path
    with input_path.open(encoding="utf-8") as source:
        data = json.load(source)
    all_image_paths = []
    for item in data:
        raw = item.get("path_to_gt_image")
        if raw:
            path = pathlib.Path(raw)
            if not path.is_absolute():
                path = DATA_DIR / path
            if path.exists():
                all_image_paths.append(path)
    if NUM_SAMPLES is not None:
        all_image_paths = all_image_paths[:NUM_SAMPLES]
    if not all_image_paths:
        raise ValueError("No reference images found; no guide was generated")
    if BATCH_SIZE < 1 or CONCURRENCY_LIMIT < 1:
        raise ValueError("Batch size and concurrency must be positive")
    batches = [all_image_paths[i:i + BATCH_SIZE] for i in range(0, len(all_image_paths), BATCH_SIZE)]
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    reports = await tqdm.gather(*[
        analyze_batch(sem, i, batch, model_name, provider) for i, batch in enumerate(batches)
    ], desc="Analyzing Batches")
    valid_reports = [report for report in reports if report]
    if not valid_reports:
        raise RuntimeError("All style-analysis batches failed; no guide was generated")
    if len(valid_reports) < len(reports):
        print(f"Coverage warning: {len(valid_reports)}/{len(reports)} batches analyzed successfully.")
    final_report = await _generate_report([
        {"type": "text", "text": FINAL_SUMMARY_PROMPT.format(
            num_batches=len(valid_reports), all_reports="\n\n".join(valid_reports),
        )},
    ], model_name, provider)
    atomic_write(output_path, final_report.encode("utf-8"))
    print(f"Style guide saved: {output_path}. Aesthetic observations are not scientific validation.")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a reference-based aesthetic style guide")
    parser.add_argument("--task", choices=("diagram", "plot"), default=MODE)
    parser.add_argument("--model", default=None, help="Registry model or explicit provider/model ID")
    parser.add_argument("--provider", choices=("gemini", "openai", "anthropic", "openrouter"), default=None)
    parser.add_argument("--input-json", type=pathlib.Path, default=None)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()
    MODE = args.task
    DATA_DIR = WORK_DIR / f"data/PaperBananaBench/{MODE}"
    INPUT_JSON_PATH = args.input_json or DATA_DIR / "ref.json"
    OUTPUT_REPORT_PATH = args.output or WORK_DIR / f"style_guides/neurips2025_{MODE}_style_guide.md"
    BATCH_ANALYSIS_PROMPT = PLOT_BATCH_ANALYSIS_PROMPT if MODE == "plot" else DIAGRAM_BATCH_ANALYSIS_PROMPT
    FINAL_SUMMARY_PROMPT = PLOT_FINAL_SUMMARY_PROMPT if MODE == "plot" else DIAGRAM_FINAL_SUMMARY_PROMPT
    asyncio.run(main_task(args.model, args.provider))
