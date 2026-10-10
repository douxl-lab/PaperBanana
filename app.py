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
Gradio-based Web UI for PaperBanana.
Replaces the Streamlit demo.py with a modern dark-themed interface.
"""

import gradio as gr
import asyncio
import base64
import json
import zipfile
from io import BytesIO
from PIL import Image
from pathlib import Path
import sys
import os
from datetime import datetime
import html

# ---------------------------------------------------------------------------
# Logo (base64-encoded for reliable serving in Gradio)
# ---------------------------------------------------------------------------
_logo_path = Path(__file__).parent / "assets" / "logo.jpg"
if _logo_path.exists():
    LOGO_B64 = base64.b64encode(_logo_path.read_bytes()).decode("ascii")
else:
    LOGO_B64 = ""

# ---------------------------------------------------------------------------
# Project imports (reuse demo.py's logic)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

import yaml
import shutil

configs_dir = Path(__file__).parent / "configs"
config_path = configs_dir / "model_config.yaml"
template_path = configs_dir / "model_config.template.yaml"


from agents.planner_agent import PlannerAgent
from agents.visualizer_agent import VisualizerAgent
from agents.stylist_agent import StylistAgent
from agents.critic_agent import CriticAgent
from agents.retriever_agent import RetrieverAgent
from agents.vanilla_agent import VanillaAgent
from agents.polish_agent import PolishAgent
from utils import config
from utils.legacy_generation_options import (
    generation_additional_info,
    normalize_legacy_input_content,
)
from utils.legacy_ui_results import build_evolution_stages, resolve_final_output
from utils.paperviz_processor import PaperVizProcessor
from utils.model_registry import get_default_model, list_model_options, resolve_model
from utils.legacy_ui_refinement import (
    refine_image, final_image, final_image_bytes, pil_to_png_bytes,
    new_run_id, atomic_write, save_results_json, summarize_results,
)

model_config_data = {}
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        model_config_data = yaml.safe_load(f) or {}


def get_config_val(section, key, env_var, default=""):
    val = os.getenv(env_var)
    if not val and section in model_config_data:
        val = model_config_data[section].get(key)
    return val or default


# ---------------------------------------------------------------------------
# Reuse core helpers from demo.py
# ---------------------------------------------------------------------------

def clean_text(text):
    if not text:
        return text
    if isinstance(text, str):
        return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return text


def base64_to_image(b64_str):
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        return Image.open(BytesIO(base64.b64decode(b64_str)))
    except Exception:
        return None


def create_sample_inputs(
    method_content,
    caption,
    aspect_ratio="16:9",
    figure_size=None,
    num_copies=10,
    max_critic_rounds=3,
    task_name="diagram",
):
    task_name = "plot" if "plot" in (task_name or "").lower() else "diagram"
    base_input = {
        "filename": "demo_input",
        "caption": caption,
        "content": normalize_legacy_input_content(method_content, task_name),
        "visual_intent": caption,
        "additional_info": generation_additional_info(aspect_ratio, figure_size),
        "max_critic_rounds": max_critic_rounds,
        "task_name": task_name,
    }
    inputs = []
    for i in range(num_copies):
        c = base_input.copy()
        c["filename"] = f"demo_input_candidate_{i}"
        c["candidate_id"] = str(i)
        inputs.append(c)
    return inputs


async def process_parallel_candidates(
    data_list, exp_mode="dev_planner_critic", retrieval_setting="auto",
    main_model_name="", image_gen_model_name="", task_name="diagram",
):
    task_name = "plot" if "plot" in (task_name or "").lower() else "diagram"
    exp_config = config.ExpConfig(
        dataset_name="PaperBananaBench",
        task_name=task_name,
        split_name="demo",
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        main_model_name=main_model_name,
        image_gen_model_name=image_gen_model_name,
        main_model_provider=resolve_model(main_model_name, operation="text", overrides=model_config_data.get("defaults", {}).get("model_profiles", {})).provider if main_model_name else "",
        image_gen_provider=resolve_model(image_gen_model_name, operation="image_generate", overrides=model_config_data.get("defaults", {}).get("model_profiles", {})).provider if image_gen_model_name else "",
        work_dir=Path(__file__).parent,
    )
    from utils.generation_utils import configure_generation_async, close_clients_async
    await configure_generation_async(exp_config.config_path)
    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )
    results = []
    try:
        async for result_data in processor.process_queries_batch(data_list, max_concurrent=exp_config.max_concurrent, do_eval=False):
            result_data["task_name"] = task_name
            results.append(result_data)
    finally:
        try:
            close = getattr(processor, "aclose", None)
            if close is not None:
                await close()
        finally:
            await close_clients_async()
    return results


async def refine_image_with_nanoviz(
    image_bytes, edit_prompt, aspect_ratio="21:9", image_size="2K", model_name=None,
):
    """Backward-compatible name for shared explicit-provider image editing."""
    model_name = model_name or get_config_val(
        "defaults", "image_gen_model_name", "IMAGE_GEN_MODEL_NAME", get_default_model("image"),
    )
    from utils.generation_utils import configure_generation_async, close_clients_async
    await configure_generation_async(config_path)
    try:
        return await refine_image(image_bytes, edit_prompt, model_name=model_name,
                                  aspect_ratio=aspect_ratio, image_size=image_size,
                                  overrides=model_config_data.get("defaults", {}).get("model_profiles", model_config_data.get("model_profiles", {})))
    finally:
        await close_clients_async()





def get_evolution_stages(result, exp_mode):
    return build_evolution_stages(result, exp_mode=exp_mode)


def get_final_image(result, exp_mode):
    """Return (PIL.Image, desc_text) for the best available stage."""
    selection = resolve_final_output(result, exp_mode=exp_mode)
    img = final_image(result, exp_mode)
    desc = clean_text(result.get(selection.text_key, "")) if selection.text_key else ""
    return img, desc


# ---------------------------------------------------------------------------
# Example content
# ---------------------------------------------------------------------------

EXAMPLE_METHOD = r"""## Methodology: The PaperBanana Framework

In this section, we present the architecture of PaperBanana, a reference-driven agentic framework for automated academic illustration. As illustrated in Figure \ref{fig:methodology_diagram}, PaperBanana orchestrates a collaborative team of five specialized agents—Retriever, Planner, Stylist, Visualizer, and Critic—to transform raw scientific content into publication-quality diagrams and plots. (See Appendix \ref{app_sec:agent_prompts} for prompts)

### Retriever Agent

Given the source context $S$ and the communicative intent $C$, the Retriever Agent identifies $N$ most relevant examples $\mathcal{E} = \{E_n\}_{n=1}^{N} \subset \mathcal{R}$ from the fixed reference set $\mathcal{R}$ to guide the downstream agents. As defined in Section \ref{sec:task_formulation}, each example $E_i \in \mathcal{R}$ is a triplet $(S_i, C_i, I_i)$.
To leverage the reasoning capabilities of VLMs, we adopt a generative retrieval approach where the VLM performs selection over candidate metadata:
$$
\mathcal{E} = \text{VLM}_{\text{Ret}} \left( S, C, \{ (S_i, C_i) \}_{E_i \in \mathcal{R}} \right)
$$

### Planner Agent

The Planner Agent serves as the cognitive core of the system. It takes the source context $S$, communicative intent $C$, and retrieved examples $\mathcal{E}$ as inputs:
$$
P = \text{VLM}_{\text{plan}}(S, C, \{ (S_i, C_i, I_i) \}_{E_i \in \mathcal{E}})
$$

### Stylist Agent

The Stylist refines each initial description $P$ into a stylistically optimized version $P^*$:
$$
P^* = \text{VLM}_{\text{style}}(P, \mathcal{G})
$$

### Visualizer Agent

The Visualizer Agent leverages an image generation model:
$$
I_t = \text{Image-Gen}(P_t)
$$

### Critic Agent

The Critic provides targeted feedback and produces a refined description:
$$
P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)
$$
The Visualizer-Critic loop iterates for $T=3$ rounds."""

EXAMPLE_CAPTION = "Figure 1: Overview of our PaperBanana framework. Given the source context and communicative intent, we first apply a Linear Planning Phase to retrieve relevant reference examples and synthesize a stylistically optimized description. We then use an Iterative Refinement Loop (consisting of Visualizer and Critic agents) to transform the description into visual output and conduct multi-round refinements to produce the final academic illustration."

PIPELINE_DESCRIPTIONS = {
    "demo_planner_critic": "Retriever \u2192 Planner \u2192 Visualizer \u2192 Critic \u2192 Visualizer (no Stylist)",
    "demo_full": "Retriever \u2192 Planner \u2192 Stylist \u2192 Visualizer \u2192 Critic \u2192 Visualizer",
}

# ---------------------------------------------------------------------------
# Custom CSS for dark theme matching the screenshot
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ---- Global ---- */
.gradio-container {
    max-width: 1400px !important;
    width: 100% !important;
    margin: 0 auto !important;
}
.gradio-container > .main {
    max-width: 100% !important;
}

/* ---- Accent colour (orange/amber) ---- */
.accent { color: #f59e0b; }
.orange-btn {
    background: linear-gradient(135deg, #f59e0b, #d97706) !important;
    color: #fff !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    border-radius: 10px !important;
}
.orange-btn:hover {
    background: linear-gradient(135deg, #d97706, #b45309) !important;
}

/* ---- Section labels ---- */
.section-label {
    text-transform: uppercase;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 1.5px;
    color: #f59e0b;
    margin-bottom: 8px;
}

/* ---- Card-like blocks ---- */
.settings-panel, .input-panel, .results-panel {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px;
}

/* ---- Candidate gallery (orange border) ---- */
.candidate-card {
    border: 2px solid #f59e0b;
    border-radius: 12px;
    padding: 8px;
    text-align: center;
}

/* ---- Footer ---- */
#footer-row {
    text-align: center;
    padding: 12px 0;
    font-size: 13px;
    color: #6b7280;
}
#footer-row a { color: #f59e0b; text-decoration: none; }
#footer-row a:hover { text-decoration: underline; }

/* ---- Evolution timeline ---- */
.evo-stage { margin-bottom: 12px; }
.evo-stage-title { font-weight: 600; color: #f59e0b; }

/* ---- Status ---- */
.status-box {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 10px 16px;
    background: #f9fafb;
    font-size: 14px;
}

/* ---- Left settings column: prevent label truncation ---- */
.left-settings { min-width: 320px; }
.left-settings .gr-block label,
.left-settings .gr-input label,
.left-settings label span {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
}
.left-settings .gradio-dropdown,
.left-settings .gradio-textbox,
.left-settings .gradio-slider,
.left-settings .gradio-number {
    min-width: 0 !important;
}

/* ---- Compact info text ---- */
.gradio-dropdown .wrap .info,
.gradio-textbox .wrap .info { font-size: 0.8em !important; }

/* ---- Header button style (outlined) ---- */
.header-link-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    border-radius: 20px;
    border: 1.5px solid #d1d5db;
    background: #fff;
    color: #374151;
    font-weight: 600;
    font-size: 14px;
    text-decoration: none;
    transition: border-color 0.2s, background 0.2s;
}
.header-link-btn:hover {
    border-color: #f59e0b;
    background: #fffbeb;
    text-decoration: none;
    color: #374151;
}
"""

# ---------------------------------------------------------------------------
# Build the Gradio Blocks UI
# ---------------------------------------------------------------------------

def build_app():

    default_main_model = get_config_val("defaults", "main_model_name", "MAIN_MODEL_NAME", get_default_model("text"))
    default_image_model = get_config_val("defaults", "image_gen_model_name", "IMAGE_GEN_MODEL_NAME", get_default_model("image"))

    with gr.Blocks(title="PaperBanana") as app:
        # ---- State to hold results across interactions ----
        gen_results_state = gr.State([])
        gen_mode_state = gr.State("demo_planner_critic")
        gen_timestamp_state = gr.State("")
        gen_json_path_state = gr.State("")

        # ================================================================
        # HEADER
        # ================================================================
        gr.HTML(f"""
        <div style="background: #fff; border-radius: 16px; padding: 24px 36px; margin-bottom: 16px; width: 100%;
                    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;
                    border: 1px solid #e5e7eb;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <img src="data:image/jpeg;base64,{LOGO_B64}" alt="PaperBanana logo"
                     style="height: 60px; width: auto; border-radius: 10px; object-fit: contain;" />
                <div>
                    <p style="font-size: 28px; font-weight: 800; color: #111; margin: 0 0 4px 0;">
                        PaperBanana
                    </p>
                    <div style="display: flex; gap: 6px; align-items: center;">
                        <span style="display:inline-block; padding:3px 12px; border-radius:12px; font-size:11px; font-weight:600; background:#f59e0b; color:#fff;">Multi-Agent</span>
                        <span style="display:inline-block; padding:3px 12px; border-radius:12px; font-size:11px; font-weight:600; background:#f59e0b; color:#fff;">Scientific Diagrams</span>
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 10px; align-items: center;">
                <a href="https://arxiv.org/abs/2601.23265" target="_blank" class="header-link-btn">
                    &#128196; Paper
                </a>
                <a href="https://github.com/dwzhu-pku/PaperBanana" target="_blank" class="header-link-btn">
                    &#128187; GitHub
                </a>
            </div>
        </div>
        """)

        # ================================================================
        # API KEYS ACCORDION
        # ================================================================
        with gr.Accordion("API Keys", open=False):
            gr.Markdown(
                "Single-user server: applied credentials are shared by all browser sessions in this process. "
                "The selected model determines the provider. Use openrouter/<provider>/<model> for OpenRouter."
            )
            with gr.Row():
                openrouter_key_input = gr.Textbox(
                    label="OpenRouter API Key (optional)", type="password", placeholder="sk-or-...",
                    value="",
                )
                google_key_input = gr.Textbox(
                    label="Google API Key (optional)", type="password", placeholder="AIza...",
                    value="",
                )
                openai_key_input = gr.Textbox(label="OpenAI API Key (optional)", type="password", value="")
                anthropic_key_input = gr.Textbox(label="Anthropic API Key (optional)", type="password", value="")
            gr.Markdown("Keys remain in server-process memory until exit; they are not written to results or config. Blank inputs keep existing credentials. Do not expose this UI to untrusted users.")

            def apply_keys(or_key, g_key, o_key, a_key):
                for variable, value in (("OPENROUTER_API_KEY", or_key), ("GOOGLE_API_KEY", g_key),
                                        ("OPENAI_API_KEY", o_key), ("ANTHROPIC_API_KEY", a_key)):
                    if value:
                        os.environ[variable] = value
                from utils.generation_utils import reinitialize_clients
                initialized = reinitialize_clients()
                if initialized:
                    return f"Server-process clients initialized: {', '.join(initialized)}."
                return "No API clients initialized; configure credentials for the selected model provider."

            apply_keys_btn = gr.Button("Apply Keys", size="sm")
            keys_status = gr.Textbox(visible=False)
            apply_keys_btn.click(apply_keys, inputs=[openrouter_key_input, google_key_input, openai_key_input, anthropic_key_input], outputs=[keys_status], concurrency_id="model_requests", concurrency_limit=1)

        # ================================================================
        # TABS
        # ================================================================
        with gr.Tabs():

            # ============================================================
            # TAB 1 — Generate Candidates
            # ============================================================
            with gr.TabItem("Generate Candidates"):
                with gr.Row():
                    # ---------- LEFT COLUMN: SETTINGS ----------
                    with gr.Column(scale=1, min_width=280, elem_classes=["left-settings"]):
                        gr.HTML('<p class="section-label">Settings</p>')

                        pipeline_mode = gr.Dropdown(
                            choices=["demo_planner_critic", "demo_full"],
                            value="demo_full",
                            label="Pipeline Mode",
                            info="Select which agent pipeline to use",
                        )
                        task_name = gr.Dropdown(
                            choices=["diagram", "plot"],
                            value="diagram",
                            label="Output Type",
                            info="Generate a scientific diagram or a statistical plot",
                        )
                        pipeline_desc = gr.Textbox(
                            label="Pipeline Description",
                            value=PIPELINE_DESCRIPTIONS["demo_full"],
                            interactive=False, lines=2,
                        )
                        pipeline_mode.change(
                            lambda m: PIPELINE_DESCRIPTIONS.get(m, ""),
                            inputs=[pipeline_mode],
                            outputs=[pipeline_desc],
                        )

                        retrieval_setting = gr.Dropdown(
                            choices=["auto", "manual", "random", "none"],
                            value="auto",
                            label="Retrieval Setting",
                            info="How to retrieve reference examples",
                        )
                        num_candidates = gr.Number(
                            value=10, minimum=1, maximum=20, step=1,
                            label="Number of Candidates",
                        )
                        aspect_ratio = gr.Dropdown(
                            choices=["16:9", "21:9", "3:2"],
                            value="21:9",
                            label="Aspect Ratio",
                        )
                        figure_size = gr.Dropdown(
                            choices=["1-3cm", "4-6cm", "7-9cm", "10-13cm", "14-17cm"],
                            value="7-9cm",
                            label="Figure Size",
                        )
                        max_critic_rounds = gr.Slider(
                            minimum=0, maximum=5, value=3, step=1,
                            label="Max Critic Rounds",
                        )
                        main_model_name = gr.Dropdown(
                            choices=list(dict.fromkeys([default_main_model, *list_model_options("text")])),
                            allow_custom_value=True, label="Model Name",
                            info="Model name to use for reasoning",
                            value=default_main_model,
                        )
                        image_model_name = gr.Dropdown(
                            choices=list(dict.fromkeys([default_image_model, *list_model_options("image")])),
                            allow_custom_value=True, label="Image Generation Model",
                            info="Model for generating diagram images",
                            value=default_image_model,
                        )
                        save_results = gr.Dropdown(
                            choices=["Yes", "No"],
                            value="Yes",
                            label="Save Results",
                        )

                    # ---------- RIGHT COLUMN: INPUT + OUTPUT ----------
                    with gr.Column(scale=3):
                        gr.HTML('<p class="section-label">Input</p>')

                        with gr.Row():
                            method_example = gr.Dropdown(
                                choices=["None", "PaperBanana Framework"],
                                value="PaperBanana Framework",
                                label="Load Example (Method)",
                            )
                            caption_example = gr.Dropdown(
                                choices=["None", "PaperBanana Framework"],
                                value="PaperBanana Framework",
                                label="Load Example (Caption)",
                            )

                        with gr.Row():
                            method_content = gr.Textbox(
                                label="Method Content / Plot Data",
                                value=EXAMPLE_METHOD,
                                lines=12, max_lines=30,
                            )
                            caption_input = gr.Textbox(
                                label="Figure Caption / Visual Intent",
                                value=EXAMPLE_CAPTION,
                                lines=12, max_lines=30,
                            )

                        # Wire example selectors
                        def load_method_example(choice):
                            return EXAMPLE_METHOD if choice == "PaperBanana Framework" else ""
                        def load_caption_example(choice):
                            return EXAMPLE_CAPTION if choice == "PaperBanana Framework" else ""

                        method_example.change(load_method_example, inputs=[method_example], outputs=[method_content])
                        caption_example.change(load_caption_example, inputs=[caption_example], outputs=[caption_input])

                        generate_btn = gr.Button(
                            "✨ Generate Candidates", variant="primary",
                            elem_classes=["orange-btn"], size="lg",
                        )

                # ---- Status ----
                status_text = gr.Textbox(label="Status", interactive=False, lines=1)

                # ---- Results ----
                gr.HTML('<p class="section-label" style="margin-top:16px;">Generated Candidates</p>')
                results_gallery = gr.Gallery(
                    label="Generated Candidates",
                    columns=3, height="auto", object_fit="contain",
                )
                with gr.Accordion("Evolution Timeline", open=False):
                    evolution_html = gr.HTML("")
                with gr.Accordion("Download All (ZIP)", open=False):
                    zip_file_output = gr.File(label="ZIP download")

                # ---- Generate handler ----
                def run_generate(
                    method_text, caption_text, pipe_mode, task_name, ret_setting,
                    n_cands, ar, max_rounds, m_model, img_model,
                    figure_size, save_results,
                    progress=gr.Progress(track_tqdm=True),
                ):
                    if not method_text or not caption_text:
                        raise gr.Error("Please provide both method content and caption.")

                    n_cands = int(n_cands)
                    max_rounds = int(max_rounds)
                    if not 1 <= n_cands <= 20 or not 0 <= max_rounds <= 5:
                        raise gr.Error("Choose 1–20 candidates and 0–5 critic rounds.")
                    timestamp_str = new_run_id()

                    progress(0, desc="Preparing inputs...")
                    input_data = create_sample_inputs(
                        method_content=method_text, caption=caption_text,
                        aspect_ratio=ar, figure_size=figure_size,
                        num_copies=n_cands, max_critic_rounds=max_rounds,
                        task_name=task_name,
                    )

                    progress(0.1, desc=f"Generating {n_cands} candidates in parallel...")
                    try:
                        results = asyncio.run(process_parallel_candidates(
                            input_data, exp_mode=pipe_mode, retrieval_setting=ret_setting,
                            main_model_name=m_model, image_gen_model_name=img_model,
                            task_name=task_name,
                        ))
                    except Exception as exc:
                        raise gr.Error(f"Generation failed ({type(exc).__name__}); check model configuration and credentials.")

                    progress(0.9, desc="Saving results...")

                    results_dir = Path(__file__).parent / "results" / "demo"
                    json_filename = None
                    save_warning = ""
                    if save_results != "No":
                        try:
                            json_filename = save_results_json(results, results_dir, timestamp_str)
                        except OSError:
                            save_warning = " JSON save failed."

                    # Build gallery images
                    gallery_images = []
                    for idx, res in enumerate(results):
                        img, _ = get_final_image(res, pipe_mode)
                        if img:
                            gallery_images.append((img, f"Candidate {res.get('candidate_id', idx)} ({res.get('status', 'legacy output')})"))

                    # Build evolution HTML
                    evo_parts = []
                    for idx, res in enumerate(results):
                        stages = get_evolution_stages(res, pipe_mode)
                        if stages:
                            evo_parts.append(f"<h4>Candidate {idx} ({len(stages)} stages)</h4>")
                            for st in stages:
                                evo_parts.append(f'<span class="evo-stage-title">{html.escape(st["name"])}</span>: {html.escape(st["description"])}<br/>')
                    evo_html = "".join(evo_parts) if evo_parts else "<p>No evolution data available.</p>"

                    # Build ZIP
                    zip_path = None
                    if save_results != "No":
                        try:
                            zip_filename = results_dir / f"papervizagent_candidates_{timestamp_str}.zip"
                            buf = BytesIO()
                            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                                for idx, res in enumerate(results):
                                    img, _ = get_final_image(res, pipe_mode)
                                    if img:
                                        ib = BytesIO()
                                        img.save(ib, format="PNG")
                                        zf.writestr(f"candidate_{idx}.png", ib.getvalue())
                            buf.seek(0)
                            atomic_write(zip_filename, buf.getvalue())
                            zip_path = str(zip_filename)
                        except OSError:
                            save_warning += " ZIP save failed."

                    counts = summarize_results(results, pipe_mode)
                    status = f"{counts['success']} succeeded, {counts['failed']} failed, {counts['incomplete']} incomplete candidates."
                    status += save_warning
                    if json_filename and Path(str(json_filename)).exists():
                        status += f" JSON saved to {Path(str(json_filename)).name}."

                    progress(1.0, desc="Done!")
                    return (
                        gallery_images,       # results_gallery
                        evo_html,             # evolution_html
                        zip_path,             # zip_file_output
                        status,               # status_text
                        results,              # gen_results_state
                        pipe_mode,            # gen_mode_state
                        timestamp_str,        # gen_timestamp_state
                    )

                generate_btn.click(
                    fn=run_generate,
                    inputs=[
                        method_content, caption_input, pipeline_mode, task_name, retrieval_setting,
                        num_candidates, aspect_ratio, max_critic_rounds,
                        main_model_name, image_model_name,
                        figure_size, save_results,
                    ],
                    outputs=[
                        results_gallery, evolution_html, zip_file_output, status_text,
                        gen_results_state, gen_mode_state, gen_timestamp_state,
                    ],
                    concurrency_id="model_requests", concurrency_limit=1,
                )

            # ============================================================
            # TAB 2 — Refine Image
            # ============================================================
            with gr.TabItem("Refine Image"):
                gr.Markdown("### Refine and upscale your diagram to high resolution (2K/4K)")
                gr.Markdown("Upload an image and describe changes. Resolution and ratio must be supported by the selected model; legacy OpenAI image models use 1K with 1:1, 3:2, or 2:3.")

                with gr.Row():
                    with gr.Column():
                        refine_upload = gr.Image(label="Upload Image", type="filepath", image_mode=None, format="png", height=400)
                    with gr.Column():
                        refine_prompt = gr.Textbox(
                            label="Edit Instructions", lines=6,
                            placeholder="E.g., 'Change the color scheme to match academic paper style' or 'Keep everything the same but output in higher resolution'",
                        )
                        with gr.Row():
                            refine_resolution = gr.Dropdown(choices=["1K", "2K", "4K"], value="2K", label="Resolution")
                            refine_aspect = gr.Dropdown(choices=["1:1", "21:9", "16:9", "3:2", "2:3"], value="21:9", label="Aspect Ratio")
                        refine_model = gr.Dropdown(
                            choices=list(dict.fromkeys([default_image_model, *list_model_options("image_edit")])),
                            allow_custom_value=True, value=default_image_model, label="Image Editing Model",
                        )
                        refine_btn = gr.Button("Refine Image", variant="primary", elem_classes=["orange-btn"])

                refine_status = gr.Textbox(label="Status", interactive=False)

                with gr.Row():
                    refine_before = gr.Image(label="Before", image_mode=None, format="png", interactive=False, height=400)
                    refine_after = gr.Image(label="After", image_mode=None, format="png", interactive=False, height=400)
                refine_download = gr.File(label="Download refined image")

                def run_refine(image_path, prompt, resolution, ar, model_name):
                    if image_path is None:
                        raise gr.Error("Please upload an image first.")
                    if not prompt:
                        raise gr.Error("Please provide edit instructions.")

                    image_bytes = Path(image_path).read_bytes()
                    with Image.open(BytesIO(image_bytes)) as source_image:
                        pil_img = source_image.copy()
                    refined_bytes, msg = asyncio.run(refine_image_with_nanoviz(
                        image_bytes, prompt, aspect_ratio=ar, image_size=resolution, model_name=model_name,
                    ))

                    if not refined_bytes:
                        raise gr.Error(msg)

                    refined_img = Image.open(BytesIO(refined_bytes))

                    # Save to temp file for download
                    ts = new_run_id()
                    out_dir = Path(__file__).parent / "results" / "demo"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"refined_{resolution}_{ts}.png"
                    atomic_write(out_path, pil_to_png_bytes(refined_img))

                    return pil_img, refined_img, str(out_path), msg

                refine_btn.click(
                    fn=run_refine,
                    inputs=[refine_upload, refine_prompt, refine_resolution, refine_aspect, refine_model],
                    outputs=[refine_before, refine_after, refine_download, refine_status],
                    concurrency_id="model_requests", concurrency_limit=1,
                )

        # ================================================================
        # FOOTER
        # ================================================================
        gr.HTML("""
        <div id="footer-row">
            <a href="https://github.com/dwzhu-pku/PaperBanana" target="_blank">GitHub</a> &middot;
            <a href="https://arxiv.org/abs/2601.23265" target="_blank">Paper</a><br/>
            PaperBanana &copy; 2026
        </div>
        """)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.amber,
            secondary_hue=gr.themes.colors.gray,
            neutral_hue=gr.themes.colors.gray,
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
    )
