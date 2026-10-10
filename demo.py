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
Parallel Streamlit Demo for PaperBanana
Accepts user text input, duplicates it 10 times, and runs parallel processing
to generate multiple diagram candidates for comparison.
"""

import streamlit as st
import asyncio
import base64
import json
from io import BytesIO
from PIL import Image
from pathlib import Path
import sys
import os
from datetime import datetime
import html

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("DEBUG: Importing agents...")
import yaml
import shutil
configs_dir = Path(__file__).parent / "configs"
config_path = configs_dir / "model_config.yaml"
template_path = configs_dir / "model_config.template.yaml"

try:
    from agents.planner_agent import PlannerAgent
    print("DEBUG: Imported PlannerAgent")
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    print("DEBUG: Imported all agents")
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
    print("DEBUG: Imported utils")

    model_config_data = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            model_config_data = yaml.safe_load(f) or {}

    def get_config_val(section, key, env_var, default=""):
        val = os.getenv(env_var)
        if not val and section in model_config_data:
            val = model_config_data[section].get(key)
        return val or default

except ImportError as e:
    print(f"DEBUG: ImportError: {e}")
    import traceback
    traceback.print_exc()
    raise e
except Exception as e:
    print(f"DEBUG: Exception during import: {e}")
    import traceback
    traceback.print_exc()
    raise e

st.set_page_config(
    layout="wide",
    page_title="PaperBanana Parallel Demo",
    page_icon="🍌"
)

def clean_text(text):
    """Clean text by removing invalid UTF-8 surrogate characters."""
    if not text:
        return text
    if isinstance(text, str):
        # Remove surrogate characters that cause UnicodeEncodeError
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    return text

def base64_to_image(b64_str):
    """Convert base64 string to PIL Image."""
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        image_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data))
    except Exception:
        return None

def create_sample_inputs(
    method_content,
    caption,
    diagram_type="Pipeline",
    aspect_ratio="16:9",
    figure_size=None,
    num_copies=10,
    max_critic_rounds=3,
    task_name="diagram",
):
    """Create multiple copies of the input data for parallel processing."""
    task_name = "plot" if "plot" in (task_name or "").lower() else "diagram"
    base_input = {
        "filename": "demo_input",
        "caption": caption,
        "content": normalize_legacy_input_content(method_content, task_name),
        "visual_intent": caption,
        "additional_info": generation_additional_info(aspect_ratio, figure_size),
        "max_critic_rounds": max_critic_rounds,  # Add critic rounds control
        "task_name": task_name,
    }
    
    # Create num_copies identical inputs, each with a unique identifier
    inputs = []
    for i in range(num_copies):
        input_copy = base_input.copy()
        input_copy["filename"] = f"demo_input_candidate_{i}"
        input_copy["candidate_id"] = str(i)
        inputs.append(input_copy)
    
    return inputs

async def process_parallel_candidates(
    data_list,
    exp_mode="dev_planner_critic",
    retrieval_setting="auto",
    main_model_name="",
    image_gen_model_name="",
    task_name="diagram",
):
    """Process multiple candidates in parallel using PaperVizProcessor."""
    task_name = "plot" if "plot" in (task_name or "").lower() else "diagram"
    # Create experiment config
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
    
    # Initialize processor with all agents
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
    
    # Process all candidates in parallel (concurrency controlled by processor)
    results = []
    concurrent_num = exp_config.max_concurrent
    
    try:
        async for result_data in processor.process_queries_batch(
            data_list, max_concurrent=concurrent_num, do_eval=False
        ):
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
    """Extract all evolution stages (images and descriptions) from the result."""
    return build_evolution_stages(result, exp_mode=exp_mode)

def display_candidate_result(result, candidate_id, exp_mode):
    """Display a single candidate result."""
    selection = resolve_final_output(result, exp_mode=exp_mode)
    final_image_key = selection.image_key
    final_desc_key = selection.text_key
    
    if result.get("status") == "failed":
        st.warning("Generation failed; any shown image is a retained intermediate output.")

    # Display the final image
    if final_image_key and final_image_key in result:
        img = final_image(result, exp_mode)
        if img:
            st.image(img, use_container_width=True, caption=f"Candidate {candidate_id} (Final)")
            
            # Add download button
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            st.download_button(
                label="⬇️ Download",
                data=buffered.getvalue(),
                file_name=f"candidate_{candidate_id}.png",
                mime="image/png",
                key=f"download_candidate_{candidate_id}",
                use_container_width=True
            )
        else:
            st.error(f"Failed to decode image for Candidate {candidate_id}")
    else:
        st.warning(f"No image generated for Candidate {candidate_id}")
    
    # Show evolution timeline in an expander
    stages = get_evolution_stages(result, exp_mode)
    if len(stages) > 1:
        with st.expander(f"🔄 View Evolution Timeline ({len(stages)} stages)", expanded=False):
            st.caption("See how the diagram evolved through different pipeline stages")
            
            for idx, stage in enumerate(stages):
                st.markdown(f"### {stage['name']}")
                st.caption(stage['description'])
                
                # Display the image for this stage
                stage_img = base64_to_image(result.get(stage['image_key']))
                if stage_img:
                    st.image(stage_img, use_container_width=True)
                
                # Show description
                if stage['desc_key'] in result:
                    with st.expander(f"📝 Description", expanded=False):
                        cleaned_desc = clean_text(result[stage['desc_key']])
                        st.write(cleaned_desc)
                
                # Show critic suggestions if available
                if 'suggestions_key' in stage and stage['suggestions_key'] in result:
                    suggestions = result[stage['suggestions_key']]
                    with st.expander(f"💡 Critic Suggestions", expanded=False):
                        cleaned_sugg = clean_text(suggestions)
                        if cleaned_sugg.strip() == "No changes needed.":
                            st.success("✅ No changes needed - iteration stopped.")
                        else:
                            st.write(cleaned_sugg)
                
                # Add separator between stages (except for the last one)
                if idx < len(stages) - 1:
                    st.divider()
    else:
        # If only one stage, show description in simpler expander
        with st.expander(f"📝 View Description", expanded=False):
            if final_desc_key and final_desc_key in result:
                # Clean the text to remove invalid UTF-8 characters
                cleaned_desc = clean_text(result[final_desc_key])
                st.write(cleaned_desc)
            else:
                st.info("No description available")

def main():
    st.title("🍌 PaperBanana Demo")
    st.markdown("AI-powered scientific diagram generation and refinement")
    st.caption("Single-user server: credentials in config/environment are shared by this process. Selected models choose their providers; credentials are excluded from exports.")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Generate Candidates", "✨ Refine Image"])
    
    # ==================== TAB 1: Generate Candidates ====================
    with tab1:
        st.markdown("### Generate multiple diagram candidates from your method section and caption")
        
        # Sidebar configuration for Tab 1
        with st.sidebar:
            st.title("⚙️ Generation Settings")
            
            exp_mode = st.selectbox(
                "Pipeline Mode",
                ["demo_full", "demo_planner_critic"],
                index=0,
                key="tab1_exp_mode",
                help="Select which agent pipeline to use"
            )
            
            mode_info = {
                "demo_planner_critic": "Retriever → Planner → Visualizer → Critic → Visualizer (no Stylist)",
                "demo_full": "Retriever → Planner → Stylist → Visualizer → Critic → Visualizer. (The stylist can make the diagram more aesthetically pleasing, but prone to be overly simplied. So we recommend trying both modes and select the best one)"
            }
            st.info(f"**Pipeline:** {mode_info[exp_mode]}")
            
            retrieval_setting = st.selectbox(
                "Retrieval Setting",
                ["auto", "manual", "random", "none"],
                index=0,
                key="tab1_retrieval_setting",
                help="How to retrieve reference examples: auto (automatic selection), manual (use specified references), random (random selection), none (no retrieval)"
            )

            task_name = st.selectbox(
                "Output Type",
                ["diagram", "plot"],
                index=0,
                key="tab1_task_name",
                help="Generate a scientific diagram or a statistical plot",
            )
            
            num_candidates = st.number_input(
                "Number of Candidates",
                min_value=1,
                max_value=20,
                value=10,
                key="tab1_num_candidates",
                help="How many parallel candidates to generate"
            )
            
            aspect_ratio = st.selectbox(
                "Aspect Ratio",
                ["21:9", "16:9", "3:2"],
                key="tab1_aspect_ratio",
                help="Aspect ratio for the generated diagrams"
            )

            figure_size = st.selectbox(
                "Figure Size",
                ["1-3cm", "4-6cm", "7-9cm", "10-13cm", "14-17cm"],
                index=2,
                key="tab1_figure_size",
                help="Target figure size; providers that support image_size map this to 1k, 2k, or 4k",
            )
            
            max_critic_rounds = st.number_input(
                "Max Critic Rounds",
                min_value=0,
                max_value=5,
                value=3,
                key="tab1_max_critic_rounds",
                help="Maximum number of critic refinement iterations"
            )
            
            save_results = st.selectbox("Save Results", ["Yes", "No"], key="save_results")

            default_model = get_config_val("defaults", "main_model_name", "MAIN_MODEL_NAME", get_default_model("text"))
            text_model_presets = list(dict.fromkeys([default_model, *list_model_options("text")])) + ["Custom"]
            text_model_selection = st.selectbox(
                "Main Model Name",
                text_model_presets,
                index=0,
                key="tab1_model_name",
                help="Vision-language model for understanding and describing diagrams"
            )
            if text_model_selection == "Custom":
                main_model_name = st.text_input(
                    "Custom Main Model",
                    value="",
                    key="tab1_main_model_name_custom",
                    placeholder="e.g., openrouter/google/gemini-3.1-pro"
                )
            else:
                main_model_name = text_model_selection

            default_image_model = get_config_val("defaults", "image_gen_model_name", "IMAGE_GEN_MODEL_NAME", get_default_model("image"))
            image_model_presets = list(dict.fromkeys([default_image_model, *list_model_options("image")])) + ["Custom"]
            image_model_selection = st.selectbox(
                "Image Generation Model Name",
                image_model_presets,
                index=0,
                key="tab1_image_model_name",
                help="Model for generating diagram images"
            )
            if image_model_selection == "Custom":
                image_gen_model_name = st.text_input(
                    "Custom Image Generation Model",
                    value="",
                    key="tab1_image_gen_model_name_custom",
                    placeholder="e.g., openrouter/openai/gpt-image-1"
                )
            else:
                image_gen_model_name = image_model_selection
        
        st.divider()
        
        # Input section
        st.markdown("## 📝 Input")
        
        # Example content
        example_method = r"""## Methodology: The PaperBanana Framework
        
        In this section, we present the architecture of PaperBanana, a reference-driven agentic framework for automated academic illustration. As illustrated in Figure \ref{fig:methodology_diagram}, PaperBanana orchestrates a collaborative team of five specialized agents—Retriever, Planner, Stylist, Visualizer, and Critic—to transform raw scientific content into publication-quality diagrams and plots. (See Appendix \ref{app_sec:agent_prompts} for prompts)

### Retriever Agent

Given the source context $S$ and the communicative intent $C$, the Retriever Agent identifies $N$ most relevant examples $\mathcal{E} = \{E_n\}_{n=1}^{N} \subset \mathcal{R}$ from the fixed reference set $\mathcal{R}$ to guide the downstream agents. As defined in Section \ref{sec:task_formulation}, each example $E_i \in \mathcal{R}$ is a triplet $(S_i, C_i, I_i)$.
To leverage the reasoning capabilities of VLMs, we adopt a generative retrieval approach where the VLM performs selection over candidate metadata:
$$
\mathcal{E} = \text{VLM}_{\text{Ret}} \left( S, C, \{ (S_i, C_i) \}_{E_i \in \mathcal{R}} \right)
$$
Specifically, the VLM is instructed to rank candidates by matching both research domain (e.g., Agent & Reasoning) and diagram type (e.g., pipeline, architecture), with visual structure being prioritized over topic similarity. By explicitly reasoned selection of reference illustrations $I_i$ whose corresponding contexts $(S_i, C_i)$ best match the current requirements, the Retriever provides a concrete foundation for both structural logic and visual style.

### Planner Agent

The Planner Agent serves as the cognitive core of the system. It takes the source context $S$, communicative intent $C$, and retrieved examples $\mathcal{E}$ as inputs. By performing in-context learning from the demonstrations in $\mathcal{E}$, the Planner translates the unstructured or structured data in $S$ into a comprehensive and detailed textual description $P$ of the target illustration:
$$
P = \text{VLM}_{\text{plan}}(S, C, \{ (S_i, C_i, I_i) \}_{E_i \in \mathcal{E}})
$$

### Stylist Agent

To ensure the output adheres to the aesthetic standards of modern academic manuscripts, the Stylist Agent acts as a design consultant.
A primary challenge lies in defining a comprehensive “academic style,” as manual definitions are often incomplete.
To address this, the Stylist traverses the entire reference collection $\mathcal{R}$ to automatically synthesize an *Aesthetic Guideline* $\mathcal{G}$ covering key dimensions such as color palette, shapes and containers, lines and arrows, layout and composition, and typography and icons (see Appendix \ref{app_sec:auto_summarized_style_guide} for the summarized guideline and implementation details). Armed with this guideline, the Stylist refines each initial description $P$ into a stylistically optimized version $P^*$:
$$
P^* = \text{VLM}_{\text{style}}(P, \mathcal{G})
$$
This ensures that the final illustration is not only accurate but also visually professional.

### Visualizer Agent

After receiving the stylistically optimized description $P^*$, the Visualizer Agent collaborates with the Critic Agent to render academic illustrations and iteratively refine their quality. The Visualizer Agent leverages an image generation model to transform textual descriptions into visual output. In each iteration $t$, given a description $P_t$, the Visualizer generates:
$$
I_t = \text{Image-Gen}(P_t)
$$
where the initial description $P_0$ is set to $P^*$.

### Critic Agent

The Critic Agent forms a closed-loop refinement mechanism with the Visualizer by closely examining the generated image $I_t$ and providing refined description $P_{t+1}$ to the Visualizer. Upon receiving the generated image $I_t$ at iteration $t$, the Critic inspects it against the original source context $(S, C)$ to identify factual misalignments, visual glitches, or areas for improvement. It then provides targeted feedback and produces a refined description $P_{t+1}$ that addresses the identified issues:
$$
P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)
$$
This revised description is then fed back to the Visualizer for regeneration. The Visualizer-Critic loop iterates for $T=3$ rounds, with the final output being $I = I_T$. This iterative refinement process ensures that the final illustration meets the high standards required for academic dissemination.

### Extension to Statistical Plots

The framework extends to statistical plots by adjusting the Visualizer and Critic agents. For numerical precision, the Visualizer converts the description $P_t$ into executable Python Matplotlib code: $I_t = \text{VLM}_{\text{code}}(P_t)$. The Critic evaluates the rendered plot and generates a refined description $P_{t+1}$ addressing inaccuracies or imperfections: $P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)$. The same $T=3$ round iterative refinement process applies. While we prioritize this code-based approach for accuracy, we also explore direct image generation in Section \ref{sec:discussion}. See Appendix \ref{app_sec:plot_agent_prompt} for adjusted prompts."""

        example_caption = "Figure 1: Overview of our PaperBanana framework. Given the source context and communicative intent, we first apply a Linear Planning Phase to retrieve relevant reference examples and synthesize a stylistically optimized description. We then use an Iterative Refinement Loop (consisting of Visualizer and Critic agents) to transform the description into visual output and conduct multi-round refinements to produce the final academic illustration."
        
        col_input1, col_input2 = st.columns([3, 2])
        
        with col_input1:
            # Example selector for method content
            method_example = st.selectbox(
                "Load Example (Method)",
                ["None", "PaperBanana Framework"],
                key="method_example_selector"
            )
            
            # Set value based on example selection or session state
            if method_example == "PaperBanana Framework":
                method_value = example_method
            else:
                method_value = st.session_state.get("method_content", "")
            
            method_content = st.text_area(
                "Method Section Content / Plot Data (Markdown or JSON recommended)",
                value=method_value,
                height=250,
                placeholder="Paste the method section content here...",
                help="The method section or plot data that describes the figure. Markdown and JSON are supported."
            )
        
        with col_input2:
            # Example selector for caption
            caption_example = st.selectbox(
                "Load Example (Caption)",
                ["None", "PaperBanana Framework"],
                key="caption_example_selector"
            )
            
            # Set value based on example selection or session state
            if caption_example == "PaperBanana Framework":
                caption_value = example_caption
            else:
                caption_value = st.session_state.get("caption", "")
            
            caption = st.text_area(
                "Figure Caption / Visual Intent (Markdown recommended)",
                value=caption_value,
                height=250,
                placeholder="Enter the figure caption...",
                help="The caption or description of the figure to generate. Markdown format is recommended."
            )
        
        # Process button
        if st.button("🚀 Generate Candidates", type="primary", use_container_width=True):
            if not method_content or not caption:
                st.error("Please provide both method content and caption!")
            else:
                # Save to session state
                st.session_state["method_content"] = method_content
                st.session_state["caption"] = caption
                
                with st.spinner(f"Generating {num_candidates} candidates in parallel... This may take a few minutes."):
                    # Create input data list
                    input_data_list = create_sample_inputs(
                        method_content=method_content,
                        caption=caption,
                        aspect_ratio=aspect_ratio,
                        figure_size=figure_size,
                        num_copies=num_candidates,
                        max_critic_rounds=max_critic_rounds,
                        task_name=task_name,
                    )
                    
                    # Process in parallel
                    try:
                        results = asyncio.run(process_parallel_candidates(
                            input_data_list,
                            exp_mode=exp_mode,
                            retrieval_setting=retrieval_setting,
                            main_model_name=main_model_name,
                            image_gen_model_name=image_gen_model_name,
                            task_name=task_name,
                        ))
                        st.session_state["results"] = results
                        st.session_state["exp_mode"] = exp_mode
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state["timestamp"] = timestamp_str
                        
                        counts = summarize_results(results, exp_mode)
                        summary = f"{counts['success']} succeeded, {counts['failed']} failed, {counts['incomplete']} incomplete candidates."
                        st.info(summary)
                        st.session_state.pop("json_file", None)
                        if save_results == "Yes":
                            try:
                                json_filename = save_results_json(results, Path(__file__).parent / "results" / "demo")
                                st.session_state["json_file"] = str(json_filename)
                                st.info(f"Results saved to: {json_filename.name}")
                            except OSError:
                                st.warning("Results are available, but saving JSON failed.")
                    except Exception as e:
                        st.error(f"Processing failed ({type(e).__name__}); check the selected models and credentials.")
        
        # Display results
        if "results" in st.session_state and st.session_state["results"]:
            results = st.session_state["results"]
            current_mode = st.session_state.get("exp_mode", exp_mode)
            timestamp = st.session_state.get("timestamp", "N/A")
            
            st.divider()
            st.markdown("## 🎨 Generated Candidates")
            st.caption(f"Generated at: {timestamp} | Pipeline: {mode_info.get(current_mode, current_mode)}")
            
            # Show JSON file download if available
            if "json_file" in st.session_state:
                json_file_path = Path(st.session_state["json_file"])
                if json_file_path.exists():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info(f"📄 Results saved to: `{json_file_path.name}`")
                    with col2:
                        with open(json_file_path, "r", encoding="utf-8") as f:
                            json_data = f.read()
                        st.download_button(
                            label="⬇️ Download JSON",
                            data=json_data,
                            file_name=json_file_path.name,
                            mime="application/json",
                            use_container_width=True
                        )
            
            # Display results in a grid (3 columns)
            num_cols = 3
            num_results = len(results)
            
            for row_start in range(0, num_results, num_cols):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    result_idx = row_start + col_idx
                    if result_idx < num_results:
                        with cols[col_idx]:
                            display_candidate_result(results[result_idx], result_idx, current_mode)
            
            # Add ZIP download button
            st.divider()
            st.markdown("### 💾 Batch Download")
            
            try:
                import zipfile
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for candidate_id, result in enumerate(results):
                        final_image_key = resolve_final_output(
                            result,
                            exp_mode=current_mode,
                        ).image_key
                        
                        if final_image_key and final_image_key in result:
                            img = final_image(result, current_mode)
                            if img:
                                img_buffer = BytesIO()
                                img.save(img_buffer, format="PNG")
                                zip_file.writestr(
                                    f"candidate_{candidate_id}.png",
                                    img_buffer.getvalue()
                                )
                
                zip_buffer.seek(0)
                st.download_button(
                    label="⬇️ Download ZIP",
                    data=zip_buffer.getvalue(),
                    file_name=f"paperbanana_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
                st.success("ZIP file ready for download!")
            except Exception as e:
                st.error(f"Failed to create ZIP: {e}")
    
    # ==================== TAB 2: Refine Image ====================
    with tab2:
        st.markdown("### Refine and upscale your diagram to high resolution (2K/4K)")
        st.caption("Choose provider-supported resolution and ratio. Legacy OpenAI image models use 1K with 1:1, 3:2, or 2:3.")
        
        # Sidebar for refinement settings
        with st.sidebar:
            st.title("✨ Refinement Settings")
            
            refine_resolution = st.selectbox(
                "Target Resolution",
                ["1K", "2K", "4K"],
                index=1,
                key="refine_resolution",
                help="Higher resolution takes longer but produces better quality"
            )
            
            refine_aspect_ratio = st.selectbox(
                "Aspect Ratio",
                ["21:9", "16:9", "3:2", "1:1", "2:3"],
                index=0,
                key="refine_aspect_ratio",
                help="Aspect ratio for the refined image"
            )
        
            refine_model = st.selectbox(
                "Image Editing Model", list(dict.fromkeys([default_image_model, *list_model_options("image_edit")])),
                key="refine_model",
            )

        st.divider()
        
        # Upload section
        st.markdown("## 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg"],
            help="Upload the diagram you want to refine"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            uploaded_image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Original Image")
                st.image(uploaded_image, use_container_width=True)
            
            with col2:
                st.markdown("### Edit Instructions")
                edit_prompt = st.text_area(
                    "Describe the changes you want",
                    height=200,
                    placeholder="E.g., 'Change the color scheme to match academic paper style' or 'Make the text larger and bolder' or 'Keep everything the same but output in higher resolution'",
                    help="Describe what you want to change or use 'Keep everything the same' for just upscaling",
                    key="edit_prompt"
                )
                
                if st.button("✨ Refine Image", type="primary", use_container_width=True):
                    if not edit_prompt:
                        st.error("Please provide edit instructions!")
                    else:
                        with st.spinner(f"Refining image to {refine_resolution} resolution... This may take a minute."):
                            try:
                                image_bytes = uploaded_file.getvalue()

                                # Call nanoviz API
                                refined_bytes, message = asyncio.run(
                                    refine_image_with_nanoviz(
                                        image_bytes=image_bytes,
                                        edit_prompt=edit_prompt,
                                        aspect_ratio=refine_aspect_ratio,
                                        image_size=refine_resolution,
                                        model_name=refine_model,
                                    )
                                )
                                
                                if refined_bytes:
                                    st.session_state["refined_image"] = refined_bytes
                                    st.session_state["refine_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                            except Exception as e:
                                st.error(f"Refinement failed ({type(e).__name__}); check the selected model and options.")
            
            # Display refined result if available
            if "refined_image" in st.session_state:
                st.divider()
                st.markdown("## 🎨 Refined Result")
                st.caption(f"Generated at: {st.session_state.get('refine_timestamp', 'N/A')} | Resolution: {refine_resolution}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Before")
                    st.image(uploaded_image, use_container_width=True)
                
                with col2:
                    st.markdown(f"### After ({refine_resolution})")
                    refined_image = Image.open(BytesIO(st.session_state["refined_image"]))
                    st.image(refined_image, use_container_width=True)
                    
                    # Download button
                    st.download_button(
                        label=f"⬇️ Download {refine_resolution} Image",
                        data=pil_to_png_bytes(refined_image),
                        file_name=f"refined_{refine_resolution}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()
