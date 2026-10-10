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
Streamlit Visualizer for Pipeline Evolution
Shows the progression of diagrams through Planner → Stylist → Critic stages
"""

import streamlit as st
import json
import base64
from io import BytesIO
from PIL import Image
import os
import sys
from pathlib import Path

# Ensure local imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from utils.legacy_ui_results import (
    infer_task_name, build_evolution_stages, resolve_final_output,
    resolve_display_mode_output, resolve_gt_image_path,
)
from utils.legacy_ui_refinement import final_image, summarize_results

st.set_page_config(layout="wide", page_title="PaperVizAgent Pipeline Evolution", page_icon="🍌")



@st.cache_data
def load_data(path):
    """Read JSON or JSONL data."""
    data = []
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        # Try to load as JSON array first
        if content.startswith("["):
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        
        # If that fails, try JSONL format
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
                
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return []
    return data

def base64_to_image(b64_str):
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        image_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data))
    except Exception:
        return None

def detect_task_type(item):
    return infer_task_name(item)



def display_stage_comparison(item, results_file_path=None):
    """Display 2x2 grid comparison: Ground Truth + three pipeline stages."""
    st.markdown("### 📊 Pipeline Evolution Comparison")
    
    task_type = detect_task_type(item)
    prefix = "target_plot" if task_type == "plot" else "target_diagram"
    
    available_stages = [{"title": "🎯 Human (Ground Truth)", "is_human": True}]
    for stage in build_evolution_stages(item):
        available_stages.append({**stage, "title": stage["name"],
                                 "img_key": stage["image_key"], "is_human": False})
    if item.get("status") == "failed":
        st.warning("Generation failed. Any available intermediate images are retained for inspection.")
    num_stages = len(available_stages)
    cols_per_row = 2
    stages = available_stages

    # Display stages in a grid
    for row_start in range(0, num_stages, cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            stage_idx = row_start + col_idx
            if stage_idx >= num_stages:
                break
            
            stage = stages[stage_idx]
            with cols[col_idx]:
                st.markdown(f"**{stage['title']}**")
                
                # Display image
                if stage["is_human"]:
                    # Handle Human (Ground Truth) image
                    human_path = resolve_gt_image_path(item, task_name=task_type,
                                                        results_file_path=results_file_path, repo_root=REPO_ROOT)
                    if human_path and os.path.exists(human_path):
                        try:
                            img = Image.open(human_path)
                            st.image(img, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to load Human image: {e}")
                    else:
                        st.info("No Human image available")
                    
                    # Show caption instead of description
                    caption = item.get("visual_intent") or item.get("caption") or item.get("brief_desc", "No caption available")
                    with st.expander("View Caption", expanded=False):
                        st.write(caption)
                else:
                    # Handle pipeline stage images
                    img_b64 = item.get(stage["img_key"])
                    if img_b64:
                        img = base64_to_image(img_b64)
                        if img:
                            st.image(img, use_container_width=True)
                        else:
                            st.error("Failed to decode image")
                    else:
                        st.info("No image available")
                    
                    # Display description in expander
                    desc = item.get(stage.get("desc_key", ""), "No description available")
                    with st.expander("View Description", expanded=False):
                        if task_type == "plot" and desc:
                             # Try to format as code if it looks like code, or just text
                             st.code(desc, language="python") # Plots are usually python code
                        else:
                             st.write(desc)
                    
                    # Display critic suggestions if this is a critic stage
                    if "suggestions_key" in stage:
                        suggestions = item.get(stage["suggestions_key"], "")
                        if isinstance(suggestions, str) and suggestions.strip() != "No changes needed.":
                            with st.expander("💬 Critic Suggestions", expanded=False):
                                st.write(suggestions)

def display_critique(item):
    """Display the critique if available."""
    if "critique0" in item and item["critique0"]:
        st.markdown("### 💬 Critic's Feedback")
        with st.expander("View Critique", expanded=False):
            st.write(item["critique0"])

def display_evaluation_results(item):
    """Display evaluation results if available."""
    dimensions = ["Faithfulness", "Conciseness", "Readability", "Aesthetics", "Overall"]
    
    has_eval = any(f"{dim.lower()}_outcome" in item for dim in dimensions)
    
    if has_eval:
        st.markdown("### 📈 Evaluation Results")
        cols = st.columns(len(dimensions))
        
        for i, dim in enumerate(dimensions):
            outcome_key = f"{dim.lower()}_outcome"
            reasoning_key = f"{dim.lower()}_reasoning"
            outcome = item.get(outcome_key, "N/A")
            reasoning = item.get(reasoning_key, "N/A")
            
            with cols[i]:
                st.markdown(f"**{dim}**")
                if outcome == "Model":
                    st.success(outcome)
                elif outcome == "Human":
                    st.info(outcome)
                elif outcome == "Tie":
                    st.warning(outcome)
                else:
                    st.text(outcome)
                
                with st.expander("View Reasoning", expanded=False):
                    st.write(reasoning)

def main():
    st.sidebar.title("🍌 Pipeline Evolution Viewer")
    file_path = st.sidebar.text_input("Results JSONL Path", placeholder="Enter path to results file...")
    
    if st.sidebar.button("🔄 Refresh Data"):
        load_data.clear()
        st.rerun()
    
    if not file_path:
        st.info("👆 Please enter a file path to begin")
        st.stop()
    
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        st.stop()
    
    data = load_data(file_path)
    
    # --- Search Functionality ---
    search_query = st.sidebar.text_input("🔍 Search ID", value="", help="Filter by ID (case-insensitive)")
    if search_query:
        data = [item for item in data if search_query.lower() in str(item.get("id", "")).lower()]
        st.sidebar.caption(f"Found {len(data)} matching cases")
    
    total_items = len(data)
    
    if total_items == 0:
        if search_query:
            st.warning(f"No samples found matching '{search_query}'.")
        else:
            st.warning("Data is empty or format is incorrect.")
        return
    
    st.title("🍌 PaperVizAgent Pipeline Evolution Viewer")
    st.markdown(f"Visualizing the progression through **Planner → Stylist → Critic** stages")
    
    st.divider()
    
    # --- Global Statistics ---
    with st.expander("📊 Global Statistics", expanded=False):
        total = len(data)
        
        counts = summarize_results(data)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", total)
        col2.metric("Succeeded", counts["success"])
        col3.metric("Failed / Incomplete", counts["failed"] + counts["incomplete"])

    st.divider()
    
    # --- Pagination ---
    PAGE_SIZE = 10  # Changed from 5 to 10
    if "page" not in st.session_state:
        st.session_state.page = 0
    
    total_pages = max((total_items + PAGE_SIZE - 1) // PAGE_SIZE, 1)
    st.session_state.page = min(st.session_state.page, total_pages - 1)
    
    # Navigation buttons
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_left:
        if st.button("⬅️ Previous Page", disabled=(st.session_state.page == 0)):
            st.session_state.page -= 1
            st.rerun()
    
    with col_center:
        page_input = st.number_input(
            "Page", 
            min_value=1, 
            max_value=total_pages, 
            value=st.session_state.page + 1,
            label_visibility="collapsed"
        )
        if page_input != st.session_state.page + 1:
            st.session_state.page = page_input - 1
            st.rerun()
        st.caption(f"Page {st.session_state.page + 1} of {total_pages}")
    
    with col_right:
        if st.button("Next Page ➡️", disabled=(st.session_state.page >= total_pages - 1)):
            st.session_state.page += 1
            st.rerun()
    
    start_idx = st.session_state.page * PAGE_SIZE
    end_idx = min(start_idx + PAGE_SIZE, total_items)
    batch = data[start_idx:end_idx]
    
    st.markdown(f"**Displaying {start_idx + 1} - {end_idx} of {total_items}**")
    
    # --- Display Samples ---
    for i, item in enumerate(batch):
        idx = start_idx + i
        anno = item  # Flattened structure
        
        with st.container(border=True):
            # Header
            st.subheader(f"#{idx + 1}: {item.get('visual_intent', 'N/A')}")
            st.caption(f"ID: `{item.get('id', 'Unknown')}`")
            
            # Method/Data section
            task_type = detect_task_type(item)
            label = "📚 Raw Data" if task_type == "plot" else "📚 Method Section"
            
            with st.expander(label, expanded=False):
                if task_type == "plot":
                    st.code(json.dumps(item.get('content', {}), indent=2), language="json")
                else:
                    method_content = item.get('content', 'N/A')
                    st.markdown(method_content)
            
            # Pipeline comparison
            display_stage_comparison(item, results_file_path=file_path)
            
            # Critique
            display_critique(item)
            
            # Evaluation results
            display_evaluation_results(item)
            
            st.divider()

if __name__ == "__main__":
    main()
