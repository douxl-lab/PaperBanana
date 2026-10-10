# PaperBanana RFS upgrade: baseline audit and implementation gates

## Baseline and scope

- Repository: https://github.com/douxl-lab/PaperBanana
- Audited branch: `main`, commit `836455537e863b5a2f40dace487a782c0bc5ef94` (2026-06-25).
- Review date: 2026-10-10 UTC.
- The full Git tree contains 80 tracked files. `BASELINE_INVENTORY.json` records every baseline file, size, SHA-256, and Python function boundaries.
- No repository-level `AGENTS.md`, `.agents/skills`, build-system metadata, or `.github/workflows` exists at this baseline.
- Review covered application code, prompts and style guides, CLI and configuration, five-agent orchestration, supplementary agents, benchmark evaluation, viewers, notebook code cells, scripts, tests, dependency/runtime declarations, and documentation. Static web/vendor/image files were inventoried and checked for their role; their illustration pixels are not a runtime acceptance test.
- This is a static review. The implementation coordinator separately reported that the unchanged baseline passed all 37 pytest tests after installing the required SOCKS extra (`httpx[socks]`), in 8.57 seconds. This report does not claim that live model requests, browser interactions, Colab, Docker, or editable-PPTX reconstruction have passed.
- Preserve the five core agents: Retriever, Planner, Stylist, Visualizer, Critic. Keep Vanilla/Polish, diagram/plot, UI, benchmark, metaphor option, and legacy result keys available. Add RFS capabilities around these interfaces rather than replacing the agent system.

## Most consequential findings

| ID | Priority | Evidence at baseline | Consequence and required change |
|---|---|---|---|
| F01 | P0 | `skill/run.py:85-112` accepts `args.task` but does not pass `task_name` into `ExpConfig` | CLI `--task plot` runs the diagram path. Pass the task and normalize plot input through the existing helper. |
| F02 | P0 | `skill/run.py:64-82`; `agents/visualizer_agent.py:89-103` | CLI hardcodes diagram and critic rounds 3→0; Visualizer only renders rounds 0..2 although both UIs allow five rounds. Use `eval_image_field` and dynamic legacy output selection; discover arbitrary critic round keys. |
| F03 | P0 | `skill/run.py:85-87` before input validation | A no-retrieval run still downloads PaperBananaBench. Validate arguments/configuration first; download only when the selected retrieval path needs it, with an explicit disable/prepare control and useful failure handling. |
| F04 | P0 | `utils/paperviz_processor.py:193-203` | Retrieval for the first record is copied to every record in a heterogeneous benchmark batch. Deduplicate only identical source/intent/task/settings groups and copy mutable lists independently. |
| F05 | P0 | `utils/paperviz_processor.py:212-223`; generation retry functions return `['Error']` | One raised candidate failure aborts the whole batch; string error sentinels can become design text or malformed image data. Make per-candidate terminal status/error explicit, isolate failures, and keep cancellation visible. |
| F06 | P0 | `agents/critic_agent.py:129-147` | Missing/malformed JSON defaults to “No changes needed.”; non-string revised descriptions cause `.strip()` failures. Fail closed on review validity, validate typed fields, retain the last valid image, and record an incomplete review instead of declaring success. |
| F07 | P0 | `utils/generation_utils.py:798-804` vs `288-299` | The unified router supplies `max_completion_tokens`, while the Claude adapter requires `max_output_tokens`; routed Claude calls raise `KeyError`. Preserve and test the existing non-Gemini paths. |
| F08 | P0 | `utils/paperviz_processor.py:249-257`; `utils/eval_toolkits.py:120-223,228-314` | The processor never passes the configured evaluation model; the default empty name falls into direct Claude dispatch. Evaluation also bypasses OpenRouter/registry and does not consistently validate winner values. Route through the unified interface, pass the actual configured model, distinguish failed/skipped evaluation, and never present it as a scientific pass. |
| F09 | P0 | notebook cell 14, lines 614-622 | `ExpConfig(model_name=..., image_model_name=...)` uses unsupported keyword names and fails before generation. Cell 5 writes those same obsolete defaults keys. Use the actual configuration names and cover notebook compatibility statically and in runtime-independent tests. |
| F10 | P0 | `visualize/show_referenced_eval.py:160-165` | Hot-reload imports nonexistent `prompts.plots_eval_prompts`. Correct to `plot_eval_prompts`; use the final-output helper and a real configured evaluation model/work directory. |
| F11 | P1 | `utils/generation_utils.py:41-113,748-819`; Visualizer/Vanilla/Polish provider branches | Provider is often chosen by whichever key exists, with different rules across text, images, editing, and evaluation. An OpenRouter key can override a direct Gemini model. Add explicit provider/model/capability resolution with compatible legacy aliases, and clear unsupported-role errors. Do not guess capabilities from substrings. |
| F12 | P1 | `utils/generation_utils.py:145-228,372-514` | Retry loops treat permanent/auth/config errors like transient outages; some return an error sentinel and discard useful cause/status. Mutable Gemini candidate-count configuration is changed in place. Use bounded retries, timeout/backoff/cancellation, retriable-error classification, and immutable per-request options. |
| F13 | P1 | `agents/visualizer_agent.py:108-189`; `utils/image_utils.py:24-46` | Every generated image is converted to lossy JPEG, then the CLI exports it as PNG. Preserve the original bytes/MIME/alpha for RFS delivery while retaining JPEG compatibility keys for old UIs and critics. |
| F14 | P1 | `agents/visualizer_agent.py:125-139`; `agents/vanilla_agent.py:93-110` | OpenAI always uses `1536x1024` and ignores requested ratio; there is no image-edit route. Resolve provider-supported dimensions/options explicitly, document any approximation, and support reference-image edits where the model/API actually supports them. |
| F15 | P1 | `demo.py:219-224` vs `_convert_to_openai_format` | Streamlit refinement sends an image representation the converter does not recognize, so the image is silently dropped. Reject unsupported content shapes or normalize them; unify both UI editing paths. |
| F16 | P1 | `app.py:167-235,778-805`; `demo.py:191-293,784-805` | Two duplicated editing implementations bypass registry and OpenAI edits; inputs are forced to JPEG and RGBA uploads can fail. Preserve/normalize MIME and explicitly control provider fallbacks. Close temporary clients and loops. |
| F17 | P1 | `utils/config.py:46-84` | Configuration mutates process-wide `TZ` to America/Los_Angeles; YAML model settings precede env overrides, unlike key precedence. No numeric/routing validation; creates output directories in construction. Preserve user timezone; document and test CLI→environment→config→default precedence, aliases, explicit config path, and validated runtime limits. |
| F18 | P1 | `main.py:64-71,125-152` | Default experiment mode is `dev`, which the processor rejects; concurrency is fixed at ten and checkpoint writes are non-atomic and not resumable. Choose a supported default, expose limits, use atomic records, and resume only matching runs/candidates. |
| F19 | P1 | `agents/visualizer_agent.py:40,69-71`; `agents/vanilla_agent.py:41,59-61`; `utils/plot_execution.py:29-64` | Two plot agents allocate separate 32-worker pools even when one agent is unused; cleanup relies on destructor. Plot code runs arbitrary Python with no timeout. A ProcessPool is not a security sandbox. Use explicit lifecycle, bounded workers and execution timeout; document trusted-code limits. Do not claim safety from an AST blacklist alone. |
| F20 | P1 | `agents/planner_agent.py:95-98`; `agents/retriever_agent.py:155-199`; style-guide loader | Missing/bad reference images interrupt a run, returned IDs are only weakly validated, large reference prompts have no budget, and guide file loading is fixed to NeurIPS 2025. Validate/filter IDs, cache metadata, preserve provenance, report omitted references, and make guides selectable without changing scientific content. |
| F21 | P1 | `agents/polish_agent.py:133-141,165-203` | “No changes needed” executes `pass` and still generates another image; empty suggestion failures also generate. Return original/last valid image with explicit status; use the shared edit adapter and correct system prompt. |
| F22 | P1 | `visualize/show_pipeline_evolution.py:81-86,132-147,172-181`; referenced viewer | Viewers still use old task heuristics, three fixed critic rounds, stale stage keys, and cwd-relative GT paths. Reuse `legacy_ui_results` task/final/stage/path helpers and tolerate explicit failed records. |
| F23 | P2 | `app.py:497-524,649-737`; `demo.py:620-661` | Gradio global env keys are process-global, not isolated browser-session credentials; success counts count returned records instead of valid outputs. JSON is saved even when “Save Results = No”; timestamp-only file names can collide. Document single-user deployment, avoid credential leakage in exports/logs, provide success/failed counts, and use unique atomic run paths. |
| F24 | P2 | `style_guides/generate_category_style_guide.py:29-44,177-299`; notebook cells 12/14/15 | Style generator has an import-time Gemini client, fixed old model, cwd-relative outputs; notebook duplicates pricing, model choices, selection logic, and fixed stage display. Reuse shared configuration/adapters where practical; label price/coverage estimates as estimates and keep visual/scientific validation separate. |
| F25 | P2 | `skill/SKILL.md`, `skill/clawhub.json`, root `LICENSE`, `requirements.txt` | Skill package metadata says MIT-0 despite Apache-2.0 headers/root; runtime SDK dependencies are unbounded and test tools are not declared. Align metadata to evidenced licensing, document development dependencies and supported Python/SDK ranges, preserve upstream notices. |

## RFS contract requirements and boundaries

The known design calls for a lightweight orchestrator using PaperBanana as the generation engine, academic-plotting for real-data plotting, paper-fig as the main editable-PPTX reconstruction backend, and Sivia as an optional native drawing/review backend. This upgrade should not vendor the other three projects or recreate their agent systems.

The input scientific contract must retain source references, exact labels, nodes, directed typed connections, formulas, quantitative provenance, approved/locked facts, layout/style constraints, and verification status. Unknown facts stay pending or unverified. Generated image appearance cannot silently replace that contract.

The exchange result must provide all of: original image artifact, actual generation prompt, detailed human-readable design/reconstruction instructions, original scientific FigureSpec, actual provider/model/options, candidate/stage identities, review/selection evidence, and failure/verification status. Editable reconstruction uses image + FigureSpec + textual instructions together; raster image delivery alone is insufficient.

Use a versioned, documented exchange schema and backward-compatible optional fields. The earlier RFS package/consumer source has not been retrieved, so this audit does not establish exact schema-level interoperability with an external RFS installation. Implement round-trip tests and a mapping contract; report external-consumer verification separately.

Prompt Compiler must separate initial design, candidate comparison, and local refinement. Preserve exact scientific topology and terminology, include layout/visual/text/negative constraints, and adapt to provider capabilities rather than model-name guesses. Local edits specify allowed edits and protected regions. Quantitative figures remain driven by real data and reproducible code. Scientific review cannot be replaced by a keyword-coverage or aesthetic score.

## Layered implementation plan

1. Baseline/regression: record the complete inventory, run all current tests under pytest, compile Python, and identify blocked live/runtime stages. Do not remove existing tests or APIs to obtain a pass.
2. Provider/config foundation: shared capability registry, explicit resolution, current documented API adapters, compatible wrapper functions, consistent options/error/retry/timeout handling, preserved secrets boundary.
3. Runtime/legacy repairs: exact task propagation, arbitrary critic rounds, fail-closed critique, grouped retrieval, per-candidate isolation, cancellation/cleanup, no-JPEG RFS output, evaluation/viewer/notebook correctness.
4. RFS additions: versioned FigureSpec and result schema, deterministic Prompt Compiler and model adapters, candidate comparison/local-edit/reconstruction templates, artifact exporter and provenance/review records. Keep all five agent stages; pass RFS context as optional fields.
5. CLI/UI/documentation integration: optional RFS input/output modes, config path, download policy, concurrency, checkpoint/resume, atomic files, truthful success/failure messages, shared refinement helper. Legacy commands and result readers remain usable.
6. Validation/publication gate: run final aggregate checks and mock integration across providers/tasks/modes plus rendered artifact checks. Live model tests, user-repository push, exact remote commit presence, and any CI status are separate verification stages. Publish to `main` only after required checks pass; do not create a branch or PR for this request. If any required gate fails, do not push.

## Required test matrix

- Existing five test modules: retain every legacy behavior, including diagram/plot result keys, requested image-size mapping, plotting code retention, and diagram-only metaphor semantics. `test_planner_metaphor.py` uses pytest functions and `tmp_path`, so unittest discovery alone is insufficient.
- Configuration: precedence/aliases, invalid YAML/types/numeric limits, custom models, explicit providers, no credentials, no timezone mutation, no secret values in artifacts/logs.
- Provider mocks: Gemini text/image/edit, OpenAI text/image/edit, OpenRouter text/image, Anthropic legacy routing, multimodal content normalization, unsupported role/options, empty/malformed/blocked responses, 400/401/403/404 fail-fast, 429/5xx retry, timeout/cancellation.
- Pipeline: all experiment modes, diagram + plot, critic rounds 0/1/3/5, missing/invalid critique, text-only missing image, last-valid-image fallback, partial batch failures, heterogeneous-source retrieval isolation, duplicate-candidate retrieval reuse, explicit executor close.
- RFS: schema validation/roundtrip/version rejection, exact labels/topology/provenance retained, no invented quantitative values, compiler sections and provider adapters, local edit protected constraints, comparison prompt, lossless image/MIME delivery, four-part reconstruction handoff, empty/failure manifests.
- CLI: help/preflight without key/network, plot task path, invalid arguments before downloads, retrieval none/offline/local cache, candidate identity/output names, all-output failure nonzero, partial failure semantics, unique atomic saves, checkpoint hash/settings validation, safe resume without rerunning valid candidates.
- UI/readers/notebook: mock shared refinement with PNG/RGBA/JPEG, original image actually transmitted, current model settings used, arbitrary rounds, success versus failure counts, GT relative path, hot reload correct imports, notebook correct ExpConfig/default names and every code cell syntactically checked without executing credential/network cells.
- Plot execution: benign plots, code preservation, no figure, exception, hang timeout, cleanup, lifecycle, documented trust boundary. No test should execute hostile model-generated code with real credentials.
- Artifact validation: images decode and expected format/dimensions/alpha remain, sidecars parse, original FigureSpec and actual prompt match manifest, no claim of native PPTX editability without a separate editable reconstruction test.

## File coverage and disposition

The complete machine-readable file inventory accompanies this report. Runtime Python modules and the notebook are in the repair/regression scope. Prompts/style guides are in the scientific-prompt preservation and optional-style scope. Existing tests are retained and extended. Root docs, skill metadata, requirements and launch scripts are updated where functionality changes. LICENSE, contributor credits and conduct material are preserved. Marketing HTML, vendor CSS/JS, SVG badges, logo and sample illustration assets do not need functional changes for this upgrade; they remain tracked and are not silently deleted.

## Open verification items

- Current model IDs and parameter support must be grounded in official provider documentation with a verification date; this source-code audit does not declare any model to be the latest.
- No external RFS consumer schema/source is available in this checkout. Exact external RFS compatibility needs a separate consumer fixture or integration run.
- Live API availability and visual scientific correctness require real credentials and approved source material. Mock success must not be described as a completed live figure generation.
- Browser UX, Colab execution, Docker launch, resource/security isolation and actual native editable reconstruction require their own appropriate runtime checks.
- Public GitHub metadata reports push permission and an unprotected main; no mutation has been performed in this audit, and successful publication must be confirmed from the remote after tests.

## Complete tracked-file checklist

| Baseline file | Coverage and disposition |
|---|---|
| `.all-contributorsrc` | Reviewed documentation/legal/credit role; retain licenses/credits; update only usage/contribution policy relevant to this fork. |
| `.gitattributes` | Reviewed runtime/dependency/packaging declarations; update only required supported behavior. |
| `.gitignore` | Reviewed runtime/dependency/packaging declarations; update only required supported behavior. |
| `CONTRIBUTING.md` | Reviewed documentation/legal/credit role; retain licenses/credits; update only usage/contribution policy relevant to this fork. |
| `Dockerfile` | Reviewed runtime/dependency/packaging declarations; update only required supported behavior. |
| `LICENSE` | Reviewed documentation/legal/credit role; retain licenses/credits; update only usage/contribution policy relevant to this fork. |
| `README.md` | Reviewed documentation/legal/credit role; retain licenses/credits; update only usage/contribution policy relevant to this fork. |
| `agents/__init__.py` | Reviewed agent/runtime/prompt code; preserve core and optional agents; repair and extend with regression tests. |
| `agents/base_agent.py` | Reviewed agent/runtime/prompt code; preserve core and optional agents; repair and extend with regression tests. |
| `agents/critic_agent.py` | Reviewed agent/runtime/prompt code; preserve core and optional agents; repair and extend with regression tests. |
| `agents/planner_agent.py` | Reviewed agent/runtime/prompt code; preserve core and optional agents; repair and extend with regression tests. |
| `agents/polish_agent.py` | Reviewed agent/runtime/prompt code; preserve core and optional agents; repair and extend with regression tests. |
| `agents/retriever_agent.py` | Reviewed agent/runtime/prompt code; preserve core and optional agents; repair and extend with regression tests. |
| `agents/stylist_agent.py` | Reviewed agent/runtime/prompt code; preserve core and optional agents; repair and extend with regression tests. |
| `agents/vanilla_agent.py` | Reviewed agent/runtime/prompt code; preserve core and optional agents; repair and extend with regression tests. |
| `agents/visualizer_agent.py` | Reviewed agent/runtime/prompt code; preserve core and optional agents; repair and extend with regression tests. |
| `app.py` | Reviewed full application entry points and handlers; repair shared interfaces and truthful status. |
| `assets/dataset-on-hf-xl.svg` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `assets/logo.jpg` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `assets/method_diagram.png` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `assets/paper-page-xl.svg` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `assets/spaces-hf-wordmark.png` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `assets/spaces-on-hf-xl.png` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `assets/teaser_figure.jpg` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `code-of-conduct.md` | Reviewed documentation/legal/credit role; retain licenses/credits; update only usage/contribution policy relevant to this fork. |
| `configs/model_config.template.yaml` | Reviewed model configuration template; add documented providers/capabilities/runtime options. |
| `demo.py` | Reviewed full application entry points and handlers; repair shared interfaces and truthful status. |
| `index.html` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `main.py` | Reviewed full application entry points and handlers; repair shared interfaces and truthful status. |
| `notebooks/PaperBanana_Colab_End_to_End.ipynb` | Reviewed all 17 notebook cells and code interfaces; repair config keys and shared model/stage behavior; Colab live execution pending. |
| `prompts/__init__.py` | Reviewed evaluation prompt definitions; preserve scientific criteria; validate structured results. |
| `prompts/diagram_eval_prompts.py` | Reviewed evaluation prompt definitions; preserve scientific criteria; validate structured results. |
| `prompts/plot_eval_prompts.py` | Reviewed evaluation prompt definitions; preserve scientific criteria; validate structured results. |
| `requirements.txt` | Reviewed runtime/dependency/packaging declarations; update only required supported behavior. |
| `scripts/run_demo.sh` | Reviewed launch scripts; correct cwd/arguments/download assumptions where needed. |
| `scripts/run_main.sh` | Reviewed launch scripts; correct cwd/arguments/download assumptions where needed. |
| `skill/SKILL.md` | Reviewed skill contract, CLI and licensing metadata; expand compatible RFS modes and output contract. |
| `skill/clawhub.json` | Reviewed skill contract, CLI and licensing metadata; expand compatible RFS modes and output contract. |
| `skill/run.py` | Reviewed skill contract, CLI and licensing metadata; expand compatible RFS modes and output contract. |
| `static/css/bulma-carousel.min.css` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/css/bulma-slider.min.css` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/css/bulma.min.css` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/css/fontawesome.all.min.css` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/css/index.css` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/images/case_diagram.png` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/images/case_failure.png` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/images/case_plot_img_vs_code.png` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/images/case_style_polish.png` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/images/discussion_enhance_style.jpg` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/images/logo.jpg` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/images/main_results.png` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/images/method_diagram.png` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/images/plot_bench_stat.jpg` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/images/plot_code_vs_img.jpg` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/images/plot_vanilla_vs_ours_v2.jpg` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/images/teaser_figure.jpg` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/js/bulma-carousel.js` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/js/bulma-carousel.min.js` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/js/bulma-slider.js` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/js/bulma-slider.min.js` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `static/js/fontawesome.all.min.js` | Inventoried static site, vendor or illustration asset; no runtime refactor required; retain unchanged. |
| `style_guides/generate_category_style_guide.py` | Reviewed style content and analysis generator; preserve optional historic guides; add selectable RFS style contract. |
| `style_guides/neurips2025_diagram_style_guide.md` | Reviewed style content and analysis generator; preserve optional historic guides; add selectable RFS style contract. |
| `style_guides/neurips2025_plot_style_guide.md` | Reviewed style content and analysis generator; preserve optional historic guides; add selectable RFS style contract. |
| `tests/test_legacy_generation_options.py` | Reviewed existing regression coverage; retain and extend; baseline pytest pass separately reported. |
| `tests/test_legacy_plot_agents.py` | Reviewed existing regression coverage; retain and extend; baseline pytest pass separately reported. |
| `tests/test_legacy_ui_result_keys.py` | Reviewed existing regression coverage; retain and extend; baseline pytest pass separately reported. |
| `tests/test_planner_metaphor.py` | Reviewed existing regression coverage; retain and extend; baseline pytest pass separately reported. |
| `tests/test_plot_execution.py` | Reviewed existing regression coverage; retain and extend; baseline pytest pass separately reported. |
| `utils/__init__.py` | Reviewed runtime/config/provider/evaluation/result helpers; repair, reuse, and test compatibility. |
| `utils/config.py` | Reviewed runtime/config/provider/evaluation/result helpers; repair, reuse, and test compatibility. |
| `utils/eval_toolkits.py` | Reviewed runtime/config/provider/evaluation/result helpers; repair, reuse, and test compatibility. |
| `utils/generation_utils.py` | Reviewed runtime/config/provider/evaluation/result helpers; repair, reuse, and test compatibility. |
| `utils/image_utils.py` | Reviewed runtime/config/provider/evaluation/result helpers; repair, reuse, and test compatibility. |
| `utils/legacy_generation_options.py` | Reviewed runtime/config/provider/evaluation/result helpers; repair, reuse, and test compatibility. |
| `utils/legacy_ui_results.py` | Reviewed runtime/config/provider/evaluation/result helpers; repair, reuse, and test compatibility. |
| `utils/paperviz_processor.py` | Reviewed runtime/config/provider/evaluation/result helpers; repair, reuse, and test compatibility. |
| `utils/plot_execution.py` | Reviewed runtime/config/provider/evaluation/result helpers; repair, reuse, and test compatibility. |
| `visualize/show_pipeline_evolution.py` | Reviewed viewer loading/display/evaluation flows; migrate to shared result/path helpers. |
| `visualize/show_referenced_eval.py` | Reviewed viewer loading/display/evaluation flows; migrate to shared result/path helpers. |
