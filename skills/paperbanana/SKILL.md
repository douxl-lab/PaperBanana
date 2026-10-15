---
name: paperbanana
description: >-
  Generate scientific paper method, mechanism or architecture diagrams with PaperBanana's Retriever/Planner/Stylist/Visualizer/Critic engine. Use when the user explicitly asks for PaperBanana or a research-figure orchestrator delegates image generation to this installed engine. Preserve exact FigureSpec v1 science, candidate images, submitted prompts and written design for RFS handoff. Do not use as the general scientific-figure router, a real-data plotter, an independent TikZ compiler, a generic presentation designer, or a native editable-PPTX reconstruction backend.
license: Apache-2.0
compatibility: Installed PaperBanana engine checkout and configured, authorized provider for live generation; Python 3.10+ with jsonschema/Pillow for offline exchange; full generation tested on Python 3.12.
metadata:
  version: "0.5.0"
  category: "scientific-figure-engine"
---

# PaperBanana — delegated scientific image generator

Use PaperBanana **only** when explicitly requested or delegated by RFS. Keep the five-agent Retriever → Planner → Stylist → Visualizer → Critic workflow; do not create a competing outer planning/critic loop. Scientific routing, real-data Matplotlib plots, local TikZ compilation, publication-quality acceptance and native editable reconstruction belong to RFS or their respective installed specialist backends.

## Resolve inputs and permissions

- Run from the separately installed engine checkout; set `PAPERBANANA_ROOT` to its directory. The installed `skills/paperbanana` folder is an invocation manifest, **not** an independent runtime. Existing entrypoints live at `skill/run.py` and `skill/exchange.py`.
- Read the scientific source or canonical FigureSpec v1. Preserve the caller's modules, labels, formulas, edge endpoints/direction, groups, evidence, data and locked constraints; only permitted visual `design` fields may vary.
- Do not send unpublished materials or invoke a paid provider without explicit authorization for this task/service and possible cost. An installed key or existing model profile does not establish permission or availability.
- Read [exchange instructions](references/exchange.md) for request and bundle formats; if no compatible engine/model is available, report the blocker instead of pretending to have produced an image.

## Paired RFS workflow

1. Run `python "$PAPERBANANA_ROOT/skill/exchange.py" describe`; have RFS verify the schema/validator/manifest descriptor before transmission. Never soften a mismatch by dropping the contract and sending a prose summary.
2. Validate the canonical request with `validate-request --request <request.json>` locally. Reject missing sources or changed scientific hashes.
3. Only with permission, run `run --request <request.json> --allow-external`. The wrapper delegates to the five-agent pipeline and creates actual candidates; do not separately run the same generation job.
4. Preserve each candidate's actual image bytes, exact submitted prompt, design description and full FigureSpec, and record review/failure states honestly. For an approved reference, RFS owns image-anchored notes and the native reconstruction handoff.
5. Use `verify-bundle --spec <original.json> --bundle <candidate>` before making a claim about byte/source integrity. Even a valid bundle leaves scientific correctness, visual fidelity and native editability to a separate review.

~~~bash
python "$PAPERBANANA_ROOT/skill/exchange.py" describe
python "$PAPERBANANA_ROOT/skill/exchange.py" validate-request --request /project/request.json
python "$PAPERBANANA_ROOT/skill/exchange.py" run --request /project/request.json --allow-external
python "$PAPERBANANA_ROOT/skill/exchange.py" verify-bundle --spec /project/figure-spec.json --bundle /project/candidate
~~~

## Figure style and optional vector reference

Default standalone academic art direction: **no large figure title** or internal subtitle, unsolicited panel letters, decorative stage numbers or oversized banners. Prefer **three independent outputs** with separate captions when a complete overview and two mechanisms all need space; actual venue/panel requirements and scientific source outrank this default. Keep tensor dimensions, Q/K/V, uncertainty equations and branch weights only when justified by the supplied FigureSpec. See [journal-mechanism-style.md](https://github.com/douxl-lab/research-figure-skill/blob/main/research-figure/references/journal-mechanism-style.md).

PaperBanana does **not** render TeX. The offline `vector-handoff` command checks a successful candidate's full scientific fingerprint and file hashes, then copies its image, design notes and FigureSpec v1 into a fresh directory as visual reference for an explicit RFS `--engine tikz` run. This does not produce a vector drawing or guarantee pixel-exact reconstruction.

~~~bash
python "$PAPERBANANA_ROOT/skill/exchange.py" vector-handoff \
  --spec /project/figure-spec.json --bundle /project/candidate --out /project/vector-reference
~~~

Do not interpret synthetic fixtures as real scientific results, prompt-only output as generated art, or flattened PNGs as editable PowerPoint objects. No unverified multiclass Dirichlet formula may be repurposed as a multi-label uncertainty model. Keep credentials, unpublished source datasets, internal records and generated artifacts out of the public repository.
