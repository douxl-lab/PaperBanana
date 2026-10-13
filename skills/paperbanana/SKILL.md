---
name: paperbanana
description: >-
  Generate scientific method, architecture and overview diagrams with PaperBanana's five-agent engine; export full FigureSpec v1, actual image, exact prompt and written design for RFS reconstruction or TikZ reference handoff. Use when PaperBanana is explicitly requested or the research-figure orchestrator delegates generation. Do not use to recreate an approved figure as native PPTX, to invent experimental measurements, or to repeat a completed generative design stage.
license: Apache-2.0
compatibility: Installed PaperBanana checkout and configured model provider for generation; Python 3.10+ plus jsonschema and Pillow for offline exchange. Full generation is tested on Python 3.12.
metadata:
  version: "0.4.1"
  category: "scientific-figure-engine"
---

# PaperBanana engine

Preserve Retriever → Planner → Stylist → Visualizer → Critic. Do not add another outer agent loop with the same responsibilities. This engine produces scientific visual references and detailed written design, not native editable slides.

## Resolve runtime and scope

Use `PAPERBANANA_ROOT` or an explicitly supplied trusted engine checkout. The installed Skill folder is not the Python runtime. Existing command entrypoints remain `skill/run.py` and `skill/exchange.py` relative to that checkout. Read the root README and `skills/paperbanana/references/exchange.md` there for complete usage.

## Paired RFS request

1. Read actual source material and canonical FigureSpec. Never generate measurements or alter supplied topology, formulas, labels, groups, data, evidence or verification status.
2. Run `python "$PAPERBANANA_ROOT/skill/exchange.py" describe`; let the consumer compare the protocol/version/schema/validator fingerprint. Never bypass a mismatch by converting the spec to plain prose.
3. Validate the entire bound request using `validate-request --request /project/request.json`. Request preparation and validation must not call a provider.
4. Only with authorization for the current material and potential fees, call `run --request /project/request.json --allow-external`. Select explicit configured model profiles; do not guess from model names or key availability. Never switch paid providers silently. Do not log/publish credentials or unpublished sources.
5. Preserve images, exact selected-stage prompts, detailed design and full FigureSpec in the versioned manifest. A failed review is not a success; prior valid images may be retained with failure status. Use a fresh request/output directory.
6. Return the actual bundle locations and explicit status to RFS. The consumer validates its original scientific fingerprint and all artifact bytes. Do not fabricate a consumer acceptance, blind review or native-editability result.

**Default art direction for standalone academic figures:** no large figure title/subtitle, decorative stage-number boxes, (a)/(b)/(c) subfigure letters, contribution banners or poster-like surrounding whitespace *inside the image*. Keep precise module names, query/key/value operations, branch weights, tensor dimensions and evidence paths only if grounded in the caller's FigureSpec. When a detailed overall framework and two mechanism figures are required, render them as **three independent outputs** with separate specs and manuscript captions, not a single stitched poster. The target venue's explicit multipanel rules override this standalone default. Follow RFS's [journal mechanism style reference](https://github.com/douxl-lab/research-figure-skill/blob/main/research-figure/references/journal-mechanism-style.md).

## Design quality and repair

Use professional scientific prompts covering complete science, visual hierarchy, composition/regions, exact typography, connector lanes, and specific forbidden changes. Image model capabilities and current API parameters require actual provider verification. Local editing requires a real reference and supported editing operation; do not silently regenerate the whole composition. Preserve every non-target entity and inspect the actual edited output.

Detailed design text describes intended composition. It is not evidence that the generated image exactly followed the plan. After selection, RFS must inspect that specific image and write image-bound observed reconstruction notes. Only then can an available native builder perform faithful reconstruction.

## Optional RFS TikZ reference handoff

PaperBanana does not render TeX. For a successful, verified candidate, use skill/exchange.py vector-handoff with --spec, --bundle and --out to preserve the reference PNG, detailed design and immutable FigureSpec v1. RFS owns the explicit local --engine tikz vector renderer. The handoff is offline and must not be described as pixel-identical reconstruction or editable PPTX.

## Numeric data, legacy mode and evidence

The paired canonical endpoint accepts diagrams only. Quantitative/hybrid data panels are routed to deterministic source-data plotting by RFS. Standalone legacy plotting remains available but executes generated Python; subprocess/timeouts are not a sandbox. Use only trusted isolated single-user execution and the required explicit opt-in. Reference download `--offline` is not offline image generation.

Report software integrity, scientific validity, visual fidelity, actual model execution and native reconstruction separately. Protocol tests use synthetic fixtures and do not certify model image quality. Keep private internal notes and generated artifacts outside the public source tree.
