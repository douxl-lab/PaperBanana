---
name: paperbanana
description: Generate scientific diagrams and real-data plots with the five-agent PaperBanana pipeline, and export versioned RFS reconstruction bundles.
license: Apache-2.0
---

# PaperBanana research-figure engine

Use this engine to design scientific figures from supplied method text or real
quantitative data. Preserve the Retriever → Planner → Stylist → Visualizer →
Critic architecture and the original Vanilla/Polish alternatives.

## Before generation

1. Read the source material, desired figure purpose and any supplied FigureSpec.
   Never invent experimental measurements or scientific relationships.
2. Choose explicit text/image model profiles from `configs/model_registry.yaml`.
   Check capability requirements (vision, structured text, generation or edit).
   Account access is not guaranteed by the catalog. Do not guess a provider from
   available keys or silently switch a paid provider.
3. Configure credentials privately via environment or the git-ignored model
   YAML. Never include credentials in prompts, artifacts or version control.
4. Select reference retrieval only if required. `--retrieval-setting none` needs
   no benchmark download. `--offline` disables dataset downloads, not model API
   calls. Validate input and configuration before starting remote work.
5. For plot-code execution, use only trusted input in a disposable environment.
   A subprocess timeout is not a security sandbox. RFS CLI mode requires
   `--allow-generated-plot-code`; see `SECURITY.md`.

## Environment

Python 3.12 is the tested runtime. From the repository root:

```sh
python -m pip install -r requirements.txt
python skill/run.py --help
```

CLI options override environment values, then YAML, then registry defaults.
For a new/unknown model use an explicit `model_profiles` capability declaration,
not a name-based assumption. Model IDs and limits belong in the registry rather
than this skill document.

## Generate RFS candidates

```sh
python skill/run.py \
  --content-file method.txt --caption "Scientific architecture overview" \
  --task diagram --render-mode rfs --retrieval-setting none \
  --num-candidates 3 --max-concurrent 2 \
  --output output/figure.png --rfs-output-dir output/rfs \
  --checkpoint-dir output/checkpoints
```

Add `--rfs-file specification.json` to lock an existing FigureSpec while keeping
its original source input. Schema and illustrative fixtures are in `rfs/`.
Use `--render-mode legacy` for existing image-only contracts. Statistical plots
use `--task plot` and actual supplied numeric content. Do not assume the image
backend and the plotting-code backend support the same repair operations.

## Scientific and visual responsibilities

- FigureSpec records scientific entities, exact labels, directed edges, groups,
  source evidence, data and protected constraints.
- The image is the appearance reference; detailed design text explains the
  composition and local decisions. None replaces source evidence.
- Candidate variation changes presentation while preserving locked science.
- Local image repair requires an actual reference and a model with edit support.
  Do not silently substitute text-only regeneration. Protected pixels must still
  be inspected; a prompt cannot guarantee pixel-identical preservation.
- A malformed plan/review is a failure, not approval or “No changes needed.”
- Model-proposed structure remains unverified. A valid schema or attractive image
  does not prove scientific correctness.

## Deliverables and recovery

Each RFS candidate has a versioned manifest and image, exact prompt, design
instructions, FigureSpec and hashes. Preserve alpha/original bytes when possible.
The selected prompt/spec must describe the selected image's actual stage.
Keep failures explicit and keep valid prior artifacts without relabelling them
as fully reviewed successes. Verify bundle hashes before reuse.

Resume with the same source/configuration and `--resume --checkpoint-dir ...`.
Only matching successful candidates are reused; failed candidates may incur new
provider requests. API timeouts may have an unknown provider-side outcome.
Never overwrite unrelated artifacts or assume arrival order is candidate order.

The external RFS editable-reconstruction consumer is separate. This repository
provides a public versioned exchange contract, not verified native PPTX/draw.io
output. Consult `docs/RFS_GUIDE.md` and `rfs/README.md` for integration details.

## Verification

```sh
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
```

Controlled-transport tests validate software behavior. Do not describe them as
live provider generation, scientific-result validation or external consumer QA.
