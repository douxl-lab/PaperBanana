# PaperBanana as an RFS research-figure engine

## What is retained and what is added

The five core agents remain Retriever, Planner, Stylist, Visualizer and Critic.
The legacy image workflows remain available. RFS mode adds a structured,
versioned exchange contract around them. It does not replace the generative
designer with an SVG template and does not itself produce editable PowerPoint.

The exchange has three complementary sources of information:

1. `figure_spec.json`: scientific entities, directed relationships, grouping,
   labels, source evidence, raw data and constraints.
2. `image.png`: the generated visual reference.
3. `design.md` and `prompt.txt`: detailed composition instructions and the actual
   generation prompt used for the selected image.

Each bundle also contains a manifest with identities, model/provider provenance,
status, file sizes and hashes. Downstream RFS consumers should verify it before
rebuilding an editable figure. The schema is defined here; compatibility with
an external RFS consumer has not been demonstrated without that consumer.

## Setup

Python 3.12 is the acceptance environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python skill/run.py --help
```

Copy `configs/model_config.template.yaml` to the git-ignored
`configs/model_config.yaml`, or set the selected provider's environment key.
Never put a key in a prompt, FigureSpec or output bundle. Model names resolve
through `configs/model_registry.yaml`; configuration overrides do not require
editing agents. Explicit CLI model options override environment values, which
override YAML defaults, then the registry defaults are used.

The bundled profiles record their documentation sources and verification date.
They cannot guarantee model availability on an account. Unknown models need an
explicit capability profile rather than guessing from a name. Preview/retired
models must not be silently mapped to a different paid provider.

## Generate a research diagram

```sh
python skill/run.py \
  --content-file method.txt --caption "Architecture overview" \
  --task diagram --render-mode rfs \
  --retrieval-setting none --num-candidates 3 --max-concurrent 2 \
  --output output/architecture.png \
  --rfs-output-dir output/rfs \
  --checkpoint-dir output/checkpoints
```

An existing scientific specification may be supplied with `--rfs-file` while
keeping the original method/data input. See `rfs/examples/` and
`rfs/schemas/figure-spec-v1.schema.json`. A model-proposed specification is
explicitly unverified; syntactic validation does not establish scientific truth.
Locked source facts and numeric data must not be changed by style or repair.

The Prompt Compiler separates scientific constraints from presentation, adapts
to explicit provider capabilities, gives candidates presentation diversity, and
requires reference-image editing support for local repair. Exact preserved
pixels and semantic correctness still depend on model output and human review.

## Statistical plots

Use real structured numeric input, not invented measurements. The current plot
backend generates Python/matplotlib code. RFS CLI plotting requires explicit
`--allow-generated-plot-code`. Read [SECURITY.md](../SECURITY.md): a child process
and a timeout are reliability controls, not a security sandbox. Use a disposable
trusted environment. The library's legacy execution behavior is retained for
compatibility and must not be exposed to untrusted public input.

## References, concurrency and recovery

- `--retrieval-setting none` avoids reference-dataset downloads.
- Reference data is fetched only when required by the selected workflow.
- `--offline` disables dataset downloads; it does **not** turn model generation
  into a local/offline model. Required missing references must be supplied or
  explicitly disabled. `--dataset-policy` controls automatic/never/download.
- `--max-concurrent` bounds active candidates. The configured candidate timeout
  bounds the pipeline; per-provider attempts and plot-code execution are also
  bounded. Cancellation must not become a successful image.
- `--resume --checkpoint-dir ...` reuses only successful candidates with matching
  source/configuration/dependency bindings. Different papers do not share the
  first paper's retrieval result.
- Failed candidates retain a failure record. Resume may retry unsuccessful
  candidates and incur new provider cost; a prior timeout may have an unknown
  provider-side outcome. It is not an exactly-once payment guarantee.
- Candidate IDs are independent of completion order. Existing RFS bundle
  directories are not overwritten; matching resumed bundles are verified.
  The legacy CLI image path and its `.results.json` sidecar are explicit output
  targets and may be replaced on another invocation. Use a unique `--output`
  path for each independent run to retain earlier images and result records.

## Validation and integration

```sh
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
python -m compileall -q agents utils rfs skill main.py app.py demo.py
```

Tests include controlled provider responses and actual local artifact I/O.
These establish software contracts, not live provider access, generated image
quality or scientific validity. The acceptance report lists exact executed
checks and explicitly identifies credentialed tests that were not run.

For maintainers, see [the pre-implementation audit](BASELINE_AUDIT.md),
[implementation gates](RFS_UPGRADE_PLAN.md), and
[official model/API evidence](MODEL_API_AUDIT.md).
