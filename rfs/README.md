# FigureSpec exchange contract v1

This package adds an optional scientific contract to PaperBanana's existing
five agents. It is a generative-first exchange layer, not a deterministic SVG
renderer, a sixth agent, or an editable-PPTX reconstruction implementation.
No external RFS consumer schema was available during implementation. This
versioned public mapping contract has local tests, but external consumer
acceptance and actual native-editable reconstruction remain unverified.

## Public API

- `validate_figure_spec(value, input_content=None)` returns an independent,
  validated dictionary or raises `RFSValidationError`.
- `scientific_fingerprint(spec)` hashes every scientific field, source and user
  constraint. Only `design` is excluded. All style/refinement steps preserve it.
- `compile_prompt(spec, design_description, capabilities, candidate_id=...,
  stage=..., repair=None)` is a pure deterministic function returning
  `CompiledPrompt(prompt, system_prompt, metadata)`.
- `compile_comparison_prompt(spec, candidates, capabilities)` creates a text
  review prompt. Supplying/identifying candidate images and executing the review
  are caller responsibilities. It does not select a candidate by itself.
- `export_bundle(data, new_output_dir, candidate_id=None)` exports a terminal
  candidate without overwriting an earlier directory.
- `verify_bundle(output_dir)` verifies the manifest and every artifact's bytes,
  hashes, relative paths, image MIME and scientific fingerprint before reuse.

Schemas: `schemas/figure-spec-v1.schema.json` and
`schemas/manifest-v1.schema.json`, both `schema_version: "1.0.0"`.
Unsupported versions and extra scientific fields are rejected; extend the
schema deliberately and version the exchange before changing it. Examples in
`examples/` are illustrative user-supplied fixtures, not paper-derived claims.

## Scientific structure and consumer mapping

| FigureSpec field | Meaning | Suggested consumer mapping |
|---|---|---|
| `nodes` | Stable ID, exact label, kind, optional formula/dataset refs, quoted evidence | Editable scientific shapes and formula/text objects |
| `edges` | Explicit source/target IDs, kind, direction, optional exact label, evidence | Typed connectors with correct endpoints and arrow direction |
| `groups` | Label, explicit member node IDs and evidence | Container/group objects without inventing additional nodes |
| `labels` | Exact text and explicit target IDs | Independent editable annotations tied to their targets |
| `datasets` | Exact original columns/rows, units, encodings, source ID and SHA-256 | Native chart/reproducible code data, never values inferred from raster pixels |
| `sources` | Complete supplied excerpt/table, content hash, optional URI/locator/safe relative path | Provenance/evidence record; paths are metadata and are never opened by this package |
| `constraints` | Locked required labels and layout/visual/negative instructions | Protected reconstruction invariants |
| `design` | Composition, layout, palette, typography and allowed candidate variation | Aesthetic reconstruction instructions guided by image and detailed design |
| `provenance` | User/model origin and scientific-verification status | Visible review boundary; model generation cannot assert verified facts |

Quoted evidence and matching source hashes establish traceability, not semantic
scientific truth. Auto-generated structure is explicitly `model_generated` and
`unverified`. The validator rejects unsupported numerical claims in model
labels/formulas, checks references and source excerpts, and preserves complete
supplied tables. These checks cannot prove that every scientific claim or
rendered pixel is correct. Generated plotting code is not checked for mathematical
or numerical correctness by this package. Inspect its computation and plotted
values independently before accepting scientific results. Structured input uses
strict JSON; missing values, duplicate keys and nonfinite numbers are rejected,
never repaired into allegedly original data. User-supplied contracts remain the authority and
cannot be rewritten by the model. Original input contracts are additionally
saved as `figure_spec.input.json` when supplied.

## Compilation stages and capabilities

Capabilities come from the explicit provider registry, not model-name guesses:
`provider`, `model`, `supports_system_prompt`, `supports_reference_image`,
`supports_editing`, and optional `prompt_notes`. System instructions are folded
into the exact submitted text when a provider cannot accept a separate system
message.

- `design`: initial scientific design with constrained presentation diversity
- `candidate`: alternate presentation of the same locked scientific structure
- `comparison`: separate scientific-fidelity and readability/aesthetic review;
  appearance is never evidence that a scientific fact is true
- `local_repair`: bounded reference-image editing; unsupported model/API
  capabilities fail rather than silently regenerate the whole figure
- `code_repair`: bounded quantitative plotting-code edits; the caller must
  supply previous code. It is not image editing and needs no image-edit API

A repair names targets, allowed presentation changes, every non-target protected
entity, and optional normalized protected rectangles. Data values, topology,
source evidence and terminology stay locked even inside target entities.
Protected pixels ultimately require inspecting the actual model edit output;
the prompt does not guarantee pixel-identical editing.

## Runtime and artifact record

RFS is enabled by `render_mode="rfs"`. Legacy description/JPEG-preview keys
remain available. Optional fields are `figure_spec`, `rfs_locked_spec`,
`rfs_input_figure_spec`, `rfs_stage_specs`, `rfs_design_description`, `rfs_repair`,
`rfs_reviews` and `rfs_artifacts`.

`rfs_artifacts` maps each legacy description key to an exact stage record:
`candidate_id`, `stage`, `provider`, `model`, actual `prompt`, separately submitted
`system_prompt` or null, `design_description`, `figure_spec` snapshot and its
`figure_spec_sha256`, original `image_base64`, `image_mime`, actual `options`,
`provenance`, and optional reproducible plotting `code`.
`eval_image_field` determines the selected record; the exporter does not guess
which stage is best or export a prompt from a different stage.

A successful directory includes `image.png`, `prompt.txt`, `design.md`,
`figure_spec.json` and `manifest.json`. It can include `prompt.system.txt`,
`figure_spec.input.json`, `plot_code.py`, and an original JPEG/WebP artifact.
Generated PNG bytes/alpha are preserved exactly. Non-PNG originals are preserved
separately and converted losslessly to a PNG reference; this does not restore
information already lost in JPEG. The manifest records sizes, SHA-256 values,
provider/model/options, stage/candidate identity, provenance, visual-model review
and explicit failure/verification states. Provider secrets are excluded.

Failure manifests may have no image/prompt, or retain the last valid image with
failed review status. Missing artifacts are never fabricated. Queued/running,
unknown or missing candidate status cannot be exported as successful. Existing
output directories, symlink destinations and unsafe exchange paths are rejected.
