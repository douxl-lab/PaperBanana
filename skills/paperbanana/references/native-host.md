# Native host execution

`skill/native_host.py` is an offline stage coordinator for a host that already
provides language/vision reasoning and a built-in image-generation tool. It does
not discover credentials, call a provider, execute a model, or install a cloud
runtime. The existing provider-backed `skill/run.py` and `skill/exchange.py`
entrypoints remain separate and unchanged.

## Requirements and initialization

Use Python 3.10+ and `requirements-exchange.txt` (jsonschema and Pillow). Run from
the engine checkout, not a standalone copy of the installed Skill directory.
Do not run Python with `-O`; the adapter refuses optimized execution because its
validation guards use assertions.

```bash
python skill/native_host.py prepare --spec /project/figure-spec.json --job /project/native-job
python skill/native_host.py next --job /project/native-job
```

A fresh directory is mandatory. The canonical FigureSpec and source text remain
bound by the repository's RFS request validator and scientific fingerprint.

## Host responsibilities

Execute the returned stage, then record its actual output. `next` never executes
it for you. Keep the fixed order Retriever → Planner → Stylist → Visualizer →
Critic. Do not fabricate task IDs, successful payloads, images or reviews. Use an
actual host task/tool ID; if the platform exposes none, explicitly record
`tool-managed-not-reported`. Receipts are local attestations, not platform-signed
proof or a tamper-proof audit log.

The first three stages and Critic require a JSON object with `status: succeeded`
and no error, plus these fields:

- Retriever: `observations`, `source_paths`.
- Planner: `design_description`.
- Stylist: `prompt`, `design_description`; optional `referenced_image_paths`.
- Critic: `image_sha256`, `verdict` (`accept` or `revise`),
  `observed_scientific_fidelity`, `aesthetics`, `issues` (may be empty).

```bash
python skill/native_host.py record --job /project/native-job --stage Retriever \
  --artifact /project/retriever.json --task-id ACTUAL_HOST_TASK_ID
```

Repeat for Planner and Stylist. For Visualizer, call the host's built-in image
generator with the exact returned `tool_arguments` (the tool name is a host
integration hint). Preserve the actual returned PNG bytes and the returned prompt
hash. Do not substitute a synthetic image or claim a planned call ran.

```bash
python skill/native_host.py record --job /project/native-job --stage Visualizer \
  --artifact /project/generated.png --task-id ACTUAL_IMAGE_TOOL_ID \
  --prompt-sha256 HASH_FROM_NEXT
```

The host must open the actual image before creating the Critic payload. Record
Critic with the same `record` command and inspect `verify`:

```bash
python skill/native_host.py verify --job /project/native-job
```

`ACCEPTED` means the recorded Critic accepted the image; it does not establish
scientific truth, measured performance, publication readiness or native
editability. `REVISION_REQUIRED` is not success. Native job directories are not
provider exchange bundles and cannot be passed directly to `verify-bundle` or
`vector-handoff`.

## Optional reference binding

Before recording any stage, the host may create `context.json` in the fresh job,
with `files` entries containing absolute `path` and SHA-256 `sha256` fields. Only
ordinary bounded files are allowed; every verification checks their bytes.
Stylist reference paths must be present in this context. Do not change context
once stages are recorded. The host is responsible for inspecting references,
obtaining any required data-sharing permission, and verifying that the image tool
can access those exact files. Local validation is not permission to upload them.

## Bounded repairs

A repair requires a fully verified source whose Critic verdict is `revise` and a
new output directory. Its JSON edit must pass `rfs.contract.validate_repair`:
`instruction`, known `target_ids`, explicit `allowed_changes`, and all remaining
IDs in `protected_ids`; optional protected regions follow the RFS contract.

```bash
python skill/native_host.py repair --source-job /project/native-job \
  --job /project/native-repair --edit /project/bounded-edit.json
```

The adapter retains source receipts for the first three stages and anchors the
new Visualizer request to an unchanged copy of the original image. The host must
execute Visualizer and Critic again. `--repair-only` replaces the original style
prompt with the bounded edit instruction; it does not unlock scientific fields.
The edit describes allowed changes but cannot guarantee that an image model will
obey them: inspect the repaired pixels and reject scientific drift.

Keep all jobs, actual prompts, reference images, source materials and local
receipts outside version control. Tests use synthetic fixtures, never live
research artifacts or paid model execution.
