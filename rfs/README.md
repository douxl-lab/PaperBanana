# RFS exchange v1 public API

This is a generative-first producer boundary around the existing five agents, not an additional agent and not a native-PPTX renderer. Paired consumer: `douxl-lab/research-figure-skill` v0.4.0+.

`contract.py` and both `schemas/` files are the canonical v1 scientific contract. The consumer retains the identical validator and semantically identical schemas, with original Apache-2.0 notices. `interop.descriptor()` exposes their fingerprints; incompatible pairs fail preflight. Keep the descriptor and the paired tests updated together when changing the contract. Do not silently reinterpret 0.1 consumer specs.

## API

- `validate_figure_spec` validates a full scientific spec; source traceability is not scientific proof.
- `scientific_fingerprint` includes all scientific fields and constraints, excluding only `design`.
- `compile_prompt` / `compile_comparison_prompt` construct model-aware instructions. Comparison prompts do not run a reviewer.
- `export_bundle` writes a new terminal producer candidate directory; original source, exact selected-stage prompt and design stay together.
- `verify_bundle` remains the legacy producer validator. At the paired boundary use `interop.verify_result(folder, expected_spec)` for stricter role/path/size checks and caller-science comparison.
- `interop.prepare_request`, `validate_request`, `request_argv` preserve the full spec/source and feed the existing CLI.

Spec fields: nodes; typed directed/undirected edges; groups; independent labels/formulas; full datasets with units/encoding; sources with quotes/hashes; locked constraints; aesthetic design; provenance. Generated structure stays unverified. The v1 external-consumer flag stays false; consumer acceptance is recorded separately rather than mutating the scientific fingerprint.

Successful wire bundle: image, prompt, design, spec and manifest. Input spec/system prompt/original JPEG/WebP/plot code are optional. Imported code is untrusted data, not an authorized program. Consumers must compare against their original expected scientific contract, not merely recompute the producer's hashes.

Cross-repository tests exercise the real producer exporter and real consumer importer using synthetic protocol images. They establish interchange, not live provider access, image quality, mathematical correctness or actual editable reconstruction. Existing runtime retry/checkpoint and failure semantics remain unchanged. See the main README and `skills/paperbanana/references/exchange.md`.
