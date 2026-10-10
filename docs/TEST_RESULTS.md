# RFS upgrade acceptance report

Date: 2026-10-10 (UTC). Baseline: `836455537e863b5a2f40dace487a782c0bc5ef94`.
This report covers the RFS upgrade in the same commit as this file.

## Final local gate

- Complete repository suite: **353 passed, 0 failed**, 20.91 seconds.
- Independent clean-checkout-equivalent snapshot, without local credential
  configuration and with provider-key environment variables removed: **353
  passed**, 18.72 seconds. The 15 independent SDK/security cases also pass
  separately (6.68 seconds). No necessary review blockers remain.
- All 37 original baseline tests remain. Legacy image mocks were updated to the
  shared adapter boundary with valid PNG bytes; the original behavior assertions
  were retained.
- Dependency consistency: `pip check` reports no broken requirements.
- Python compilation: agents, utils, rfs, skill, main.py, app.py and demo.py pass.
- Ruff correctness checks (`E9,F63,F7,F82`): pass. This is not a claim that all
  optional formatting/style rules were applied.
- Both CLI help entrypoints, shell script syntax, both example FigureSpecs,
  Markdown local links and `git diff --check`: pass.

Environment: Linux, Python 3.12.14; google-genai 2.29.0, openai 3.28.0,
anthropic 1.13.0, Gradio 6.30.0, Streamlit 1.65.0, pytest 9.1.1.
`requirements-tested.txt` records the exact installed environment;
`requirements.txt` specifies bounded runtime dependency ranges.

## What the suite actually exercises

- Exact provider/model capability resolution, configuration precedence, retired
  and unknown profiles, unsupported operations/geometry and preflight rejection.
- Actual installed Gemini and OpenAI SDK serialization through controlled HTTP
  transports, including the OpenRouter-compatible route. No live paid model
  generation was made. Test image/data fixtures are protocol inputs, not research
  results or evidence of model quality.
- Complete real five-agent pipelines, real runtime and local filesystem exports:
  initial planning, styling, generation, critique, bounded reference editing and
  final bundle verification. Both Gemini and OpenAI SDK paths are exercised.
- Scientific source/structure invariants, strict JSON (including duplicate keys
  and nonfinite values), task/contract matching, malformed critique and explicit
  distinction between code repair and image editing.
- Original image bytes, MIME and PNG alpha; matching final prompt, design and
  scientific specification; bundle hashes, unsafe paths and refusal to overwrite
  existing bundle directories.
- Bounded concurrency and cancellation, shared-retrieval waiter lifetime,
  per-candidate failure isolation, atomic checksums, invalid checkpoint rejection
  and matching-source resume.
- Legacy diagram and plot workflows, dynamic critic stages, UI-generated sample
  identities, lossless reference editing, evaluation paths, repeated event-loop
  cleanup and notebook export cells.

## Independent review regressions

`tests/test_independent_review.py` contains 15 independent regression cases.
They use actual installed SDKs with controlled HTTP transports and temporary
configuration, not a populated local credentials file. Review found and verified
fixes for last-valid-image loss on failed refinement, incomplete stage rendering,
corrupt successful-checkpoint reuse, candidate output path traversal, incompatible
UI candidate IDs, timed-out retrieval concurrency leakage, terminal gateway
responses incorrectly treated as successful images, and no-system-capability
prompt/wire mismatches. The original scientific and image-byte assertions were
not removed to obtain a passing result.

## UI and entrypoint smoke checks

- Gradio 6.30 application launch with its CSS/theme: HTTP 200.
- Streamlit server health: HTTP 200.
- Streamlit AppTest execution of the main UI and both result viewers: no
  exceptions. The main UI rendered 13 select boxes.
- All application processes started for these checks were stopped afterward.
- Notebook code cells compile; Simple/Advanced export cells execute against
  controlled local fixtures. The notebook defaults to this fork and does not
  overwrite an existing custom checkout.

## Reproduce the local gate

```sh
python -m pip install -r requirements-dev.txt
python -m pip check
HF_HUB_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False MPLBACKEND=Agg python -m pytest -q
python -m ruff check --select E9,F63,F7,F82 agents utils rfs skill main.py app.py demo.py tests
python -m compileall -q agents utils rfs skill main.py app.py demo.py
python skill/run.py --help
python main.py --help
bash -n scripts/run_demo.sh scripts/run_main.sh
```

## Explicitly not established

- Live account access, billing/quotas, provider-side availability, image quality,
  or real remote timeout/rate-limit behavior.
- Semantic or mathematical correctness of generated figures and plotting code.
- End-to-end integration with an external RFS consumer, editable PowerPoint or
  draw.io reconstruction. This repository provides a versioned exchange contract
  and documented mapping; the consumer implementation was not supplied.
- Full authenticated Colab execution, Docker build/runtime or browser pixel QA.
- GitHub Actions status before pushing. The workflow is included, but its actual
  remote result must be checked separately after the main branch update.

Generated Python plotting remains trusted code execution, not a security
sandbox. See `SECURITY.md` and `RFS_GUIDE.md` for deployment and recovery limits.
