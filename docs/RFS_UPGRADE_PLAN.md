# RFS engine upgrade plan

Baseline reviewed before implementation: `836455537e863b5a2f40dace487a782c0bc5ef94`.
The complete baseline inventory and findings are in [BASELINE_AUDIT.md](BASELINE_AUDIT.md).

## Architecture retained

Retriever → Planner → Stylist → Visualizer → Critic remains the core workflow.
Vanilla, Polish, statistical plots, the two existing UIs, benchmark evaluation,
and legacy result keys remain supported. The RFS layer supplies contracts and
adapters around those agents; it is not a sixth agent or a replacement renderer.

## Implementation boundaries

1. A data-driven model registry specifies provider, operation, limits and
   parameter support. Public documentation is evidence for a profile, not proof
   that a particular account has access. Custom profiles must state capabilities.
2. Shared adapters perform explicit routing, image-reference normalization,
   provider-specific options, bounded transient retries and typed failures.
   Changing available keys must not silently change the chosen provider.
3. A versioned FigureSpec and scientific-invariant validation distinguish user
   constraints from unverified model-proposed structure. A pure Prompt Compiler
   handles initial design, candidate composition and targeted repair.
4. Each RFS candidate exports its image, exact generation prompt, detailed design
   description, scientific structure and integrity/provenance manifest. The
   image is visual evidence, not an authoritative replacement for source facts.
5. The runtime validates inputs before optional dataset downloads, bounds
   concurrency, isolates candidate errors, persists atomic checkpoints and
   refuses to resume against changed inputs/configuration.
6. Legacy entrypoints use the same routing and output-resolution mechanisms.
   Generated plotting code is trusted code execution, not a security sandbox;
   its risk and timeout are explicit.

## Acceptance gate before main is updated

- Retain and run the baseline pytest suite, not only unittest discovery.
- Add focused provider, contract, compiler, export, CLI, concurrency, recovery,
  dataset and legacy-entrypoint regressions.
- Validate adapter request shapes against the installed SDKs without secrets.
- Exercise the complete five-agent path with controlled transports and actual
  local artifact I/O. Test malformed outputs, retries, cancellation, partial
  failures, recovery and scientific-invariant violations.
- Run dependency consistency, syntax/import/CLI checks and independent review.
- No failing necessary test may be suppressed to permit a push.
- Only after the gate passes: commit on main, push main without force or PR,
  verify the exact remote commit and report the commit link.

## Verification boundaries

Offline test transports do not establish live image quality, account model
availability, billing, or a successful paid provider request. These require a
separate credentialed smoke run and must be reported as not run when absent.
The external RFS consumer implementation/schema has not been supplied; this
repository defines a documented, versioned exchange contract for integration,
not a claim of verified PowerPoint/draw.io reconstruction.
