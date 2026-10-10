# Execution and data boundaries

PaperBanana sends scientific source text, prompts and selected reference images
to the explicitly configured model provider. Choose accounts and material you
are permitted to use. A second configured API key must not silently reroute a
request. Never commit credentials or include them in FigureSpec/artifact bundles.

## Generated plotting code

Statistical-plot mode can execute generated Python. A child process and a time
limit improve reliability; they are **not a security sandbox**. Such code can
access files, network and credentials available to that process. Only execute
trusted inputs in a disposable, least-privilege environment, and use the CLI's
explicit execution opt-in. Do not expose this feature to untrusted public users.
No AST blacklist is represented as a comprehensive security boundary.

## Web applications

The legacy Streamlit/Gradio credential controls are intended for a trusted
single-user deployment. Process-global client configuration is not tenant
isolation. Use a dedicated instance per account; do not publish a shared UI
containing personal provider credentials. Container port exposure requires the
operator's own access controls. Local launch does not authorize public sharing.
The Docker build context excludes local credentials, datasets, generated outputs
and checkpoints. Supply provider configuration privately at runtime, not by
copying a populated configuration into an image.

## Scientific integrity

A successfully generated image is not proof that a method or scientific claim
is correct. User-provided scientific invariants remain authoritative. Generated
structure is marked as proposed/unverified and must be checked against the
source material. Invalid structured responses and incomplete critique must not
be labelled as successful scientific validation.

## Recovery

Candidate checkpoints and exported images may contain unpublished research.
Store them privately. Resume requires matching inputs and configuration; do not
hand-edit a checkpoint to make a different run appear completed. A request
timeout may leave a provider-side outcome unknown and may have incurred cost.
