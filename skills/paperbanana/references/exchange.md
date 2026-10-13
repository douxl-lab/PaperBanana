# Canonical generation boundary

When generating a standalone figure, enforce the **scientific mechanism, not infographic** default: no internal title, step numbers, unsolicited panel letters or large decorative regions. The RFS style reference defines the reusable rule, but the verified FigureSpec and explicit journal instructions outrank aesthetics. Do not silently promote multiclass Dirichlet/argmax equations to independent multi-label prediction: only draw the model's documented probability and uncertainty head.


Use the installed engine checkout, not the copied Skill folder, to execute Python. `requirements-exchange.txt` is enough for pure protocol validation; actual generation requires full requirements and privately configured provider credentials.

```bash
python skill/exchange.py describe
python skill/exchange.py validate-request --request /project/request/request.json
python skill/exchange.py run --request /project/request/request.json --allow-external
python skill/exchange.py verify-bundle --spec /project/request/figure-spec.json --bundle /project/request/generated/bundles/ACTUAL_CANDIDATE
```

The request contains the exact canonical FigureSpec, complete source text, protocol descriptor and bounded options; fields are individually hashed. No credential, arbitrary executable, output traversal or generated plotting program is accepted as a request field. Preserve user authorization boundaries and stable candidate identities.

The run wrapper feeds the existing five-agent CLI using `--render-mode rfs`, the exact `--rfs-file`, original `--content-file`, no dataset retrieval and a new output directory. Existing standalone CLI, UIs, resume and model-profile rules remain available. The wrapper does not promise exactly-once provider billing or remote cancellation. Missing/failed output is not editable reconstruction.

## Local TikZ visual-reference handoff

After a **successful** and hash-verified generation candidate, export a separate, immutable visual-reference package without calling the provider:

~~~bash
python skill/exchange.py vector-handoff \
  --spec /project/F01.json --bundle /project/request/generated/bundles/CANDIDATE \
  --out /project/vector-reference
python /PATH/research-figure/scripts/rfs.py run \
  --spec /project/vector-reference/figure-spec.json \
  --engine tikz --out /project/F01-tikz
~~~

The package includes the verified candidate PNG, design.md, figure-spec.json and an integrity manifest with scientific fingerprint and reference hashes. Failed or tampered bundles are rejected. A second handoff cannot overwrite the first. PaperBanana does not generate TikZ source, and RFS does not promise geometrically identical reconstruction of the image: science and publication-size visual quality remain pending until separately reviewed. FigureSpec v1 schema and descriptor fingerprints are unchanged.

Only `design` may vary. Scientific sources, nodes, edge type/direction, labels/formulas, groups, data and constraints stay locked. Successful manifest + hashes establish integrity; original expected science must also be checked. A matching quote does not establish semantic or mathematical correctness.

RFS consumes image, canonical spec and written design together. It must inspect the actual chosen image and write observed reconstruction notes before passing to paper-fig/Sivia/draw.io. Local tests exercise real exporter/importer code with synthetic images; live image quality and native-file rendering are separate checks. Do not commit private inputs, output manifests or internal review reports to the public repository.
