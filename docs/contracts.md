# Versioned Data/ML contracts

The standalone contract package is `ml-service/contracts/`. Its
`index.json` maps the public names to JSON Schema Draft 2020-12 files; runtime
validation does not import Rust Core, React, `data/**`, or training code.

| Contract | Version | Purpose |
| --- | --- | --- |
| `UnifiedTicket` | `unified-ticket.v1` | Normalized ticket exchanged with Core storage |
| `DatasetManifest` | `dataset-manifest.v1` | Dataset identity, source files, checksums, counts and synthetic marker |
| `ClassifierManifest` | `classifier-manifest.v1` | Classifier artifact/runtime metadata |
| `EmbedderManifest` | `embedder-manifest.v1` | Embedder metadata, dimension, distance and preprocessing |
| `ModelEvaluation` | `model-evaluation.v1` | Versioned offline, shadow or backtest result |
| `LearningFeedbackExport` | `learning-feedback-export.v1` | Validated ticket references, predictions and operator decisions |
| `CandidateEvaluation` | `candidate-evaluation.v1` | Offline/shadow evidence and policy decision state |
| `ModelManifest` | `model-manifest.v1` | ML service model manifest envelope |

The checked-in `data/schemas/unified_ticket.schema.json` remains the Data
importer's schema. The standalone package carries the same v1 schema so an
artifact bundle can be checked by itself; `--check-demo` rejects any drift
between the two copies and validates every checked-in normalized demo record.
The export schema stores ticket identifiers and decision evidence, not a second
copy of ticket text. A dataset builder resolves those IDs against its versioned
normalized dataset package.

Trained manifests require a `sha256:` checksum with 64 lowercase hexadecimal
characters and an artifact URI. A deterministic baseline must instead declare
`artifact_kind: DETERMINISTIC_BASELINE`, `status: DEMO_BASELINE`,
`synthetic: true`, an implementation name, a null `artifact_checksum` and a
separate `demo_artifact_id` beginning with `demo-baseline:`. The demo identifier
cannot pass the trained artifact checksum schema branch.

Evaluation metric values are limited to numeric aggregates and nested numeric
structures. Feedback exports contain only allow-listed prediction/decision
fields; free-form operator comments and ticket text do not cross this contract.

The current service has deterministic inference adapters only. In demo,
development and test modes it serves the explicitly labeled baseline. A
missing/invalid manifest, or a trained manifest without its serving adapter,
keeps `/healthz` live but makes `/readyz` and inference return `503`; the
runtime does not substitute demo versions. Production mode remains unready
until a real artifact-serving adapter is available.

Run the independent artifact check from the repository root:

```bash
cd ml-service
python -m contracts.validate --check-demo
python -m contracts.validate --schema DatasetManifest --input ../data/manifests/demo-2026-09-21.json
```

Validation failures report the schema, field path and failed rule without
printing rejected field values. Contract shape changes require a new schema
version; the corresponding artifacts and runtime/OpenAPI adapters must be
updated together.
