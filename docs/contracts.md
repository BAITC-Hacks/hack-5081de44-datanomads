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
| `CandidateDatasetBuildRequest` | `candidate-dataset-build-request.v1` | Cycle, feedback references and frozen evaluation identifiers passed to the builder |
| `CandidateDatasetManifest` | `candidate-dataset-manifest.v1` | Immutable candidate artifact version, checksum and source lineage |
| `CandidateTrainingJob` | `candidate-training-job.v1` | Checksummed dataset reference, candidate/baseline versions and output location for the ML trainer |
| `CandidateTrainingResult` | `candidate-training-result.v1` | Candidate artifact/manifest locations, checksums and training status |
| `TrainedClassifierArtifact` | `trained-classifier-artifact.v1` | Versioned Naive Bayes parameters without copied training rows |
| `CandidateEvaluation` | `candidate-evaluation.v1` | Offline/shadow evidence and policy decision state |
| `ModelManifest` | `model-manifest.v1` | ML service model manifest envelope |

Candidate shadow inference uses the classify request with an explicit
`model_version` and `expected_artifact_checksum`. ML resolves only a trained
candidate manifest whose version and checksum match; a missing or invalid
candidate fails that shadow prediction and never falls back to production.
Core stores the production and candidate outputs separately in
`learning_cycle_shadow_predictions` and links an operator decision by its
PostgreSQL decision ID. Ticket text and operator notes are not copied into
that evidence row. Blind A/B preference collection is disabled and reported.

The checked-in `data/schemas/unified_ticket.schema.json` remains the Data
importer's schema. The standalone package carries the same v1 schema so an
artifact bundle can be checked by itself; `--check-demo` rejects any drift
between the two copies and validates every checked-in normalized demo record.
The export schema stores ticket identifiers and decision evidence, not a second
copy of ticket text. The Data/ML builder resolves those IDs against normalized
PostgreSQL ticket rows and uses dataset links to record source-version lineage.

The candidate build job stores only cycle, model, feedback and evaluation IDs,
versions, and export artifact locations. The Data/ML builder resolves the
referenced normalized tickets, excludes the frozen evaluation IDs, and writes
the checksummed candidate artifact. Its manifest records the evaluation
version and the exact ticket IDs retained for training.

The ML worker validates the dataset artifact and manifest checksums before
calling the versioned Data/ML trainer. Training writes a separate candidate
classifier artifact and manifest; it does not change the production model
pointer. The worker job and result contain only IDs, versions, artifact
locations, checksums, and aggregate counts.

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
