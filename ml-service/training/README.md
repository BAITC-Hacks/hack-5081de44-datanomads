# Offline Data/ML contracts

Use Python 3.11.15 for the training environment. The runtime service installs
`requirements.txt` separately; training packages stay out of its default image.

```bash
uv venv --python 3.11.15 .venv
uv pip sync --python .venv/bin/python ml-service/training/requirements.lock
PYTHONPATH=ml-service .venv/bin/python -m unittest discover \
  -s ml-service/training/tests -v
```

The lock resolves 63 packages for Python 3.11.15 on Linux x86_64. Direct
requirements and the core version constraints are kept separately so the lock
can be regenerated when training code changes. This lock pins package versions,
but does not pin wheel hashes or other operating systems.

`contracts.py` validates dataset, classifier, embedder and evaluation manifests.
All require explicit synthetic origin and SHA-256 evidence. Dataset versions
also require a saved seed, source checksums, a group split policy and a frozen
evaluation version. The classifier handoff schema applies to reviewed-package
candidate artifacts; the existing deterministic baseline and synthetic
classifier demo are not retroactively promoted to it. The embedder schema is
for a future artifact.

## Reviewed dataset package

`scripts/build_training_dataset.py` requires two reviewed JSONL inputs and
their versioned scenario/relation sources. Classifier rows must carry a
canonical topic, `scenario_id`, `variant_id`, RU/KZ/MIXED language, style,
generator model/prompt/seed, source checksum and human review provenance.
Retrieval rows must carry a `relation_group`, query/candidate IDs and texts,
one of `DUPLICATE`, `SIMILAR_BUT_NOT_DUPLICATE`, `UNRELATED` or `REPEAT`, source
checksum and the same review provenance. Both inputs require
`review_status=APPROVED`, reviewer ID, aware timestamp and a SHA-256 evidence
reference. The builder also requires the original classifier candidates and
both complete review queues. It reruns the existing review validators and
requires each approved input row to match an approved review decision exactly.
Their checksums are stored in the dataset manifest. This links the package to
review evidence; reviewer identity and the fact of human review still require
the normal organizational approval process. `PENDING` demo candidates are rejected.

The synthetic retrieval pilot source and its manual review/export workflow are
documented in [`docs/retrieval-gold-pilot.md`](../../docs/retrieval-gold-pilot.md).
Use the unchanged source file as `--relation-source` when the human-approved
export is eventually passed to the dataset builder. The pilot has no approved
labels yet and is insufficient for quality claims.

```bash
.venv/bin/python scripts/build_training_dataset.py \
  --classifier /path/to/reviewed_classifier.jsonl \
  --classifier-candidates /path/to/pending_classifier_candidates.jsonl \
  --classifier-review /path/to/classifier_review.jsonl \
  --retrieval /path/to/reviewed_relations.jsonl \
  --retrieval-review /path/to/retrieval_review.jsonl \
  --scenario-source /path/to/versioned_scenarios.jsonl \
  --relation-source /path/to/versioned_relation_evidence.jsonl \
  --dataset-version reviewed-v1 --frozen-evaluation-version eval-v1
```

The command creates `data/processed/<dataset_version>/` once. It writes all
six classifier/retrieval splits, `membership.json`, `frozen_evaluation.json`,
`audit.json` and a manifest with per-file checksums and lineage. Scenario and
relation groups remain in one split. At least 10 reviewed topics with RU/KZ
coverage and MIXED examples are required. This builder handles synthetic
scenario data; real chronological exports need a separate time-aware policy.

Candidate builds reuse the immutable test files and reject frozen IDs, groups
or exact text:

```bash
.venv/bin/python scripts/build_training_dataset.py \
  --classifier /path/to/new_reviewed_classifier.jsonl \
  --classifier-candidates /path/to/new_pending_classifier_candidates.jsonl \
  --classifier-review /path/to/new_classifier_review.jsonl \
  --retrieval /path/to/new_reviewed_relations.jsonl \
  --retrieval-review /path/to/new_retrieval_review.jsonl \
  --scenario-source /path/to/new_versioned_scenarios.jsonl \
  --relation-source /path/to/new_versioned_relation_evidence.jsonl \
  --dataset-version reviewed-v2 --frozen-evaluation-version eval-v1 \
  --frozen-from data/processed/reviewed-v1
```

Offline evaluators must receive a local artifact path and run with
`HF_HUB_OFFLINE=1` to avoid Hub checks or downloads.

## Classifier baselines

After a reviewed package exists, run both fixed baselines on its train split:

```bash
.venv/bin/python scripts/evaluate_classifier_baselines.py \
  --dataset data/processed/reviewed-v1 \
  --output data/processed/reports/reviewed-v1-classifier-baselines.json
```

The command checks package and frozen test checksums, then writes majority and
character TF-IDF + LinearSVC metrics for validation and test. It reports
macro/weighted/per-class F1, accuracy, confusion matrices, RU/KZ/MIXED slices,
and region slices with at least 30 examples. Train class counts and imbalance
are explicit. Model settings are fixed before test evaluation; the output is
synthetic evidence only until a real reviewed dataset is available. The output
path must be new.

## Classifier input length

Before selecting a truncation strategy for a reviewed classifier candidate,
audit only its train and validation splits with a local tokenizer:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/audit_classifier_tokens.py \
  --dataset data/processed/reviewed-v1 \
  --tokenizer /path/to/local/xlm-roberta-tokenizer \
  --output data/processed/reports/reviewed-v1-token-lengths.json
```

The report records p50/p95/p99/max and counts above 384 and 512 tokens for
RU, KZ, MIXED and all examples. It identifies head-384, head-512 and both
head+tail variants as strategies to compare on validation. Length counts alone
do not select a strategy or prove model quality. Frozen test examples are
verified by package checksum, but their token lengths are not computed. The
current unreviewed synthetic demo is not evidence about customer appeal lengths.

The existing offline classifier trainer also accepts a verified reviewed
package. It checks the package, requires all 16 topics used by the current
runtime and matches the token audit to the package and local tokenizer before
loading weights. The following command is an example after human review and
input-strategy selection; `384` is not an automatic recommendation:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python ml-service/train_classifier.py \
  --reviewed-dataset data/processed/reviewed-v1 \
  --token-audit data/processed/reports/reviewed-v1-token-lengths.json \
  --base-model /path/to/local/xlm-roberta-base \
  --max-length 384 \
  --output-dir ml-service/artifacts/classifier-reviewed-v1
```

Use `--input-length-strategy head-tail` with either `--max-length 384` or
`--max-length 512` to keep tokens from both ends of a long appeal. The default
is `head`. Run the four combinations (`head` and `head-tail` at 384 and 512)
with `--validation-only` into separate output directories. Those runs leave
`metrics.test=null` and cannot be packaged as evaluated candidates. Compare
their validation metrics, fix the chosen configuration, then run that
configuration once without `--validation-only` for frozen-test reporting.
The token audit alone cannot select a strategy. The selected strategy is stored
in the model manifest and applied by offline feedback retraining and runtime
inference as well.

```bash
.venv/bin/python scripts/compare_classifier_input_strategies.py \
  --artifact /path/to/head-384-selection \
  --artifact /path/to/head-tail-384-selection \
  --artifact /path/to/head-512-selection \
  --artifact /path/to/head-tail-512-selection \
  --output data/processed/reports/input-strategy-comparison.json
```

The comparison checks shared dataset, tokenizer audit, base model and training
settings, verifies model weight checksums, and reports only validation scores.
It picks the highest validation macro-F1; ties prefer 384 tokens, then `head`.
The report recommends a configuration, not a model promotion.

The output directory must be new and outside the dataset package. The trainer
uses only local model files in reviewed mode. Current reviewed package schema
is synthetic; its holdout metrics are not real-citizen quality evidence. This
runner uses full fine-tuning so the runtime can load one standalone local
`safetensors` artifact without an adapter dependency. Batch size is configurable;
GPU memory, inference latency and the 384/512/head+tail quality comparison must
still be measured on the reviewed corpus. A failed run leaves no final artifact
directory, and successful artifacts are not promoted automatically.

Temperature and a candidate confidence threshold are selected on validation
before the frozen test is evaluated. Policy `classifier-uncertainty.v1` requires
at least 90% precision among at least 30 selected examples, including at least
10 RU and 10 KZ examples; otherwise the candidate threshold is 1.0. Scores
below 0.55 are `LOW_CONFIDENCE`. Synthetic holdout evidence always disables
`CONFIDENT`, and every prediction still requires operator review. Validation
and test reports include NLL, 10-bin ECE, language slices and state coverage.
Unknown/ambiguous challenge examples and real-citizen calibration remain
unverified.

The pinned Transformers version can emit a misleading Mistral-regex warning
when loading a local non-Mistral XLM-R tokenizer; this is tracked in the
[Transformers issue](https://github.com/huggingface/transformers/issues/42591).
Do not set `fix_mistral_regex=True` for this XLM-R artifact: it changes the
token IDs used by its existing synthetic model.

## Classifier candidate evaluation

After a reviewed candidate artifact and the baseline report exist, compare
them on the same frozen test:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/evaluate_classifier_candidate.py \
  --dataset data/processed/reviewed-v1 \
  --model ml-service/artifacts/classifier-reviewed-v1 \
  --baseline-report data/processed/reports/reviewed-v1-classifier-baselines.json \
  --output data/processed/reports/reviewed-v1-classifier-candidate.json
```

The evaluator verifies dataset, baseline and model lineage before loading the
local artifact. It checks the model weight checksum through the runtime loader,
then reports accuracy, macro/weighted/per-class F1, confusion matrix, RU/KZ/MIXED
slices, confidence states, review share and single-text p50/p95 latency after
three warmup calls. It stores no ticket text. A candidate below TF-IDF macro-F1
gets `NO_GO_BASELINE_OUTPERFORMS`; any other result remains
`PENDING_HUMAN_REVIEW`. This comparison does not apply a critical-regression
policy or authorize promotion. Latency is evidence only for the machine named
in that report, and synthetic holdout results cannot establish quality on
customer appeals.

## Classifier handoff bundle

Build a self-contained candidate handoff from the verified reviewed package,
local model artifact and fixed baseline report:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/package_classifier_candidate.py build \
  --dataset data/processed/reviewed-v1 \
  --model ml-service/artifacts/classifier-reviewed-v1 \
  --baseline-report data/processed/reports/reviewed-v1-classifier-baselines.json \
  --output ml-service/artifacts/classifier-bundle-reviewed-v1
HF_HUB_OFFLINE=1 .venv/bin/python scripts/package_classifier_candidate.py verify \
  --bundle ml-service/artifacts/classifier-bundle-reviewed-v1
```

The builder reruns the frozen-test comparison, records aggregate metrics and
copies the model weights, tokenizer and config into `model/`. The root manifest
pins the dataset checksum, evaluation version, report checksum, model checksum
and every bundle file. `verify` rejects missing, changed, extra or linked files
and inconsistent lineage. The output path must be new; keep it under the ignored
artifact directory because it contains model weights. Status remains
`CANDIDATE`; this handoff does not promote the model. Its model card states that
the reviewed synthetic package contains no real citizen appeal texts and cannot
establish quality on the customer CSV exports.

## Production and candidate comparison

`scripts/evaluate_classifier_pair.py` compares two local classifier artifacts
on the same verified frozen test. Supply a policy file fixed before evaluating
the candidate; it must contain `policy_version` set to
`classifier-critical-regression.v1`, `critical_topics`, `max_f1_drop`,
`min_topic_support` and `min_total_samples`. No regression threshold is chosen
by the evaluator.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/evaluate_classifier_pair.py \
  --dataset data/processed/reviewed-v1 \
  --production /path/to/production-classifier \
  --candidate /path/to/candidate-classifier \
  --policy /path/to/approved-critical-regression-policy.json \
  --output data/processed/reports/production-vs-candidate.json
```

The report records both model versions, artifact checksums, the policy checksum,
one frozen sample-ID checksum, aggregate metrics and each critical topic's F1
drop. It returns `CRITICAL_REGRESSION` when a supported critical topic exceeds
the policy limit, `INSUFFICIENT_EVIDENCE` when minimum support is unmet, or
`PENDING_HUMAN_REVIEW`. It contains no ticket text and does not promote a model.
The evaluator checks the candidate's production and frozen-test lineage,
requires matching input length and strategy, compares tokenization on every
frozen text, and loads the two classifiers sequentially to limit memory use.
Current reviewed packages are synthetic, so the comparison cannot establish
performance on actual citizen appeals. A real controlled-learning comparison
still needs reviewed feedback, a frozen dataset, a trained production artifact
and a configured offline worker.

## Fresh shadow evaluation

`scripts/evaluate_classifier_shadow.py` accepts a PII-free
`classifier-shadow-input.v1`/`.v2` JSONL. Each row has one ticket/feedback ID, one
operator-confirmed topic and **both** production and candidate predictions for
that ticket. It also carries cycle/model versions, an imported source dataset
version (v1) or `source_origin_kind=RUNTIME_API` (v2), synthetic origin,
decision time and `validation_status=VALID`. Ticket text is
not part of the contract. Duplicate tickets, mixed model versions, malformed
decisions and rows outside the policy window fail validation.

The required policy is `classifier-shadow-policy.v1` with
`promotion_policy_version`, an aware `window_start`/`window_end`,
`min_samples`, `min_real_samples`, `critical_topics`, `min_topic_support`,
`max_topic_agreement_drop` and `max_correction_rate_increase`. Critical-topic
support counts real operator decisions. Set the policy before examining
candidate results.

```bash
.venv/bin/python scripts/evaluate_classifier_shadow.py \
  --input /path/to/validated-paired-shadow.jsonl \
  --policy /path/to/approved-shadow-policy.json \
  --cycle-id cycle_1 \
  --production-model-version classifier_production_v1 \
  --candidate-model-version classifier_candidate_v1 \
  --output data/processed/reports/cycle_1-shadow.json
```

The report has production/candidate agreement, correction rates and their
delta, per-topic slices, critical regressions, real/synthetic counts and a
single sample-ID checksum. Overall metrics describe every row; promotion
gates use only real-origin rows (`gate_population=real_only.v1`), both for
critical-topic agreement drops and the global correction-rate increase.
It records `blind_ab_enabled=false` and no
preference score because the current UI has no blind A/B. Too few total, real
or critical-topic samples give `INSUFFICIENT_EVIDENCE`; an adequately sized
report with a critical or global correction-rate regression gets a `NO_GO`
decision. Only a `VALID` report without regressions can contribute to
promotion evidence. Recalculate reports created before `real_only.v1` before
review or promotion.
Core and the offline worker capture candidate shadow predictions for fresh
tickets when a trained candidate and offline report exist. Export verified
pairs from PostgreSQL after the policy window has closed:

```bash
DATABASE_URL="$DATABASE_URL" .venv/bin/python scripts/export_classifier_shadow.py \
  --cycle-id cycle_1 --policy /path/to/approved-shadow-policy.json \
  --review-links /path/to/approved-text-links.jsonl \
  --output data/processed/reports/cycle_1-shadow-input.jsonl
```

The import path requires one dataset link and valid source checksums. The
runtime API path requires no dataset links, an approved `feedback-review-link.v2`
with the exact ticket/source/external ID/text hash and reviewed synthetic/real
origin, a matching Core `CREATE_TICKET` audit event, `source_system=api`, and
creation timestamps within five minutes of the server-generated API ID. API
tickets with a custom `source` are excluded pending a platform origin marker.
Both paths require a production prediction before the candidate prediction,
and exactly one later operator decision inside the same window. Real-origin
rows also require an approved
text checksum matching the ticket text; without review links they are excluded.
The output contains no ticket text. Missing or ambiguous rows appear only in
the rejected-count summary. An empty export reports `INSUFFICIENT_EVIDENCE`.

To calculate and store the report in the candidate's `model_evaluations` row,
run the combined command. It re-reads PostgreSQL and recomputes the export in
one transaction, so it does not trust a supplied JSONL for promotion evidence:

```bash
DATABASE_URL="$DATABASE_URL" .venv/bin/python scripts/record_classifier_shadow_report.py \
  --cycle-id cycle_1 --policy /path/to/approved-shadow-policy.json \
  --review-links /path/to/approved-text-links.jsonl
```

The Core candidate-evaluation endpoint then exposes the stored offline and
shadow reports to an ML reviewer. Valid matching offline and shadow reports
set `READY_TO_REVIEW`; promotion still requires a separate human decision. The
available customer CSVs have no original appeal text, so generated texts must
not be approved as real-origin rows.

## Retrieval baselines

Evaluate relation groups from the same reviewed package:

```bash
.venv/bin/python scripts/evaluate_retrieval_baselines.py \
  --dataset data/processed/reviewed-v1 \
  --output data/processed/reports/reviewed-v1-retrieval-baselines.json \
  --e5-model /path/to/local/multilingual-e5-base
```

The lexical baseline fits character TF-IDF on the training split. The E5
baseline loads a local model with no Hub access, prefixes queries and passages,
mean-pools masked tokens and normalizes embeddings as described in the
[multilingual E5 model card](https://huggingface.co/intfloat/multilingual-e5-base).
Without `--e5-model`, only lexical metrics are produced and `e5_model_status`
is `NOT_RUN`; no pretrained result is implied. The report includes
Recall@1/3/5, MRR, Precision@1/3/5 and graded nDCG@5. Relevance grades are
`DUPLICATE=2`, `SIMILAR_BUT_NOT_DUPLICATE=1`, `REPEAT=1`, `UNRELATED=0`.
Recall/MRR/nDCG average queries with at least one relevant candidate;
precision includes queries without one. The report emits IDs and scores for a
blind top-3 expert review queue, with `PENDING` status and no ticket text.

## Offline embedder candidate

After a reviewed relation package and a saved E5 baseline report exist, train a
candidate from a **local** base model directory. The directory checksum must
match the baseline report. The declared `--base-model-id` is recorded with that
checksum; it does not establish model identity by itself.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/train_embedder_candidate.py build \
  --dataset data/processed/reviewed-v1 \
  --base-model /path/to/local/multilingual-e5-base \
  --base-model-id intfloat/multilingual-e5-base \
  --baseline-report data/processed/reports/reviewed-v1-retrieval-baselines.json \
  --model-version embedder-reviewed-v1 \
  --output ml-service/artifacts/embedder-reviewed-v1
HF_HUB_OFFLINE=1 .venv/bin/python scripts/train_embedder_candidate.py verify \
  --artifact ml-service/artifacts/embedder-reviewed-v1
```

The trainer uses only the reviewed train split. Within each query, it ranks
`DUPLICATE` above `SIMILAR_BUT_NOT_DUPLICATE`/`REPEAT`, and those above
`UNRELATED`; the latter two grades provide hard negatives. It uses the E5
`query: ` and `passage: ` prefixes, attention-mask mean pooling and L2
normalization, following the
[official model card](https://huggingface.co/intfloat/multilingual-e5-base).
Checkpoint selection uses validation nDCG@5, MRR and Recall@1, in that order.
The frozen test is scored once after selection. The immutable candidate
directory contains weights, tokenizer, `manifest.json`, `metrics.json`,
`training_config.json`, checksums and `MODEL_CARD.md`. `verify` checks every
file and loads the model for a sanity vector. Top-3 review remains `PENDING`,
and no production pointer or Qdrant collection changes. The current customer
CSVs have no appeal text or reviewed relation labels, so this pipeline has only
been exercised on a tiny synthetic test fixture; no customer-quality result or
fine-tuned production embedder exists yet.

## Provisional duplicate threshold

After relation review, evaluate an E5-compatible local model on the same
verified package. Save a policy **before** looking at model scores:

```json
{
  "policy_version": "duplicate-threshold.v1",
  "min_duplicate_count": 30,
  "min_nonduplicate_count": 30,
  "min_predictions": 10,
  "min_precision": 0.95
}
```

The counts and precision above are an example policy, not a validated runtime
setting. Choose them with the reviewer for the actual evaluation. Then run:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/evaluate_duplicate_thresholds.py \
  --dataset data/processed/reviewed-v1 \
  --model /path/to/local/e5-compatible-model \
  --model-version embedder-v1 \
  --policy /path/to/precommitted-duplicate-policy.json \
  --output data/processed/reports/reviewed-v1-duplicate-threshold.json
```

The evaluator selects a threshold only from validation pairs, where only
`DUPLICATE` is positive. `REPEAT` and related but distinct cases are negatives.
It checks the selected threshold once on the frozen test, reports a full
precision/recall curve, confusion counts and hashed false-positive pair IDs, and
records policy, dataset and model checksums. A valid empirical result is still
`PROVISIONAL` until expert review; insufficient support or test precision below
policy never approves a threshold. No ticket text appears in the report.
Repeat relations and temporal windows need their own reviewed calibration;
this command does not choose a repeat threshold. The current pilot is unreviewed
and cannot produce real threshold evidence.

For a separate `REPEAT` time-window audit, provide an immutable temporal source
JSONL with `pair_id`, `query_created_at`, `candidate_created_at`, and
`candidate_resolved_at` (nullable) for **every validation and frozen-test
retrieval pair**. Keep the source and its approved review JSONL outside Git
with the original relation reviews. Each review row has exactly:

```json
{
  "evidence_version": "repeat-temporal-evidence.v1",
  "pair_id": "reviewed_pair_id",
  "source_relation_sha256": "sha256:...",
  "review_evidence_sha256": "sha256:...",
  "temporal_source_sha256": "sha256:...",
  "review_status": "APPROVED",
  "reviewer_id": "reviewer_id",
  "reviewed_at": "2026-09-27T12:00:00+05:00",
  "query_created_at": "2026-09-20T12:00:00+05:00",
  "candidate_created_at": "2026-08-01T12:00:00+05:00",
  "candidate_resolved_at": "2026-09-10T12:00:00+05:00",
  "same_region": true,
  "same_object": true,
  "same_issue": true,
  "same_episode": false,
  "prior_episode_resolved": true
}
```

Use `null` for `candidate_resolved_at` if there is no confirmed resolution.
The reviewer must establish the relation facts and resolution event from case
evidence; a source status or `closed_at` alone is not proof. Timestamps must
include a timezone and the candidate must predate the query. The evaluator
compares the review timestamps with the immutable temporal source, checks each
pair against the approved relation label and review checksums, and rejects
missing or changed evidence. Its
time signal is restricted to same-region retrieved candidates with a confirmed
resolution before the query; it counts different-object/issue false positives
instead of assuming semantic agreement.

Create a policy outside Git with `policy_version="repeat-window.v1"`, sorted
unique positive `window_days` (for example `[7, 30, 90]`), positive
`min_repeat_count`, `min_nonrepeat_count`, `min_predictions`, and
`min_precision` in `(0, 1]`. Choose the policy before looking at the frozen
test. `min_nonrepeat_count` applies to hard negatives: same-region candidates
with a confirmed prior resolution but a non-`REPEAT` label. Then run:

```bash
PYTHONPATH=ml-service .venv/bin/python scripts/evaluate_repeat_windows.py \
  --dataset data/processed/reviewed-v1 \
  --temporal-source /path/to/immutable-temporal-source.jsonl \
  --temporal-evidence /path/to/approved-temporal-evidence.jsonl \
  --policy /path/to/precommitted-repeat-policy.json \
  --output data/processed/reports/reviewed-v1-repeat-windows.json
```

The report chooses a window from validation only and applies it once to the
frozen test. It contains no appeal text or raw pair IDs. Its resolution signal
comes from human-reviewed evidence, so `runtime_window_status` remains
`NOT_APPROVED` even when the test meets the policy. Production use requires a
separately verified runtime resolution source and expert approval. With the
current `PENDING` pilot, there is no empirical `REPEAT` window result.

## Forecast candidate comparison

The daily count runner fits [Prophet](https://facebook.github.io/prophet/docs/quick_start.html)
only on data before each rolling origin and compares it with the weekly
seasonal naive baseline on identical windows:

```bash
MPLCONFIGDIR=/tmp/pulse109-mpl .venv/bin/python scripts/evaluate_forecast_candidates.py \
  /path/to/regional_export.csv \
  --output /tmp/regional-forecast-candidates.json
```

It evaluates 30/60/90-day horizons after at least 365 days of history, reports
MAE, RMSE, WAPE, sMAPE and window counts, and uses lower WAPE with a baseline
tie-break for provisional selection. It returns `INSUFFICIENT_HISTORY` without
invented metrics when a horizon has no full window. Prophet uses fixed weekly
seasonality, no yearly or daily seasonality, and no uncertainty sampling; the
report contains no interval coverage. A single regional backtest does not
qualify a model for runtime use.

New `forecast-candidates.v3` reports also include a high-load-day proxy for
each model on the same windows. For each origin, the threshold is the nearest-rank
90th percentile of observed training-day counts before that origin; a target day
is high load only when its count strictly exceeds the threshold. The report
aggregates TP/FP/FN/TN and precision/recall/F1 for predicted versus observed
high-load days. Undefined ratios are `null`. This is an aggregate-count proxy,
not reviewed peak labels or evidence of staffing capacity; model selection
still uses WAPE only. The checked-in Eastern Kazakhstan report uses the earlier
v2 schema; its v3 proxy requires refitting Prophet on every eligible origin.
The Almaty Region report uses v3, but has no eligible windows and therefore
contains no high-load metrics or Prophet fit.

## Spike exploration on regional counts

The CSV spike runner compares count/ratio and weekday median/MAD rules on
the same daily totals without using future days:

```bash
python3 scripts/evaluate_spike_csv.py /path/to/regional_export.csv \
  --output /tmp/regional-spike-exploration.json
```

Each day uses the preceding eight values for that weekday. Three ratio and
three robust-score thresholds are reported with raw and seven-day-cooldown
alert counts; no threshold is selected. The report contains only aggregate
daily counts, dates and a 20-item `PENDING` review queue. It leaves precision,
recall and detection delay unset because no incident ground truth was supplied.
The unit is one region's total per day, not region × reviewed topic × time;
missing days are unobserved: forecast windows require 365 consecutive observed
training days and an observed target horizon, while spike scoring skips a day
if its count or one of eight same-weekday history counts is missing. This
exploration cannot establish a runtime alert threshold.

## Feedback candidate dataset

`scripts/build_feedback_candidate.py` validates an explicit
`learning-feedback-export.v1` JSONL, excludes frozen test IDs/groups/text and
writes an immutable offline candidate package only after the configured
minimum feedback count. Operator-confirmed topic is the sole training label;
prediction is retained separately. The contract, commands and reviewed Core
export requirements are documented in
[`docs/feedback-candidate-dataset.md`](../../docs/feedback-candidate-dataset.md).

## Offline drift evidence

`scripts/export_drift_snapshot.py` reads one closed PostgreSQL time window and
writes only aggregate counts. It never writes ticket text, operator notes or
ticket IDs. Run it twice for non-overlapping windows with the **same**
classifier model version. An optional complete local tokenizer records actual
model token-length bins; without it, only character-length drift is available.

```bash
DATABASE_URL="$DATABASE_URL" .venv/bin/python scripts/export_drift_snapshot.py \
  --window-start 2026-08-01T00:00:00Z --window-end 2026-09-01T00:00:00Z \
  --model-version classifier_v1 --output /tmp/drift-baseline.json
DATABASE_URL="$DATABASE_URL" .venv/bin/python scripts/export_drift_snapshot.py \
  --window-start 2026-09-01T00:00:00Z --window-end 2026-09-27T00:00:00Z \
  --model-version classifier_v1 --output /tmp/drift-recent.json
.venv/bin/python scripts/evaluate_drift.py \
  --baseline /tmp/drift-baseline.json --recent /tmp/drift-recent.json \
  --policy /path/to/approved-drift-policy.json --output /tmp/drift-report.json
```

The policy contract is `pulse-drift-policy.v1` with positive `min_tickets`,
`min_predictions`, `min_decisions`, plus `max_distribution_tv` and
`max_correction_rate_increase` in `[0,1]`. Set its limits before examining the
recent window. Categorical and binned numeric distributions use total
variation distance. The correction-rate comparison uses first operator
decisions **by decision time** after the selected model's prediction. Low
support returns `INSUFFICIENT_EVIDENCE` or `INSUFFICIENT_GROUND_TRUTH` for that
signal. A drift signal produces `REVIEW_TRIGGER`, never automatic retraining.

The current database stores relation feedback but no ranked retrieval results
captured at feedback time. The report therefore marks retrieval quality drift
`UNAVAILABLE_NO_RANKED_RELEVANCE`; relation counts are context, not Recall@K.
Both CSVs supplied by the customer lack appeal text and classifier predictions,
so they cannot supply these runtime drift snapshots.

## Several classifier challengers

`scripts/compare_classifier_challengers.py` combines at least two existing
production-versus-candidate offline reports and their paired shadow reports.
Repeat each flag in matching order:

```bash
.venv/bin/python scripts/compare_classifier_challengers.py \
  --offline-report /path/to/candidate-a-offline.json \
  --shadow-report /path/to/candidate-a-shadow.json \
  --offline-report /path/to/candidate-b-offline.json \
  --shadow-report /path/to/candidate-b-shadow.json \
  --output /tmp/classifier-challengers.json
```

The combiner requires the same champion artifact and metrics, frozen
evaluation version, sample IDs, policy, fresh window, operator labels and
champion predictions across candidates. A fresh report now includes
`champion_reference_sha256` for that last check; reports without it cannot be
compared safely. The combiner also checks that each input's decision and status
agree with its sample counts and reported regressions. Candidate scores and
regressions stay separate in the output.
No model is selected or promoted automatically. Actual candidate artifacts,
reviewed labels and paired fresh predictions are still required to produce the
input reports.
