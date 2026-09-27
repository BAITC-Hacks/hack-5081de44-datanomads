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
reference. `PENDING` demo candidates are rejected.

The synthetic retrieval pilot source and its manual review/export workflow are
documented in [`docs/retrieval-gold-pilot.md`](../../docs/retrieval-gold-pilot.md).
Use the unchanged source file as `--relation-source` when the human-approved
export is eventually passed to the dataset builder. The pilot has no approved
labels yet and is insufficient for quality claims.

```bash
.venv/bin/python scripts/build_training_dataset.py \
  --classifier /path/to/reviewed_classifier.jsonl \
  --retrieval /path/to/reviewed_relations.jsonl \
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
  --retrieval /path/to/new_reviewed_relations.jsonl \
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
Current reviewed packages are synthetic, so the comparison cannot establish
performance on actual citizen appeals. The feedback candidate trainer and a
production model pointer are still required to run a real controlled-learning
comparison.

## Fresh shadow evaluation

`scripts/evaluate_classifier_shadow.py` accepts a PII-free
`classifier-shadow-input.v1` JSONL. Each row has one ticket/feedback ID, one
operator-confirmed topic and **both** production and candidate predictions for
that ticket. It also carries cycle/model versions, source dataset version,
synthetic origin, decision time and `validation_status=VALID`. Ticket text is
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
single sample-ID checksum. It records `blind_ab_enabled=false` and no
preference score because the current UI has no blind A/B. Too few total, real
or critical-topic samples give `INSUFFICIENT_EVIDENCE`; an adequately sized
report with a critical or global correction-rate regression gets a `NO_GO`
decision. Only a `VALID` report without regressions can contribute to
promotion evidence.
Runtime capture/export of candidate shadow predictions is still required;
this evaluator does not manufacture that input from existing customer CSVs.

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
missing days are treated as zero pending source-quality verification. This
exploration cannot establish a runtime alert threshold.

## Feedback candidate dataset

`scripts/build_feedback_candidate.py` validates an explicit
`learning-feedback-export.v1` JSONL, excludes frozen test IDs/groups/text and
writes an immutable offline candidate package only after the configured
minimum feedback count. Operator-confirmed topic is the sole training label;
prediction is retained separately. Contract, command and current Core export
gap are documented in [`docs/feedback-candidate-dataset.md`](../../docs/feedback-candidate-dataset.md).
