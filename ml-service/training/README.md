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
evaluation version. The model schemas are for future real training outputs;
the existing deterministic baseline and synthetic classifier demo are not
retroactively promoted to these contracts.

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
