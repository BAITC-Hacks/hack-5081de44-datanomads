# Offline Data/ML contracts

Use Python 3.11.15 for the training environment. The runtime service installs
`requirements.txt` separately; training packages stay out of its default image.

```bash
uv venv --python 3.11.15 .venv
uv pip sync --python .venv/bin/python ml-service/training/requirements.lock
PYTHONPATH=ml-service .venv/bin/python -m unittest discover \
  -s ml-service/training/tests -v
```

The lock resolves 49 packages for Python 3.11.15 on Linux x86_64. Direct
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
