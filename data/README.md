# Data Foundation

`data/` is the safe boundary between regional 109 exports and Pulse Core. The
pipeline is intentionally dependency-free:

```text
source export
  -> source-specific importer
  -> canonical field aliases
  -> date/required-field validation
  -> RU/KZ normalization and taxonomy mapping
  -> PII minimization
  -> UnifiedTicket JSONL
  -> PostgreSQL tickets
```

The authoritative contract is implemented by
[`schemas/unified_ticket.py`](schemas/unified_ticket.py) and documented as
[`schemas/unified_ticket.schema.json`](schemas/unified_ticket.schema.json).
`original_text` in a `UnifiedTicket` is already minimized text; raw exports
must stay outside Git and are never copied to the normalized layer.

## Sources and taxonomy

Adapters are available for the seven source systems named by the product plan:
`iKOMEK109`, `АС Комек 109`, `AIKEY`, `Открытый город`, `E-SEP.SU`, `ЕКЦ-109`
and `RDJardem3.0`. Their aliases are `SYNTHETIC_TEST_ONLY` hypotheses, not
verified production schemas. Each adapter owns exact source-header aliases and
its CSV delimiter. Validation, normalization and privacy behavior is shared.
An import reports the profile version, status and SHA-256 schema fingerprint.
Missing or ambiguous required headers are quarantined as `UNKNOWN_SCHEMA`.

Generate synthetic raw exports and run each one through its importer:

```bash
python3 scripts/generate_synthetic_sources.py --check
```

The seven generated source directories and checksum manifest live under ignored
`data/synthetic_raw/v1/`. Each source has primary and alternate aliases, valid
and invalid rows, an unclosed quoted CSV row, and an unknown schema. The
manifest marks them `synthetic=true`; they are importer fixtures, not real
source evidence or training data. A real export requires schema inspection and
an explicit verified profile before its aliases can be trusted.

The initial taxonomy has 20 Kazakhstan region IDs and 16 candidate topics. A
raw direction is preserved in `topic_raw`; an unmapped direction gets
`topic_id=unknown` until a reviewed source mapping is added.

## Bad rows and privacy

Rows are never silently dropped. They are emitted to JSONL quarantine with one
of these reasons:

`BAD_CSV_STRUCTURE`, `INVALID_DATE`, `MISSING_REQUIRED_FIELD`, `UNKNOWN_SCHEMA`,
`PII_REVIEW`, `INVALID_VALUE`.

Quarantine rows keep only column/nonempty counts, reason and row number; raw
headers and values are not copied to reports or Core. Known phones, IINs,
e-mail addresses, labeled names/addresses and attachments are replaced with
typed tokens or removed from normalized text. A residual PII match is
quarantined for source-specific review.

## Deterministic demo dataset

The committed fixture is synthetic and must not be presented as operational
metrics. It contains 160 normalized rows, both `RU` and `KZ`, all 20 regions,
all 16 topics and all seven source IDs. Its immutable manifest is
`manifests/demo-2026-09-21.json`.

Regenerate and verify it:

```bash
python scripts/generate_demo_data.py --check
python -m unittest discover -s data/tests -v
```

## Synthetic classifier candidate

`scripts/generate_synthetic_classifier.py` builds a separate classifier corpus
from 160 manually written RU/KZ scenarios in `sdg/classifier_scenarios.tsv`:
20,000 train, 2,000 validation and 4,000 test rows, balanced across 16 topics
and both languages. Exact text duplicates and obvious PII patterns are rejected.
All paraphrases of one scenario stay in one split.

```bash
python3 scripts/generate_synthetic_classifier.py
```

Files appear in ignored `data/sdg/generated/classifier_v1/`, with a manifest,
checksums and `review_status=PENDING`. This is a reproducible **synthetic
candidate**, not a reviewed gold set or evidence of performance on real 109
appeals. The 26,000 rows come from only 160 base situations; increasing their
number further without adding distinct situations is unlikely to help. Before
choosing a model, compare 2k/5k/20k training subsets on the same held-out
scenario groups, review a sample of RU/KZ text and create an independent
evaluation set. This corpus is for classification; retrieval training still
needs separately defined positive and hard-negative pairs.

Import a real (uncommitted) source export:

```bash
python scripts/import_tickets.py \
  --source iKOMEK109 export.csv \
  --output data/processed/ikomek109.jsonl \
  --quarantine data/quarantine/ikomek109.jsonl
```

The importer exits with status `2` when any row is quarantined, so a job cannot
mistake a partial load for a clean import. `scripts/data_audit.py` produces a
quality report with coverage, duplicates, dates and distributions.

## PostgreSQL

The Core migration runner applies all numbered files in `migrations/` in order.
PostgreSQL remains source of truth for
normalized tickets, predictions, operator decisions and controlled-learning
state. Qdrant payloads contain only `ticket_id`, `region_id`,
`topic_id` and `created_at`.
