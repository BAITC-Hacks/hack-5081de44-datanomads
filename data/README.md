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

The initial taxonomy has 20 Kazakhstan region IDs and 16 candidate topics. The
source direction stays in the restricted raw export; its PII-minimized form is
preserved in `topic_raw`. An unmapped direction gets
`topic_id=unknown` until a reviewed source mapping is added. Only exact
canonical IDs, full RU/KZ topic names and the four explicit legacy IDs receive
a topic hint automatically; generic words and partial matches remain unknown.
A hint is not an approved training label.

## Bad rows and privacy

Rows are never silently dropped. They are emitted to JSONL quarantine with one
of these reasons:

`BAD_CSV_STRUCTURE`, `INVALID_DATE`, `MISSING_REQUIRED_FIELD`, `UNKNOWN_SCHEMA`,
`PII_REVIEW`, `INVALID_VALUE`.

Quarantine rows keep only column/nonempty counts, reason and row number; raw
headers and values are not copied to reports or Core. Known phones, IINs,
e-mail addresses, labeled names/addresses and attachments are replaced with
typed tokens or removed from normalized text. A residual PII match is
quarantined for source-specific review. Optional text fields use the same
minimization, and precise coordinates are omitted from the safe layer.

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
4,480 train, 640 validation and 1,280 test rows, balanced across 16 topics
and both languages. Exact text duplicates and obvious PII patterns are rejected.
All paraphrases of one scenario stay in one split.

```bash
python3 scripts/generate_synthetic_classifier.py
```

Files appear in ignored `data/sdg/generated/classifier_v2/`, with a manifest,
checksums and `review_status=PENDING`. This is a reproducible **synthetic
candidate**, not a reviewed gold set or evidence of performance on real 109
appeals. Generic openers and requests do not add an unsupported observation
time or a service action. The previous `classifier_v1` generator added an
unprovided time context and its local outputs must not be used as training
evidence; the new version does not overwrite that directory. The 6,400 rows
still come from only 160 base situations; increasing their
number further without adding distinct situations is unlikely to help. Before
choosing a model, compare 1k/2k/4k training subsets on the same held-out
scenario groups, review a sample of RU/KZ text and create an independent
evaluation set. This corpus is for classification; retrieval training still
needs separately defined positive and hard-negative pairs.

The same command also writes `challenges.jsonl` from the versioned
`sdg/classifier_challenges.tsv` bank. Its RU/KZ/MIXED records propose
`UNKNOWN`, `OTHER`, or `NEEDS_REVIEW` for five difficult topic boundaries,
an out-of-taxonomy situation, and a message without enough subject detail.
They have `review_status=PENDING`, no confirmed `topic_id`, and
`approved_for_training=false`. The manifest records their source/output
checksums separately from the 6,400 class-labeled rows. These proposals
require human review before they can serve as challenge evaluation evidence;
they are never included in train, validation or test splits by this generator.

Import a real (uncommitted) source export:

```bash
python scripts/import_tickets.py \
  --source iKOMEK109 export.csv \
  --output data/processed/ikomek109.jsonl \
  --quarantine data/quarantine/ikomek109.jsonl
```

The importer exits with status `2` when any row is quarantined, so a job cannot
mistake a partial load for a clean import. `scripts/data_audit.py` produces a
quality report with coverage, duplicates, dates and distributions. For a raw
source export, pass `--source` so the report uses the relevant importer. Add
`--synthetic` for generated fixtures:

```bash
python3 scripts/data_audit.py data/synthetic_raw/v1/ikomek109/primary.csv \
  --source iKOMEK109 --synthetic --output /tmp/pulse109-source-audit.json
```

The report contains aggregate counts and a source checksum, never source row
values. For a real local export, use `--real` with `--source`. The origin flag
is required so synthetic reports cannot silently look like real evidence.
Alias and label semantics remain unverified until a real export is inspected
and reviewed.

Build the normalized package from all seven synthetic fixtures after the
[synthetic quality report](reports/synthetic_source_quality_v1.json) has been
reproduced. Choose a new output directory for each run:

```bash
python3 scripts/build_synthetic_source_corpus.py \
  --output-dir /tmp/pulse109-synthetic-source-verify-v1
```

The current local package is under ignored `data/processed/synthetic_source_v1/` with
`tickets.jsonl`, redacted `quarantine.jsonl`, `audit.json`, and `manifest.json`.
Its [versioned manifest](manifests/synthetic-source-normalized-v1.json) records
checksums and `approved_for_training=false`. It exercises the seven importers
and UnifiedTicket privacy boundary; it is not a customer corpus or a reviewed
classifier/retrieval dataset.
The normalized audit checks ticket IDs within each `source_system`, matching
the source-local meaning of `external_ticket_id`.

## PostgreSQL

The Core migration runner applies all numbered files in `migrations/` in order.
PostgreSQL remains source of truth for
normalized tickets, predictions, operator decisions and controlled-learning
state. Qdrant payloads contain only `ticket_id`, `region_id`,
`topic_id` and `created_at`.
