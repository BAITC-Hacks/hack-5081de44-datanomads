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
and `RDJardem3.0`. Each adapter only owns source-header aliases. Validation,
normalization and privacy behavior is shared.

The initial taxonomy has 20 Kazakhstan region IDs and 16 candidate topics. A
raw direction is preserved in `topic_raw`; an unmapped direction gets
`topic_id=unknown` until a reviewed source mapping is added.

## Bad rows and privacy

Rows are never silently dropped. They are emitted to JSONL quarantine with one
of these reasons:

`BAD_CSV_STRUCTURE`, `INVALID_DATE`, `MISSING_REQUIRED_FIELD`, `UNKNOWN_SCHEMA`,
`PII_REVIEW`, `INVALID_VALUE`.

Quarantine snapshots are themselves masked. Known phones, IINs, e-mail
addresses, labeled names/addresses and attachments are replaced with typed
tokens or removed. A residual PII match is quarantined for source-specific
review. The normalized layer does not contain name, phone, IIN, attachment
contents or exact addresses.

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
