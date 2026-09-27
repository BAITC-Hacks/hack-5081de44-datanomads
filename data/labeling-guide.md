# Taxonomy and label review

## Canonical IDs

The 16 current topic candidates use the IDs in `schemas/taxonomy.py`.
`waste_management` and `environment` are canonical; `waste` and `ecology`
are compatibility aliases, not additional classes. Production topics require
review of volume, RU/KZ coverage, ambiguity and overlap before selection.
The Almaty metadata catalog alone cannot prove any of these text-classifier
criteria.

`UNKNOWN` means the ticket lacks enough information to choose a topic.
`OTHER` requires a reviewed situation outside the active taxonomy; it is not
a catch-all for an unfamiliar source label. `NEEDS_REVIEW` marks an ambiguous
case or conflicting evidence. None of these states silently becomes a
confident training label. The current `UnifiedTicket.topic_id=unknown` is also
used as an ingestion fallback for an unverified source label; it carries
`needs_review=true` and must not be treated as a reviewed `UNKNOWN` decision.
The runtime distinction is recorded in the handoff contract.

## Review a source mapping

Keep the original source fields. For the Almaty catalog, `source_category`
and `source_service` form a composite key. The latter describes an issue
type, not a confirmed service operator. Its source system is unverified.

Generate the review queue outside Git:

```bash
python3 scripts/review_taxonomy.py prepare \
  --output data/reviews/almaty_2025.jsonl
python3 scripts/review_taxonomy.py validate data/reviews/almaty_2025.jsonl
```

Every proposal starts `PENDING`, even if its suggested topic looks obvious.
To approve a mapping, a human reviewer records a verified canonical source
system, `source_profile_verified=true`, a canonical topic, optional subtopic,
reviewer ID, timezone-aware review timestamp and SHA-256 evidence reference.
`review_reason` must be `SOURCE_CONTEXT_VERIFIED`. Use `REJECTED` or
`DEFERRED` with an appropriate reason when evidence is insufficient.
Source labels, counts and suggestions are checked against the catalog and
cannot be changed in the review queue.

```bash
python3 scripts/review_taxonomy.py export data/reviews/almaty_2025.jsonl \
  --output data/reviews/almaty_2025_approved.json
```

This export is a metadata mapping only. The catalog has no original citizen
text or confirmed executor, so `approved_for_training` and
`approved_for_routing` remain false even after mapping review. The runtime
handoff for its composite key is in `contracts/taxonomy_mapping_handoff.json`.

## Review synthetic text

Review each scenario's facts and each generated RU, KZ and MIXED variant.
Keep all variants of one scenario in one split. Reject text that invents an
address, cause, responsible service, resolution, deadline, identity or other
fact absent from the scenario. Check short, conversational and neutral styles,
moderate errors, and the hard boundaries `street_lighting/electricity`,
`buildings/street_lighting`, `roads/public_transport`,
`wastewater/road standing water`, and `landscaping/roads`.

Generated `PENDING` records cannot enter the final training dataset. A human
review must record provenance and a decision before any `APPROVED` variant is
eligible. The existing synthetic classifier demo remains a pending candidate,
not a reviewed gold set.

`sdg/classifier_challenges.tsv` provides separate proposed `UNKNOWN`, `OTHER`
and `NEEDS_REVIEW` cases in RU/KZ/MIXED. Review the text and candidate topics
independently; its proposed decisions are not ground truth. Generated
`challenges.jsonl` has no confirmed `topic_id` and is excluded from the
class-labeled train/validation/test files. Use it for evaluation only after a
documented human review and a frozen evaluation version.

For NeMo pilot candidates, use `scripts/review_synthetic_classifier.py` as
described in `docs/nemo-sdg-pilot.md`. The queue binds each decision to the
unchanged candidate, source scenario facts and both input checksums. Approval
requires explicit positive checks for facts, label, language and style. Keep
the human review queue as evidence for its SHA-256 reference in the exported
classifier rows. These synthetic texts are not original citizen appeals.

## Review retrieval relations

Retrieval relevance is a separate judgment from the classifier topic. Use the
relation meanings and manual queue in `docs/retrieval-gold-pilot.md` for the
synthetic pilot. Keep all candidates for one query in a single relation group;
record a decision for every pair before export so metrics never treat an
unreviewed candidate as irrelevant. The three-group pilot is `PENDING`, not a
human-reviewed gold set.
