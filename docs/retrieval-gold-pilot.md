# Retrieval relations: synthetic pilot and review

[`pilot_relations.jsonl`](../data/sdg/pilot_relations.jsonl) contains 15
**invented** query–candidate pairs in three relation groups. Each pair has
appeal texts and separate case context for the reviewer. The file contains no
approved relation labels and is not a gold set. None of its texts comes from
the customer's regional CSV exports.

The pilot exercises close wording for different objects or regions, another
issue at the same object, a neighboring topic, a later episode after repair,
and a possible area-wide incident without confirmed common cause. Those cases
must be judged from context, not lexical similarity alone.

## Relation meanings

| Label | Required evidence |
| --- | --- |
| `DUPLICATE` | Same region, object, issue and still-open episode. Two reports of one underlying problem. |
| `REPEAT` | Same region, object and issue, but a new episode after the earlier one was resolved. |
| `SIMILAR_BUT_NOT_DUPLICATE` | Related issue or context worth retrieving, with a material difference that prevents duplicate status. |
| `UNRELATED` | No meaningful retrieval relevance to the query. |

The reviewer records `same_region`, `same_object`, `same_issue`,
`same_episode`, and `prior_episode_resolved` explicitly. The script checks
the required `DUPLICATE` and `REPEAT` combinations, but these booleans are
human judgments, not evidence inferred by code. Ambiguous facts remain
`DEFERRED` or are rejected with a reason.

## Review workflow

Prepare a queue outside Git:

```bash
python3 scripts/review_retrieval_relations.py prepare \
  --source data/sdg/pilot_relations.jsonl \
  --output data/reviews/retrieval-pilot.jsonl
```

Every row starts `PENDING`. The `source` object and its checksum must stay
unchanged. To approve, enter `decision="APPROVED"`, a relation label,
`reviewer_id`, a timezone-aware `reviewed_at`, `review_reason="VERIFIED"`,
all four checks as `true`, and the five relation facts. Use `REJECTED` with
a reason for a bad pair. `DEFERRED` records unresolved evidence and cannot
enter an export.

```bash
python3 scripts/review_retrieval_relations.py validate \
  data/reviews/retrieval-pilot.jsonl \
  --source data/sdg/pilot_relations.jsonl
python3 scripts/review_retrieval_relations.py export \
  data/reviews/retrieval-pilot.jsonl \
  --source data/sdg/pilot_relations.jsonl \
  --output data/reviews/retrieval-pilot-approved.jsonl
```

Export requires a decision for **every** source pair and at least two
approved candidates for each retained query. The approved set must include
`DUPLICATE`, `SIMILAR_BUT_NOT_DUPLICATE` and `UNRELATED`. It writes only approved rows
in the `RetrievalPair` format. `source_relation_sha256` points to the
versioned relation source; `review_evidence_sha256` points to the complete
review queue, which must be retained. Rejected rows do not receive labels.

Three relation groups are enough to exercise the split contract, but not to
claim retrieval quality or production coverage. A final gold set needs more
independent incidents, complete RU/KZ/MIXED coverage, reviewed evaluation
pairs and an expert review of top-3 results from the evaluated model. No
retrieval metric is reported from this unreviewed pilot.

## Expanded review seed

[`pilot_relations_v2.jsonl`](../data/sdg/pilot_relations_v2.jsonl) keeps the
original 15 pairs and adds 30 invented pairs. Its nine independent relation
groups cover waste containers, transport, heating, a possible area-wide water
incident, standing water on a road, and a fallen tree in addition to the
original lighting, road damage, and water-pressure cases. The added pairs
exercise identical wording in different synthetic regions, different objects
at one location, different issues at one object, neighboring topics, and a new
episode after repair stated in the synthetic context. The possible area-wide
incident has an explicitly unknown causal link; the reviewer can defer it.
One group includes MIXED text, but this small seed is not a balanced language
or topic sample.

The previous source and `data/reviews/retrieval-pilot.jsonl` remain intact.
Prepare and validate a separate v2 queue:

```bash
python3 scripts/review_retrieval_relations.py prepare \
  --source data/sdg/pilot_relations_v2.jsonl \
  --output data/reviews/retrieval-pilot-v2.jsonl
python3 scripts/review_retrieval_relations.py validate \
  data/reviews/retrieval-pilot-v2.jsonl \
  --source data/sdg/pilot_relations_v2.jsonl
```

Every v2 row is still `PENDING`; no relation label or quality metric is
approved by adding these synthetic cases. An approved package still requires
human review of every pair and a held-out evaluation set with enough
independent reviewed incidents.
