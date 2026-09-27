import assert from 'node:assert/strict'
import test from 'node:test'
import { combineRelatedCandidates, formatDecisionTime, formatRuntimeRate, mapRelatedFactors, mapTicketChannel, matchingTicketFactors, topRelatedCandidates } from '../src/operator.ts'

test('maps recognized sources and makes unknown sources visible', () => {
  assert.equal(mapTicketChannel('mobile'), 'Мобильное приложение')
  assert.equal(mapTicketChannel('CALL_CENTER'), 'Call-центр')
  assert.equal(mapTicketChannel('e-gov'), 'eGov')
  assert.equal(mapTicketChannel(''), 'Не указан')
  assert.equal(mapTicketChannel(undefined), 'Не указан')
})

test('merges related and duplicate/repeat candidates without losing their candidate types', () => {
  const result = combineRelatedCandidates(
    [
      { ticket_id: 'ticket-a', score: 0.72, relation: 'similar' },
      { ticket_id: 'ticket-b', score: 0.8, relation: 'similar' },
    ],
    [
      { ticket_id: 'ticket-a', score: 0.91, relation: 'similar' },
      { ticket_id: 'ticket-c', score: 0.84, relation: 'duplicate' },
    ],
    [
      { ticket_id: 'ticket-a', score: 0.86, relation: 'repeat' },
    ],
  )

  assert.deepEqual(result, [
    { ticket_id: 'ticket-a', score: 0.91, relation: 'repeat', candidateTypes: ['similar', 'duplicate', 'repeat'] },
    { ticket_id: 'ticket-c', score: 0.84, relation: 'duplicate', candidateTypes: ['duplicate'] },
    { ticket_id: 'ticket-b', score: 0.8, relation: 'similar', candidateTypes: ['similar'] },
  ])
})

test('keeps only the three highest-scoring related tickets by default', () => {
  const candidates = combineRelatedCandidates([
    { ticket_id: 'ticket-a', score: 0.77, relation: 'similar' },
    { ticket_id: 'ticket-b', score: 0.9, relation: 'similar' },
    { ticket_id: 'ticket-c', score: 0.83, relation: 'similar' },
    { ticket_id: 'ticket-d', score: 0.79, relation: 'similar' },
    { ticket_id: 'ticket-e', score: 0.61, relation: 'similar' },
  ])

  assert.deepEqual(topRelatedCandidates(candidates).map((candidate) => candidate.ticket_id), [
    'ticket-b',
    'ticket-c',
    'ticket-d',
  ])
})

test('applies the candidate-specific threshold and preserves the strongest relation metadata', () => {
  const merged = combineRelatedCandidates(
    [{
      ticket_id: 'ticket-duplicate',
      score: 0.95,
      relation: 'repeat',
      suggestion: { score: 0.95, threshold: 0.78, rule_version: 'rules-v1', model_version: 'embedder-v1', distance_metric: 'Cosine' },
    }],
    [{
      ticket_id: 'ticket-duplicate',
      score: 0.95,
      relation: 'duplicate',
      suggestion: { score: 0.95, threshold: 0.9, rule_version: 'rules-v1', model_version: 'embedder-v1', distance_metric: 'Cosine' },
    }],
  )

  assert.equal(merged[0].relation, 'duplicate')
  assert.equal(merged[0].suggestion?.threshold, 0.9)
  assert.deepEqual(topRelatedCandidates([
    ...merged,
    { ticket_id: 'weak-duplicate', score: 0.89, relation: 'duplicate', candidateTypes: ['duplicate'], suggestion: { score: 0.89, threshold: 0.9, rule_version: 'rules-v1', model_version: 'embedder-v1', distance_metric: 'Cosine' } },
  ]).map((candidate) => candidate.ticket_id), ['ticket-duplicate'])
})

test('reports only exact topic and region matches as similarity factors', () => {
  assert.deepEqual(matchingTicketFactors('TOPIC-WATER', 'R01', 'topic-water', 'r01'), [
    'Совпадает тема',
    'Совпадает регион',
  ])
  assert.deepEqual(matchingTicketFactors('UNKNOWN', 'R01', 'UNKNOWN', 'R02'), [])
})

test('maps only supported relation factors to operator-facing explanations', () => {
  assert.deepEqual(mapRelatedFactors(['topic_match', 'within_30_days', 'topic_match', 'object_match']), [
    'Совпадает тема',
    'В пределах 30 дней',
  ])
})

test('formats runtime rates and leaves missing event metrics unknown', () => {
  assert.equal(formatRuntimeRate(0.375), '38%')
  assert.equal(formatRuntimeRate(0), '0%')
  assert.equal(formatRuntimeRate(null), '—')
  assert.equal(formatRuntimeRate(1.2), '—')
})

test('formats decision time without inventing a zero for missing samples', () => {
  assert.equal(formatDecisionTime(42.4), '42 мин')
  assert.equal(formatDecisionTime(91), '1 ч 31 мин')
  assert.equal(formatDecisionTime(null), '—')
  assert.equal(formatDecisionTime(undefined, 'нет решений'), 'нет решений')
})
