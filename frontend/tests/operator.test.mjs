import assert from 'node:assert/strict'
import test from 'node:test'
import { combineRelatedCandidates, mapTicketChannel, matchingTicketFactors, topRelatedCandidates } from '../src/operator.ts'

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
    { ticket_id: 'ticket-a', score: 0.72, relation: 'similar' },
    { ticket_id: 'ticket-b', score: 0.9, relation: 'similar' },
    { ticket_id: 'ticket-c', score: 0.83, relation: 'similar' },
    { ticket_id: 'ticket-d', score: 0.61, relation: 'similar' },
  ])

  assert.deepEqual(topRelatedCandidates(candidates).map((candidate) => candidate.ticket_id), [
    'ticket-b',
    'ticket-c',
    'ticket-a',
  ])
})

test('reports only exact topic and region matches as similarity factors', () => {
  assert.deepEqual(matchingTicketFactors('TOPIC-WATER', 'R01', 'topic-water', 'r01'), [
    'Совпадает тема',
    'Совпадает регион',
  ])
  assert.deepEqual(matchingTicketFactors('UNKNOWN', 'R01', 'UNKNOWN', 'R02'), [])
})
