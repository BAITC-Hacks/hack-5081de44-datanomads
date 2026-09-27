import assert from 'node:assert/strict'
import test from 'node:test'
import { combineRelatedCandidates, mapTicketChannel } from '../src/operator.ts'

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
