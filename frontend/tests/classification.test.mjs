import assert from 'node:assert/strict'
import test from 'node:test'
import {
  classificationAlternatives,
  confidenceStateNotice,
  normalizeConfidenceState,
} from '../src/classification.ts'

test('normalizes classifier states and legacy backend labels', () => {
  assert.equal(normalizeConfidenceState('CONFIDENT', 0.79), 'CONFIDENT')
  assert.equal(normalizeConfidenceState('uncertain', 0.61), 'UNCERTAIN')
  assert.equal(normalizeConfidenceState('LOW_CONFIDENCE', 0.3), 'LOW_CONFIDENCE')
  assert.equal(normalizeConfidenceState('high', 0.86), 'CONFIDENT')
  assert.equal(normalizeConfidenceState('medium', 0.72), 'UNCERTAIN')
  assert.equal(normalizeConfidenceState('low', 0.4), 'LOW_CONFIDENCE')
})

test('uses legacy confidence thresholds only when the state is missing', () => {
  assert.equal(normalizeConfidenceState(undefined, 0.85), 'CONFIDENT')
  assert.equal(normalizeConfidenceState(undefined, 0.7), 'UNCERTAIN')
  assert.equal(normalizeConfidenceState(undefined, 0.69), 'LOW_CONFIDENCE')
  assert.equal(normalizeConfidenceState('FUTURE_STATE', 0.99), 'UNAVAILABLE')
  assert.equal(normalizeConfidenceState('CONFIDENT', 0.99, false), 'UNAVAILABLE')
})

test('shows only two non-winning alternatives for uncertain predictions', () => {
  const alternatives = [
    { topic_id: 'water', topic_label: 'Вода' },
    { topic_id: 'roads', topic_label: 'Дороги' },
    { topic_id: 'housing', topic_label: 'Жильё' },
    { topic_id: 'health', topic_label: 'Здоровье' },
  ]

  assert.deepEqual(classificationAlternatives('UNCERTAIN', 'water', alternatives), alternatives.slice(1, 3))
  assert.deepEqual(classificationAlternatives('CONFIDENT', 'water', alternatives), [])
  assert.deepEqual(classificationAlternatives('LOW_CONFIDENCE', 'water', alternatives), [])
})

test('explains review and manual-selection states in plain language', () => {
  assert.match(confidenceStateNotice('UNCERTAIN'), /проверить/i)
  assert.match(confidenceStateNotice('LOW_CONFIDENCE'), /вручную/i)
  assert.match(confidenceStateNotice('UNAVAILABLE'), /недоступна/i)
})
