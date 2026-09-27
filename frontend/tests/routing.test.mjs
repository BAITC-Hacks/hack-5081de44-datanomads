import assert from 'node:assert/strict'
import test from 'node:test'
import { mapPriority, mapRuleProvenance, manualRuleProvenance } from '../src/routing.ts'

test('preserves the critical priority level across API values', () => {
  assert.equal(mapPriority('critical'), 'Критический')
  assert.equal(mapPriority('Критический'), 'Критический')
  assert.equal(mapPriority('high'), 'Высокий')
  assert.equal(mapPriority('UNKNOWN'), 'Не определён')
})

test('keeps only recognized routing sources and positive rule versions', () => {
  assert.deepEqual(
    mapRuleProvenance({ source: 'OFFICIAL', version: 4, reason: 'Проверенное правило' }, 'fallback'),
    { source: 'OFFICIAL', version: 4, reason: 'Проверенное правило' },
  )
  assert.deepEqual(
    mapRuleProvenance({ source: 'LABEL_HISTORY', version: 2, reason: 'Историческое значение' }, 'fallback'),
    { source: 'LABEL_HISTORY', version: 2, reason: 'Историческое значение' },
  )
  assert.deepEqual(
    mapRuleProvenance({ source: 'MODEL', version: 0, reason: '  ' }, 'Источник не сохранён'),
    { source: 'MANUAL', version: null, reason: 'Источник не сохранён' },
  )
  assert.deepEqual(manualRuleProvenance('Ручная проверка'), {
    source: 'MANUAL',
    version: null,
    reason: 'Ручная проверка',
  })
})
