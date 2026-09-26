import assert from 'node:assert/strict'
import { test } from 'node:test'

import { languageLabel, languageReviewNotice, mapLanguage } from '../src/language.ts'

test('maps all supported language states without coercing uncertainty to Russian', () => {
  const cases = [
    ['RU', 'RU'],
    ['ru', 'RU'],
    ['RUS', 'RU'],
    ['KZ', 'KZ'],
    ['kk', 'KZ'],
    ['KAZ', 'KZ'],
    ['MIXED', 'MIXED'],
    ['UNKNOWN', 'UNKNOWN'],
    ['', 'UNKNOWN'],
    ['EN', 'UNKNOWN'],
  ]

  for (const [input, expected] of cases) {
    assert.equal(mapLanguage(input), expected, `maps ${JSON.stringify(input)}`)
  }
  assert.equal(mapLanguage(undefined), 'UNKNOWN')
})

test('labels all four states distinctly and asks for manual review for uncertainty', () => {
  assert.deepEqual(
    ['RU', 'KZ', 'MIXED', 'UNKNOWN'].map((language) => languageLabel(language)),
    ['Русский (RU)', 'Казахский (KZ)', 'Смешанный (MIXED)', 'Не определён (UNKNOWN)'],
  )
  assert.equal(languageReviewNotice('RU'), undefined)
  assert.equal(languageReviewNotice('KZ'), undefined)
  assert.match(languageReviewNotice('MIXED'), /смешанный язык/i)
  assert.match(languageReviewNotice('UNKNOWN'), /не определён/i)
})
