import assert from 'node:assert/strict'
import test from 'node:test'
import { buildQueryIntentChartOption } from '../src/query-intent-view.ts'

function result(overrides) {
  return {
    intent: 'trend',
    series: [{ date: '2026-09-01', label: '2026-09-01', count: 5 }],
    chart: { type: 'line', x: 'date', y: 'count', title: 'Обращения', forecast_start: null },
    forecast_start: null,
    ...overrides,
  }
}

test('builds a chart from the returned grouping labels', () => {
  const option = buildQueryIntentChartOption(result({
    intent: 'compare_regions',
    series: [{ label: 'Север', count: 12 }],
    chart: { type: 'bar', x: 'label', y: 'count', title: 'Регионы', forecast_start: null },
  }))

  assert.deepEqual(option.xAxis.data, ['Север'])
  assert.equal(option.series[0].type, 'bar')
  assert.deepEqual(option.series[0].data, [12])
})

test('compares spike counts with the previous period in a second series', () => {
  const option = buildQueryIntentChartOption(result({
    intent: 'spikes',
    series: [{ date: '2026-09-01', label: '2026-W36', count: 9, baseline: 4, is_spike: true }],
  }))

  assert.equal(option.series.length, 2)
  assert.equal(option.series[1].name, 'Предыдущий период')
  assert.deepEqual(option.series[1].data, [4])
})

test('marks the forecast boundary and keeps observed and predicted lines separate', () => {
  const option = buildQueryIntentChartOption(result({
    intent: 'forecast',
    forecast_start: '2026-09-02',
    series: [
      { date: '2026-09-01', label: '2026-09-01', count: 5, segment: 'history' },
      { date: '2026-09-02', label: '2026-09-02', count: 6, segment: 'forecast' },
    ],
  }))

  assert.deepEqual(option.series[0].data, [5, null])
  assert.deepEqual(option.series[1].data, [null, 6])
  assert.equal(option.series[0].markLine.data[0].xAxis, '2026-09-02')
})
