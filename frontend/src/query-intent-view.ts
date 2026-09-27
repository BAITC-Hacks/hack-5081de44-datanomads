import type { EChartsOption } from 'echarts'
import type { QueryIntentResult } from './api/client'

export function buildQueryIntentChartOption(result: QueryIntentResult): EChartsOption | null {
  const points = result.series ?? []
  if (points.length === 0) return null

  const xKey = result.chart?.x ?? 'label'
  const labels = points.map((point) => String(point[xKey] ?? point.label ?? point.date ?? ''))
  const forecast = result.intent === 'forecast'
  const spikes = result.intent === 'spikes'
  const chartType = result.chart?.type ?? (forecast ? 'line' : 'bar')
  const series = forecast
    ? [
      {
        name: 'История',
        type: 'line' as const,
        data: points.map((point) => point.segment === 'history' ? point.count ?? null : null),
        showSymbol: false,
        connectNulls: false,
        lineStyle: { width: 2, color: '#a7d9ff' },
      },
      {
        name: 'Прогноз',
        type: 'line' as const,
        data: points.map((point) => point.segment === 'forecast' ? point.count ?? null : null),
        showSymbol: false,
        connectNulls: false,
        lineStyle: { width: 2, color: '#8cf0c8', type: 'dashed' as const },
      },
    ]
    : spikes
      ? [
        {
          name: 'Обращения',
          type: 'line' as const,
          data: points.map((point) => point.count ?? 0),
          showSymbol: false,
          lineStyle: { width: 2, color: '#8cf0c8' },
        },
        {
          name: 'Предыдущий период',
          type: 'line' as const,
          data: points.map((point) => point.baseline ?? null),
          showSymbol: false,
          lineStyle: { width: 1, color: '#ffc875', type: 'dashed' as const },
        },
      ]
      : [
      {
        name: result.chart?.title ?? 'Обращения',
        type: chartType,
        data: points.map((point) => point.count ?? 0),
        showSymbol: false,
        barMaxWidth: 24,
        lineStyle: { width: 2, color: '#8cf0c8' },
        itemStyle: { color: '#8cf0c8' },
      },
    ]

  const forecastStart = result.forecast_start
  const renderedSeries = forecast && forecastStart
    ? series.map((item, index) => index === 0
      ? {
        ...item,
        markLine: {
          silent: true,
          symbol: 'none',
          lineStyle: { type: 'dashed' as const, color: '#ffc875' },
          label: { formatter: 'Начало прогноза', color: '#ffc875' },
          data: [{ xAxis: forecastStart }],
        },
      }
      : item)
    : series

  return {
    animation: false,
    tooltip: { trigger: 'axis' },
    legend: forecast || spikes
      ? { data: forecast ? ['История', 'Прогноз'] : ['Обращения', 'Предыдущий период'], top: 0, textStyle: { color: '#a9bbb2' } }
      : undefined,
    grid: { left: 42, right: 18, top: forecast || spikes ? 38 : 20, bottom: 48 },
    xAxis: {
      type: 'category',
      data: labels,
      boundaryGap: chartType === 'bar',
      axisLabel: { color: '#899c95', hideOverlap: true },
      axisLine: { lineStyle: { color: '#40514b' } },
    },
    yAxis: {
      type: 'value',
      min: 0,
      axisLabel: { color: '#899c95' },
      splitLine: { lineStyle: { color: 'rgba(214,236,225,.1)' } },
    },
    series: renderedSeries,
  }
}
