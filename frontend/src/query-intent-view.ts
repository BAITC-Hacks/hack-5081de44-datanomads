import type { EChartsOption } from 'echarts'
import type { QueryIntentResult } from './api/client'

export function buildQueryIntentChartOption(result: QueryIntentResult, theme: 'light' | 'dark' = 'light', translateUi: (text: string) => string = (text) => text): EChartsOption | null {
  const points = result.series ?? []
  if (points.length === 0) return null

  const axisColor = theme === 'dark' ? '#aeb9ca' : '#5a687b'
  const lineColor = theme === 'dark' ? '#3d4b60' : '#d6dfe9'
  const primaryColor = theme === 'dark' ? '#83b6ff' : '#2366ca'
  const secondaryColor = theme === 'dark' ? '#79d9bd' : '#17816d'
  const comparisonColor = theme === 'dark' ? '#f5c47e' : '#a96614'

  const xKey = result.chart?.x ?? 'label'
  const labels = points.map((point) => translateUi(String(point[xKey] ?? point.label ?? point.date ?? '')))
  const forecast = result.intent === 'forecast'
  const spikes = result.intent === 'spikes'
  const chartType = result.chart?.type ?? (forecast ? 'line' : 'bar')
  const series = forecast
    ? [
      {
        name: translateUi('История'),
        type: 'line' as const,
        data: points.map((point) => point.segment === 'history' ? point.count ?? null : null),
        showSymbol: false,
        connectNulls: false,
        lineStyle: { width: 2, color: primaryColor },
      },
      {
        name: translateUi('Прогноз'),
        type: 'line' as const,
        data: points.map((point) => point.segment === 'forecast' ? point.count ?? null : null),
        showSymbol: false,
        connectNulls: false,
        lineStyle: { width: 2, color: secondaryColor, type: 'dashed' as const },
      },
    ]
    : spikes
      ? [
        {
          name: translateUi('Обращения'),
          type: 'line' as const,
          data: points.map((point) => point.count ?? 0),
          showSymbol: false,
          lineStyle: { width: 2, color: primaryColor },
        },
        {
          name: translateUi('Предыдущий период'),
          type: 'line' as const,
          data: points.map((point) => point.baseline ?? null),
          showSymbol: false,
          lineStyle: { width: 1, color: comparisonColor, type: 'dashed' as const },
        },
      ]
      : [
      {
        name: translateUi(result.chart?.title ?? 'Обращения'),
        type: chartType,
        data: points.map((point) => point.count ?? 0),
        showSymbol: false,
        barMaxWidth: 24,
        lineStyle: { width: 2, color: primaryColor },
        itemStyle: { color: primaryColor },
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
          lineStyle: { type: 'dashed' as const, color: comparisonColor },
          label: { formatter: translateUi('Начало прогноза'), color: comparisonColor },
          data: [{ xAxis: forecastStart }],
        },
      }
      : item)
    : series

  return {
    animation: false,
    tooltip: { trigger: 'axis' },
    legend: forecast || spikes
      ? { data: forecast ? [translateUi('История'), translateUi('Прогноз')] : [translateUi('Обращения'), translateUi('Предыдущий период')], top: 0, textStyle: { color: axisColor } }
      : undefined,
    grid: { left: 42, right: 18, top: forecast || spikes ? 38 : 20, bottom: 48 },
    xAxis: {
      type: 'category',
      data: labels,
      boundaryGap: chartType === 'bar',
      axisLabel: { color: axisColor, hideOverlap: true },
      axisLine: { lineStyle: { color: lineColor } },
    },
    yAxis: {
      type: 'value',
      min: 0,
      axisLabel: { color: axisColor },
      splitLine: { lineStyle: { color: lineColor } },
    },
    series: renderedSeries,
  }
}
