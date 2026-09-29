import { localeTag, translateUi, useUiSettings } from '../uiSettings'
import { useMemo } from 'react'
import type { DashboardData } from '../types'
import type { QueryIntentResult } from '../api/client'
import { buildQueryIntentChartOption } from '../query-intent-view'
import { DataChart } from './DataChart'

type FilterKey = 'region_id' | 'topic_id' | 'service_id' | 'status' | 'district' | 'channel'
type FilterOptionKey = 'regions' | 'topics' | 'services' | 'statuses' | 'districts' | 'channels'

const FILTERS: Array<{ key: FilterKey; label: string; options: FilterOptionKey }> = [
  { key: 'region_id', label: 'Регион', options: 'regions' },
  { key: 'topic_id', label: 'Тема', options: 'topics' },
  { key: 'service_id', label: 'Служба', options: 'services' },
  { key: 'status', label: 'Статус', options: 'statuses' },
  { key: 'district', label: 'Район', options: 'districts' },
  { key: 'channel', label: 'Канал', options: 'channels' },
]

function filterDisplayValue(
  key: FilterKey,
  value: string | null | undefined,
  options: DashboardData['filterOptions'],
): string | null {
  if (!value) return null
  const optionSet = FILTERS.find((filter) => filter.key === key)?.options
  const match = optionSet ? options[optionSet].find((option) => option.id === value) : undefined
  return match?.label ?? value
}

function formatCell(value: unknown, key: string): string {
  if (typeof value === 'number') {
    const formatted = value.toLocaleString(localeTag())
    return key === 'change_pct' ? `${formatted}%` : formatted
  }
  if (typeof value === 'boolean') return translateUi(value ? 'Да' : 'Нет')
  return value == null || value === '' ? '—' : key === 'label' ? translateUi(String(value)) : String(value)
}

function sourceLabel(source: string): string {
  if (source === 'postgres+ml') return 'PostgreSQL и ML service'
  if (source === 'postgres') return 'PostgreSQL'
  if (source === 'deterministic-demo') return 'Демо-данные'
  return source
}

export function QueryIntentResultView({
  result,
  filterOptions,
  title,
}: {
  result: QueryIntentResult
  filterOptions: DashboardData['filterOptions']
  title: string
}) {
  const { locale, theme } = useUiSettings()
  const chartOption = useMemo(() => buildQueryIntentChartOption(result, theme, translateUi), [result, locale, theme])
  const activeFilters = FILTERS.flatMap(({ key, label }) => {
    const value = filterDisplayValue(key, result.filters[key], filterOptions)
    return value ? [{ label: translateUi(label), value: translateUi(value) }] : []
  })
  const rows = result.table
  const comparisonChange = result.comparison.change_abs
  const changeLabel = `${comparisonChange > 0 ? '+' : ''}${comparisonChange.toLocaleString(localeTag())}`
  const changePercent = result.comparison.change_pct == null
    ? translateUi('нет данных для расчёта')
    : `${result.comparison.change_pct > 0 ? '+' : ''}${result.comparison.change_pct.toLocaleString(localeTag())}%`
  const summaryText = locale === 'kk'
    ? result.intent === 'spikes'
      ? `Алдыңғы тең кезеңмен салыстырғанда ${result.summary.value ?? 0} өсу кезеңі табылды.`
      : result.intent === 'forecast'
        ? `${result.forecast_horizon_days ?? result.period.days} күнге күтілетін өтініш саны. Модель: ${result.forecast_model ?? 'seasonal_naive'}.`
        : `Таңдалған сүзгілер бойынша ${result.period.days} күндегі өтініш саны.`
    : result.summary.text

  return <div className="query-results" role="status" aria-live="polite">
    <div className="query-result-heading">
      <div>
        <span className="query-result-kicker">{translateUi("Результат аналитического запроса")}</span>
        <strong>{translateUi(title)}</strong>
      </div>
      <span className="query-result-source">{sourceLabel(result.source)}</span>
    </div>
    <div className="query-result-summary">
      <strong>{result.summary.value == null ? '—' : result.summary.value.toLocaleString(localeTag())}</strong>
      <span>{summaryText}</span>
    </div>
    <dl className="query-result-meta">
      <div><dt>{translateUi("Период")}</dt><dd>{result.period.range} · {result.period.start.slice(0, 10)} — {result.period.end.slice(0, 10)}</dd></div>
      <div><dt>{translateUi("Группировка")}</dt><dd>{translateUi(result.grouping)}</dd></div>
      <div><dt>{translateUi("Сравнение с предыдущим периодом")}</dt><dd>{changeLabel} · {changePercent}</dd></div>
      <div className="query-result-definition"><dt>{translateUi("Как считается")}</dt><dd>{translateUi(result.comparison_definition)}</dd></div>
      {activeFilters.map(({ label, value }) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
      {result.forecast_horizon_days != null && <div><dt>{translateUi("Горизонт прогноза")}</dt><dd>{result.forecast_horizon_days} {translateUi("дней")}</dd></div>}
      {result.forecast_status && <div><dt>{translateUi("Состояние прогноза")}</dt><dd>{translateUi(result.forecast_status)}</dd></div>}
      {result.forecast_model_version && <div><dt>{translateUi("Версия модели")}</dt><dd>{result.forecast_model_version}</dd></div>}
      {result.interpreted_filters.limit != null && <div><dt>{translateUi("Лимит строк")}</dt><dd>{result.interpreted_filters.limit}</dd></div>}
    </dl>
    {result.forecast_insufficient_history && <p className="query-result-empty" role="note">{translateUi("Истории недостаточно для прогноза. На графике показаны только наблюдения.")}</p>}
      {chartOption && <DataChart option={chartOption} label={`${translateUi('График')}: ${translateUi(result.chart.title)}`} />}
    <div className="query-result-table-wrap">
      {rows.length > 0
        ? <table className="query-result-table">
          <thead><tr>{result.table_columns.map((column) => <th key={column.key} scope="col">{translateUi(column.label)}</th>)}</tr></thead>
          <tbody>{rows.map((row, index) => <tr key={`${String(row.key ?? row.date ?? row.label ?? 'row')}-${index}`}>
            {result.table_columns.map((column) => <td key={column.key}>{formatCell(row[column.key], column.key)}</td>)}
          </tr>)}</tbody>
        </table>
        : <p className="query-result-empty">{result.intent === 'forecast' && result.forecast_insufficient_history
          ? translateUi('Точки прогноза не рассчитаны.')
          : result.intent === 'spikes'
            ? translateUi('Всплесков по указанному правилу не найдено.')
            : translateUi('По выбранным фильтрам строк нет.')}</p>}
    </div>
  </div>
}
