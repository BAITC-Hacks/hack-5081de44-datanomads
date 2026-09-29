import { localeTag, translateUi, useUiSettings, type Theme } from '../uiSettings'
import { useEffect, useState } from 'react'
import { acknowledgeAlert, closeAlert, downloadReport, startAlertMonitoring } from '../api/client'
import type { DashboardFilters } from '../api/client'
import type { Alert, DashboardData, ForecastCapacityAssessment, ForecastCapacityInput, ForecastManagerSignal, ForecastPoint, ForecastReforecast, RegionMetric, TopicMetric } from '../types'
import { DataChart } from '../components/DataChart'
import type { EChartsOption } from 'echarts'
import { NoData, PanelHeading, formatAnalyticsChange, type DrilldownHandler } from '../components/AnalyticsPrimitives'

export function CleanRegionsPage({ regions, onDrilldown }: { regions: RegionMetric[]; onDrilldown: DrilldownHandler }) {
  if (!regions.length) return <div className="analytics-page"><NoData message={translateUi("Нет данных по регионам.")} /></div>
  return <div className="analytics-page"><section className="panel"><PanelHeading title={translateUi("Нагрузка по регионам")} /><div className="region-table"><div className="region-table-head"><span>{translateUi("Регион")}</span><span>{translateUi("Текущий период")}</span><span>{translateUi("Предыдущий период")}</span><span>{translateUi("Изменение")}</span></div>{regions.map((region) => <button className="region-table-row drilldown-row" key={region.name} onClick={() => onDrilldown('region', region.id, `Регион: ${region.name}`)}><strong>{translateUi(region.name)}</strong><span>{region.tickets.toLocaleString(localeTag())}</span><span>{region.previousTickets?.toLocaleString(localeTag()) ?? '—'}</span><span>{formatAnalyticsChange(region.changeAbs, region.change)}</span></button>)}</div></section></div>
}

export function CleanTopicsPage({ topics, onDrilldown }: { topics: TopicMetric[]; onDrilldown: DrilldownHandler }) {
  if (!topics.length) return <div className="analytics-page"><NoData message={translateUi("Нет данных по темам.")} /></div>
  return <div className="analytics-page"><section className="panel"><PanelHeading title={translateUi("Распределение по темам")} /><div className="topic-bars">{topics.map((topic) => <button className="topic-bar-row drilldown-row" key={topic.name} onClick={() => onDrilldown('topic', topic.id, `Тема: ${topic.name}`)}><div className="topic-bar-label"><span>{translateUi(topic.name)}</span><strong>{topic.value}% · {topic.tickets?.toLocaleString(localeTag()) ?? '—'}</strong></div>{topic.previousTickets != null && <small className="topic-period-context">{translateUi("Предыдущий период:")} {topic.previousTickets.toLocaleString(localeTag())} {translateUi("· изменение")} {formatAnalyticsChange(topic.changeAbs, topic.change)}</small>}<div className="bar-track"><span style={{ width: String(topic.value) + '%', background: topic.color }} /></div></button>)}</div></section></div>
}

function chartPalette(theme: Theme) {
  return theme === 'dark'
    ? { axis: '#aeb9ca', grid: '#3d4b60', primary: '#79d9bd', secondary: '#83b6ff', comparison: '#f5c47e', area: 'rgba(131, 182, 255, .12)' }
    : { axis: '#5a687b', grid: '#d6dfe9', primary: '#17816d', secondary: '#2366ca', comparison: '#a96614', area: 'rgba(35, 102, 202, .10)' }
}

function chartBaseOption(dates: string[], theme: Theme): EChartsOption {
  const palette = chartPalette(theme)
  return {
    animation: false,
    tooltip: { trigger: 'axis' },
    grid: { left: 42, right: 16, top: 42, bottom: 34 },
    xAxis: { type: 'category', data: dates, boundaryGap: false, axisLabel: { color: palette.axis, formatter: (date: string) => date.slice(5) }, axisLine: { lineStyle: { color: palette.grid } } },
    yAxis: { type: 'value', min: 0, axisLabel: { color: palette.axis }, splitLine: { lineStyle: { color: palette.grid } } },
  }
}

export function CleanTimeSeriesPage({ timeSeries, onDrilldown }: { timeSeries: DashboardData['timeSeries']; onDrilldown: DrilldownHandler }) {
  const { theme } = useUiSettings()
  if (!timeSeries.length) return <div className="analytics-page"><NoData message={translateUi("За выбранный период нет обращений.")} /></div>
  const dates = timeSeries.map((point) => point.date)
  const palette = chartPalette(theme)
  const option: EChartsOption = {
    ...chartBaseOption(dates, theme),
    legend: { data: [translateUi('Обращения'), translateUi('Закрыто')], top: 0, textStyle: { color: palette.axis } },
    series: [
      { name: translateUi('Обращения'), type: 'line' as const, data: timeSeries.map((point) => point.tickets), smooth: false, showSymbol: false, lineStyle: { width: 2 }, itemStyle: { color: palette.primary } },
      { name: translateUi('Закрыто'), type: 'line' as const, data: timeSeries.map((point) => point.resolved), smooth: false, showSymbol: false, lineStyle: { width: 2 }, itemStyle: { color: palette.secondary } },
    ],
  }
  return <div className="analytics-page"><section className="panel"><PanelHeading title={translateUi("Временная динамика")} /><DataChart option={option} label={translateUi("Обращения и закрытые обращения по дням")} /><div className="region-table region-table-full"><div className="region-table-head"><span>{translateUi("Дата")}</span><span>{translateUi("Обращения")}</span><span>{translateUi("Закрыто")}</span><span>{translateUi("Доля закрытия")}</span></div>{timeSeries.map((point) => <button className="region-table-row region-table-row-full drilldown-row" key={point.date} onClick={() => onDrilldown('date', point.date, `Дата: ${point.date}`)}><strong>{point.date}</strong><span>{point.tickets}</span><span>{point.resolved}</span><span>{point.tickets ? `${Math.round(point.resolved / point.tickets * 100)}%` : '—'}</span></button>)}</div></section></div>
}

function formatAlertNumber(value: number | undefined): string {
  return value == null ? '—' : new Intl.NumberFormat(localeTag(), { maximumFractionDigits: 2 }).format(value)
}

function formatAlertTime(value: string | undefined): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat(localeTag(), { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

function alertPeriodLabel(alert: Alert): string {
  if (alert.periodStart && alert.periodEnd) {
    return `${formatAlertTime(alert.periodStart)} — ${formatAlertTime(alert.periodEnd)}`
  }
  return formatAlertTime(alert.createdAt ?? alert.detectedAt)
}

function alertMonitoringMessage(alert: Alert): string {
  const monitoring = alert.monitoring
  if (!monitoring) return ''
  if (monitoring.state === 'MONITORING') {
    return `Период наблюдения завершится ${formatAlertTime(monitoring.ends_at)}. Итог будет определён по полным интервалам после окончания срока.`
  }
  switch (monitoring.state) {
    case 'STABILIZED': return 'Последний полный интервал ниже порога детектора по обращениям, доступным в Pulse.'
    case 'PERSISTING': return 'В последнем полном интервале аномальный рост сохраняется, число обращений не выше исходного сигнала.'
    case 'WORSENING': return 'В последнем полном интервале число обращений выше исходного сигнала, порог детектора всё ещё превышен.'
    case 'RECURRED': return 'После полного интервала ниже порога детектора сигнал повторно превысил порог.'
    case 'INSUFFICIENT_HISTORY': return 'Недостаточно воспроизводимых данных для оценки динамики; вывод о сигнале не сделан.'
  }
}

function alertMonitoringStateLabel(state: NonNullable<Alert['monitoring']>['state']): string {
  switch (state) {
    case 'MONITORING': return 'Под наблюдением'
    case 'STABILIZED': return 'Ниже порога'
    case 'PERSISTING': return 'Сохраняется'
    case 'WORSENING': return 'Рост усилился'
    case 'RECURRED': return 'Повторный рост'
    case 'INSUFFICIENT_HISTORY': return 'Недостаточно данных'
  }
}

function alertChartOption(alert: Alert, theme: Theme): EChartsOption | undefined {
  const historicalCounts = [...(alert.historyCounts ?? [])].reverse()
  if (!historicalCounts.length) return undefined
  const palette = chartPalette(theme)

  const periodDays = alert.periodDays ?? 7
  const monitoringPeriods = alert.monitoring?.evidence?.periods ?? []
  const labels = [
    ...historicalCounts.map((_, index) => `${periodDays} ${translateUi('дн.')} −${(historicalCounts.length - index) * periodDays}`),
    `${translateUi('Последние')} ${periodDays} ${translateUi('дн.')}`,
    ...monitoringPeriods.map((period, index) => `${translateUi('Контроль')} ${index + 1} · ${formatAlertTime(period.period_start)}`),
  ]
  const counts = [
    ...historicalCounts,
    alert.currentCount ?? alert.affectedTickets,
    ...monitoringPeriods.map((period) => period.current_count),
  ]
  const series: NonNullable<EChartsOption['series']> = [
    {
      name: translateUi('Обращения'),
      type: 'line',
      data: counts,
      showSymbol: true,
      lineStyle: { width: 2 },
      itemStyle: { color: palette.primary },
    },
  ]
  if (alert.baseline != null) {
    series.push({
      name: translateUi('Обычный уровень'),
      type: 'line',
      data: counts.map(() => alert.baseline),
      showSymbol: false,
      lineStyle: { width: 1, type: 'dashed' },
      itemStyle: { color: palette.secondary },
    })
  }
  return {
    animation: false,
    tooltip: { trigger: 'axis' },
    legend: { data: series.map((item) => item.name).filter((name): name is string => Boolean(name)), top: 0, textStyle: { color: palette.axis } },
    grid: { left: 42, right: 16, top: 42, bottom: 34 },
    xAxis: { type: 'category', data: labels, axisLabel: { color: palette.axis }, axisLine: { lineStyle: { color: palette.grid } } },
    yAxis: { type: 'value', min: 0, axisLabel: { color: palette.axis }, splitLine: { lineStyle: { color: palette.grid } } },
    series,
  }
}

function alertTriggerDescriptions(alert: Alert): string[] {
  const reasons = alert.triggerReasons ?? []
  if (!reasons.length) return ['Причина срабатывания не сохранена в evidence этого alert.']
  return reasons.map((reason) => {
    if (reason === 'ROBUST_Z_THRESHOLD') {
      return `Robust z ${formatAlertNumber(alert.robustZ)} превысил порог ${formatAlertNumber(alert.robustZThreshold)}.`
    }
    if (reason === 'RATIO_THRESHOLD') {
      return `Количество выше baseline в ${formatAlertNumber(alert.ratio)}× (порог ${formatAlertNumber(alert.ratioThreshold)}×).`
    }
    return reason
  })
}

export function CleanAlertsPage({ alerts, onRefresh, onToast, onDrilldown }: { alerts: Alert[]; onRefresh: () => Promise<void>; onToast: (message: string) => void; onDrilldown: DrilldownHandler }) {
  const { theme } = useUiSettings()
  const [items, setItems] = useState(alerts)
  const [showHistory, setShowHistory] = useState(false)
  const [selectedAlertId, setSelectedAlertId] = useState(alerts[0]?.id)
  const [monitoringPeriodDays, setMonitoringPeriodDays] = useState(21)
  useEffect(() => setItems(alerts), [alerts])

  const prioritized = [...items].sort((left, right) => {
    const severityOrder = { critical: 0, watch: 1, info: 2 }
    const severityDifference = severityOrder[left.severity] - severityOrder[right.severity]
    if (severityDifference !== 0) return severityDifference
    return Date.parse(right.createdAt ?? right.detectedAt) - Date.parse(left.createdAt ?? left.detectedAt)
  })
  const attentionItems = prioritized.filter((alert) => alert.status !== 'Закрыт')
  const historyItems = prioritized.filter((alert) => alert.status === 'Закрыт')
  const visibleItems = showHistory ? historyItems : attentionItems
  const selectedAlert = visibleItems.find((alert) => alert.id === selectedAlertId) ?? visibleItems[0]

  const update = async (alert: Alert, action: 'ack' | 'close') => {
    try {
      const updated = action === 'ack' ? await acknowledgeAlert(alert.id) : await closeAlert(alert.id)
      const status = updated.status.toUpperCase() === 'ACKNOWLEDGED' ? 'В работе' : updated.status.toUpperCase() === 'CLOSED' ? 'Закрыт' : alert.status
      setItems((current) => current.map((item) => item.id === alert.id ? { ...item, status } : item))
      onToast(`Оповещение ${alert.id}: состояние подтверждено backend`)
    } catch (error) {
      onToast(`Не удалось обновить оповещение: ${error instanceof Error ? error.message : 'ошибка API'}`)
    }
  }

  const chart = selectedAlert ? alertChartOption(selectedAlert, theme) : undefined
  const sourceTicketIds = selectedAlert?.linkedTicketIds ?? []
  const currentCount = selectedAlert?.currentCount ?? selectedAlert?.affectedTickets ?? 0
  const periodDays = selectedAlert?.periodDays ?? 7
  const monitoringPeriodOptions = Array.from({ length: 12 }, (_, index) => periodDays * (index + 1))
  const hasActiveMonitoring = selectedAlert?.monitoring?.state === 'MONITORING'

  useEffect(() => {
    if (selectedAlert) setMonitoringPeriodDays(periodDays * 3)
  }, [selectedAlert?.id, periodDays])

  useEffect(() => {
    const endTime = items
      .filter((alert) => alert.monitoring?.state === 'MONITORING')
      .map((alert) => Date.parse(alert.monitoring?.ends_at ?? ''))
      .filter(Number.isFinite)
      .sort((left, right) => left - right)[0]
    if (endTime == null) return
    const refreshDelay = Math.max(50, Math.min(endTime - Date.now() + 50, 86_400_000))
    const timer = window.setTimeout(() => { void onRefresh() }, refreshDelay)
    return () => window.clearTimeout(timer)
  }, [items, onRefresh])

  const takeUnderMonitoring = async (alert: Alert) => {
    try {
      await startAlertMonitoring(alert.id, monitoringPeriodDays)
      await onRefresh()
      onToast(`Сигнал ${alert.id} взят под наблюдение на ${monitoringPeriodDays} дн.`)
    } catch (error) {
      onToast(`Не удалось начать наблюдение: ${error instanceof Error ? error.message : 'ошибка API'}`)
    }
  }

  return (
    <div className="analytics-page">
      <div className="alerts-layout alerts-workspace">
        <section className="panel alert-list-panel">
          <PanelHeading title={translateUi("Требует внимания")} />
          <div className="alert-filter-row" role="group" aria-label={translateUi("Фильтр истории оповещений")}>
            <button className={`filter-chip ${!showHistory ? 'active' : ''}`} aria-pressed={!showHistory} onClick={() => setShowHistory(false)}>
              {translateUi("Требуют внимания")} <span>{attentionItems.length}</span>
            </button>
            <button className={`filter-chip ${showHistory ? 'active' : ''}`} aria-pressed={showHistory} onClick={() => setShowHistory(true)}>
              {translateUi("История")} <span>{historyItems.length}</span>
            </button>
          </div>
          {visibleItems.length ? (
            <div className="alerts-table">
              {visibleItems.map((alert) => (
                  <button key={alert.id} className={`alert-row alert-drilldown ${selectedAlert?.id === alert.id ? 'active' : ''}`} aria-pressed={selectedAlert?.id === alert.id} onClick={() => setSelectedAlertId(alert.id)}>
                    <span className={`alert-dot sev-dot ${alert.severity === 'critical' ? 'high' : alert.severity === 'watch' ? 'medium' : 'low'}`} />
                    <span className="alert-row-main">
                      <strong>{translateUi(alert.title)}</strong>
                      <span>{translateUi(alert.region)} · {translateUi(alert.topic)}</span>
                      <small>{alertPeriodLabel(alert)} · {translateUi(alert.status)}</small>
                    </span>
                    <span className="alert-count">{(alert.currentCount ?? alert.affectedTickets).toLocaleString(localeTag())}<small>{translateUi("обращений")}</small></span>
                  </button>
              ))}
            </div>
          ) : (
            <NoData message={translateUi(showHistory ? 'Закрытых оповещений пока нет.' : 'Нет сигналов, требующих внимания.')} />
          )}
        </section>

        {selectedAlert ? (
          <section className="panel alert-detail-panel signal-card" aria-label={translateUi("Карточка сигнала")}>
            <div className="signal-card-topline">
              <span className={`alert-severity-badge alert-severity-${selectedAlert.severity}`}>
                {translateUi(selectedAlert.severity === 'critical' ? 'Высокий приоритет' : selectedAlert.severity === 'watch' ? 'Повышенный приоритет' : 'Обычный приоритет')}
              </span>
              <span className="signal-status">{translateUi(selectedAlert.status)}</span>
            </div>
            <h2>{translateUi(selectedAlert.title)}</h2>
            <p>{translateUi(selectedAlert.description)}</p>
            <div className="signal-location">
              <span><small>{translateUi("Регион")}</small><strong>{translateUi(selectedAlert.region)}</strong></span>
              <span><small>{translateUi("Тема")}</small><strong>{translateUi(selectedAlert.topic)}</strong></span>
              <span><small>{translateUi("Период")}</small><strong>{alertPeriodLabel(selectedAlert)}</strong></span>
            </div>

            <div className="alert-facts">
              <div><span>{translateUi("Текущий уровень ·")} {periodDays} {translateUi("дн.")}</span><strong>{currentCount.toLocaleString(localeTag())}</strong></div>
              <div><span>{translateUi("Обычный уровень ·")} {periodDays} {translateUi("дн.")}</span><strong>{formatAlertNumber(selectedAlert.baseline)}</strong></div>
              <div><span>{translateUi("Отклонение от baseline")}</span><strong>{formatAlertNumber(selectedAlert.deviation)}</strong></div>
              <div><span>{translateUi("Версия detector")}</span><strong>{selectedAlert.detectorVersion ?? translateUi('не сохранена')}</strong></div>
            </div>

            {selectedAlert.monitoring && (
              <div className="signal-monitoring" aria-live="polite">
                <div className="field-label">{translateUi("Наблюдение динамики")}</div>
                <strong>{translateUi(alertMonitoringStateLabel(selectedAlert.monitoring.state))}</strong>
                <p>{translateUi(alertMonitoringMessage(selectedAlert))}</p>
                <small>{selectedAlert.monitoring.monitoring_period_days} {translateUi("дн. · интервал")} {selectedAlert.monitoring.observation_period_days} {translateUi("дн. · начал:")} {selectedAlert.monitoring.started_by}</small>
                {selectedAlert.monitoring.evidence?.periods?.length ? (
                  <ul>
                    {selectedAlert.monitoring.evidence.periods.map((period) => (
                      <li key={period.period_start}>
                        {formatAlertTime(period.period_start)} — {formatAlertTime(period.period_end)}: {period.current_count.toLocaleString(localeTag())} {translateUi("обращений · исходных ID")} {period.source_ticket_ids.length}
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>
            )}

            {chart ? (
              <div className="alert-detail-chart">
                <div className="field-label">{translateUi("Динамика обращений и обычный уровень")}</div>
                <DataChart option={chart} label={`Динамика обращений по теме ${selectedAlert.topic} в регионе ${selectedAlert.region}`} />
              </div>
            ) : (
              <p className="signal-missing-evidence">{translateUi("Истории недостаточно для графика; сохранённое detector evidence не содержит периодных значений.")}</p>
            )}

            <div className="signal-reasons">
              <div className="field-label">{translateUi("Почему сработал detector")}</div>
              <ul>{alertTriggerDescriptions(selectedAlert).map((reason) => <li key={reason}>{translateUi(reason)}</li>)}</ul>
              {selectedAlert.robustZ != null && <small>Robust z: {formatAlertNumber(selectedAlert.robustZ)} {translateUi("· отношение к baseline:")} {formatAlertNumber(selectedAlert.ratio)}×</small>}
            </div>

            <div className="signal-sources">
              <div className="field-label">{translateUi("Исходные обращения ·")} {sourceTicketIds.length}</div>
              {sourceTicketIds.length ? (
                <ul>{sourceTicketIds.slice(0, 5).map((ticketId) => <li key={ticketId}>№ {ticketId}</li>)}</ul>
              ) : (
                <p>{translateUi("Для этого alert не сохранены связанные обращения.")}</p>
              )}
              {sourceTicketIds.length > 5 && <small>{translateUi("И ещё")} {sourceTicketIds.length - 5}</small>}
            </div>

            <div className="signal-actions">
              <button className="signal-open-tickets" disabled={!sourceTicketIds.length} onClick={() => onDrilldown('alert', selectedAlert.id, `Обращения по сигналу: ${selectedAlert.title}`)}>
                {translateUi("Открыть обращения")}
              </button>
              {!showHistory && selectedAlert.status !== 'Закрыт' && !hasActiveMonitoring && (
                <>
                  <label className="signal-monitoring-period">
                    <span>{translateUi("Период контроля")}</span>
                    <select value={monitoringPeriodDays} onChange={(event) => setMonitoringPeriodDays(Number(event.target.value))}>
                      {monitoringPeriodOptions.map((days) => <option value={days} key={days}>{days} {translateUi("дн.")}</option>)}
                    </select>
                  </label>
                  <button className="text-button" onClick={() => takeUnderMonitoring(selectedAlert)}>
                    {translateUi(selectedAlert.monitoring ? 'Взять на контроль повторно' : 'Взять на контроль')}
                  </button>
                </>
              )}
              {!showHistory && selectedAlert.status === 'Новый' && <button className="text-button" onClick={() => update(selectedAlert, 'ack')}>{translateUi("Принять")}</button>}
              {!showHistory && selectedAlert.status !== 'Закрыт' && <button className="text-button" onClick={() => update(selectedAlert, 'close')}>{translateUi("Закрыть")}</button>}
            </div>
          </section>
        ) : (
          <section className="panel alert-detail-panel"><NoData message={translateUi("Выберите сигнал или запись истории.")} /></section>
        )}
      </div>
    </div>
  )
}

function formatForecastNumber(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value)
    ? '—'
    : new Intl.NumberFormat(localeTag(), { maximumFractionDigits: 1 }).format(value)
}

function formatForecastDate(value: string): string {
  const date = new Date(`${value}T12:00:00`)
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat(localeTag(), { day: 'numeric', month: 'short', year: 'numeric' }).format(date)
}

function formatForecastRate(value: number | null | undefined): string {
  return typeof value !== 'number' || !Number.isFinite(value)
    ? '—'
    : new Intl.NumberFormat(localeTag(), { style: 'percent', maximumFractionDigits: 1 }).format(value)
}

function forecastChartOption(history: ForecastPoint[], forecast: ForecastPoint[], theme: Theme, forecastStart?: string, previousForecast: ForecastPoint[] = []): EChartsOption | undefined {
  const hasHistory = history.length > 0
  const hasForecast = forecast.length > 0
  const hasPreviousForecast = previousForecast.length > 0
  if (!hasHistory && !hasForecast && !hasPreviousForecast) return undefined
  const palette = chartPalette(theme)

  const labels = [...new Set([...history, ...previousForecast, ...forecast].map((point) => point.label))].sort()
  const historicalByDate = new Map(history.map((point) => [point.label, point.actual ?? null]))
  const previousByDate = new Map(previousForecast.map((point) => [point.label, point.forecast ?? null]))
  const forecastByDate = new Map(forecast.map((point) => [point.label, point.forecast ?? null]))
  const series: NonNullable<EChartsOption['series']> = []
  if (hasHistory) {
    series.push({
      name: translateUi(hasPreviousForecast ? 'Факт' : 'История'),
      type: 'line',
      data: labels.map((label) => historicalByDate.get(label) ?? null),
      showSymbol: false,
      lineStyle: { width: 2 },
      itemStyle: { color: palette.primary },
    })
  }
  if (hasPreviousForecast) {
    series.push({
      name: translateUi('Предыдущий прогноз'),
      type: 'line',
      data: labels.map((label) => previousByDate.get(label) ?? null),
      showSymbol: false,
      lineStyle: { width: 1, type: 'dotted' },
      itemStyle: { color: palette.comparison },
    })
  }
  if (hasForecast) {
    series.push({
      name: translateUi(hasPreviousForecast ? 'Обновлённый прогноз' : 'Прогноз'),
      type: 'line',
      data: labels.map((label) => forecastByDate.get(label) ?? null),
      showSymbol: false,
      lineStyle: { width: 2, type: 'dashed' },
      areaStyle: { color: palette.area },
      itemStyle: { color: palette.secondary },
      markLine: forecastStart ? {
        symbol: 'none',
        lineStyle: { color: palette.comparison, type: 'dashed', width: 1 },
        label: { color: palette.comparison, formatter: translateUi('Начало прогноза') },
        data: [{ xAxis: forecastStart }],
      } : undefined,
    })
  }

  return {
    ...chartBaseOption(labels, theme),
    animation: false,
    tooltip: { trigger: 'axis' },
    legend: { data: series.map((item) => item.name).filter((name): name is string => Boolean(name)), top: 0, textStyle: { color: palette.axis } },
    grid: { left: 45, right: 18, top: 43, bottom: 48 },
    xAxis: { type: 'category', data: labels, axisLabel: { color: palette.axis, interval: 'auto', hideOverlap: true }, axisLine: { lineStyle: { color: palette.grid } } },
    yAxis: { type: 'value', min: 0, axisLabel: { color: palette.axis }, splitLine: { lineStyle: { color: palette.grid } } },
    series,
  }
}

const forecastCapacityInputLabels: Record<ForecastCapacityInput, string> = {
  STAFFING: 'Подтверждённая численность сотрудников',
  HANDLING_TIME_OR_THROUGHPUT: 'Время обработки или пропускная способность',
  SCHEDULE: 'Рабочие графики',
  SERVICE_LEVEL_TARGET_OR_SLA: 'Целевой уровень обслуживания или подтверждённый SLA',
}

export function CleanForecastPage({ forecast, history, previousForecast, reforecast, managerSignals, capacityAssessment, runId, issuedAt, status, modelVersion, model, source, insufficientHistory, forecastStart, expectedPeaks, backtest, horizon, filters, filterOptions, onHorizonChange }: {
  forecast: ForecastPoint[]
  history: ForecastPoint[]
  previousForecast: ForecastPoint[]
  reforecast?: ForecastReforecast
  managerSignals: ForecastManagerSignal[]
  capacityAssessment: ForecastCapacityAssessment
  runId?: string
  issuedAt?: string
  status?: string
  modelVersion?: string
  model?: string
  source?: string
  insufficientHistory: boolean
  forecastStart?: string
  expectedPeaks: string[]
  backtest?: DashboardData['forecastBacktest']
  horizon: 30 | 60 | 90
  filters: DashboardFilters
  filterOptions: DashboardData['filterOptions']
  onHorizonChange: (horizon: 30 | 60 | 90) => void
}) {
  const { theme } = useUiSettings()
  const forecastAvailable = !insufficientHistory && (status === 'OK' || status === 'DEMO_ONLY') && forecast.length > 0
  const chart = forecastChartOption(history, forecastAvailable ? forecast : [], theme, forecastStart ?? forecast[0]?.label, previousForecast)
  const values = forecastAvailable
    ? forecast.map((point) => point.forecast).filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
    : []
  const expectedVolume = values.reduce((total, value) => total + value, 0)
  const averageDailyLoad = values.length ? expectedVolume / values.length : 0
  const peakLoad = values.length ? Math.max(...values) : 0
  const relativePeakLoad = averageDailyLoad > 0 ? peakLoad / averageDailyLoad : undefined
  const peakDetails = forecastAvailable
    ? expectedPeaks
      .map((date) => ({ date, point: forecast.find((point) => point.label === date) }))
      .filter((peak): peak is { date: string; point: ForecastPoint } => peak.point !== undefined && typeof peak.point.forecast === 'number')
    : []
  const backtestHasMetrics = [backtest?.mae, backtest?.rmse, backtest?.wape, backtest?.smape]
    .some((value) => typeof value === 'number' && Number.isFinite(value))
  const isDemoBacktest = status === 'DEMO_ONLY' || backtest?.status === 'DEMO_ONLY'
  const forecastBoundary = forecastStart ?? forecast[0]?.label
  const optionLabel = (options: Array<{ id: string; label: string }>, id: string | undefined) =>
    translateUi(id ? options.find((option) => option.id === id)?.label ?? id : 'Все')
  const activeFilters = [
    `${translateUi('Регион')}: ${optionLabel(filterOptions.regions, filters.regionId)}`,
    `${translateUi('Тема')}: ${optionLabel(filterOptions.topics, filters.topicId)}`,
    `${translateUi('Служба')}: ${optionLabel(filterOptions.services, filters.serviceId)}`,
    `${translateUi('Статус')}: ${optionLabel(filterOptions.statuses, filters.status)}`,
    `${translateUi('Район')}: ${optionLabel(filterOptions.districts, filters.district)}`,
    `${translateUi('Канал')}: ${optionLabel(filterOptions.channels, filters.channel)}`,
  ]
  const modelName = model === 'seasonal_naive' || model === 'seasonal-naive-demo'
    ? 'Seasonal Naive · сезонная базовая линия'
    : model ?? 'не указана'
  const sourceLabel = source === 'postgres+ml' || source === 'postgres'
    ? 'Операционные данные'
    : source === 'deterministic-demo'
      ? 'Демо-данные'
      : source ?? 'не указан'
  const statusLabel = status === 'OK'
    ? 'Прогноз рассчитан'
    : status === 'INSUFFICIENT_HISTORY'
      ? 'Недостаточно истории'
      : status === 'DEMO_ONLY'
        ? 'Демо-базовая линия'
        : 'Прогноз не предоставлен'

  return (
    <div className="analytics-page forecast-page">
      <section className="panel forecast-panel-main">
        <div className="forecast-toolbar">
          <PanelHeading title={translateUi("Прогноз нагрузки")} />
          <div className="forecast-horizon-control" role="group" aria-label={translateUi("Горизонт прогноза")}>
            {([30, 60, 90] as const).map((days) => (
              <button key={days} className={`forecast-horizon-button ${horizon === days ? 'active' : ''}`} aria-pressed={horizon === days} onClick={() => onHorizonChange(days)}>
                {days} {translateUi("дней")}
              </button>
            ))}
          </div>
        </div>
        <div className="forecast-provenance">
          <span><small>{translateUi("Модель")}</small><strong>{translateUi(modelName)}</strong></span>
          <span><small>{translateUi("Версия")}</small><strong>{modelVersion ?? translateUi('не указана')}</strong></span>
          <span><small>{translateUi("Состояние")}</small><strong>{translateUi(statusLabel)}</strong></span>
          <span><small>{translateUi("Источник")}</small><strong>{translateUi(sourceLabel)}</strong></span>
        </div>
        <p className="forecast-filter-summary">{translateUi("Применённый срез:")} {activeFilters.join(' · ')}</p>
        <p className="panel-note">{translateUi("Используется дневной ряд за доступную часть окна до 366 дней. Почасовая детализация для текущего ряда недоступна.")}</p>
        {runId && <p className="forecast-run-version">{translateUi("Сохранённый запуск:")} {runId} · {issuedAt ? formatForecastDate(issuedAt.slice(0, 10)) : translateUi('дата не указана')}</p>}

        {reforecast && (
          <section className="forecast-reforecast" aria-label={translateUi("Сравнение версий прогноза")}>
            <div>
              <div className="field-label">{translateUi("Скользящий пересчёт")}</div>
              <strong>forecast_v1 · {reforecast.previousModelVersion} → forecast_v2 · {modelVersion ?? translateUi('версия не указана')}</strong>
              <p>{translateUi("Сопоставляются сохранённый предыдущий прогноз, уже наблюдённый факт и обновлённый прогноз на общих будущих датах.")}</p>
              <small>{translateUi("Предыдущая версия")} {reforecast.previousRunId} {translateUi("· расчёт")} {formatForecastDate(reforecast.previousIssuedAt.slice(0, 10))}</small>
            </div>
            {reforecast.peakChange && (
              <div className="forecast-peak-change">
                <span>{translateUi("Изменение пикового объёма на общих будущих датах")}</span>
                <strong>{formatForecastNumber(reforecast.peakChange.previousPeak)} → {formatForecastNumber(reforecast.peakChange.updatedPeak)} {translateUi("обращений в день")}</strong>
                <small>{formatForecastDate(reforecast.peakChange.previousPeakDate)} → {formatForecastDate(reforecast.peakChange.updatedPeakDate)} · Δ {reforecast.peakChange.delta > 0 ? '+' : ''}{formatForecastNumber(reforecast.peakChange.delta)} · policy {reforecast.peakChange.policyVersion}{reforecast.peakChange.threshold == null ? '' : ` · порог MAE ${formatForecastNumber(reforecast.peakChange.threshold)}`}</small>
                <p>
                  {reforecast.peakChange.status === 'SIGNAL_CREATED'
                    ? `${translateUi('Создан manager signal')} ${reforecast.peakChange.managerSignalId ?? ''}: ${translateUi('изменение больше backtest MAE обеих версий.')}`
                    : translateUi(reforecast.peakChange.status === 'SOURCE_NOT_VERIFIED_REAL'
                      ? 'Manager signal не создан: происхождение всех записей среза не подтверждено как реальное.'
                      : reforecast.peakChange.status === 'WITHIN_BACKTEST_ERROR'
                        ? 'Сигнал не создан: изменение не превышает backtest MAE обеих версий.'
                        : 'Сигнал не создан: для обеих версий нет пригодных backtest MAE.')}
                </p>
              </div>
            )}
            <details>
              <summary>{translateUi("Сверить предыдущий прогноз с фактами и обновлённой версией")}</summary>
              {reforecast.actualObservations.length > 0 && (
                <>
                  <div className="field-label">{translateUi("Прошлый прогноз → факт")}</div>
                  <table>
                    <thead><tr><th>{translateUi("Дата")}</th><th>forecast_v1</th><th>{translateUi("Факт")}</th><th>{translateUi("Абс. ошибка")}</th></tr></thead>
                    <tbody>{reforecast.actualObservations.map((item) => <tr key={`observed-${item.date}`}><td>{item.date}</td><td>{item.previousForecast}</td><td>{item.actual}</td><td>{item.absoluteError}</td></tr>)}</tbody>
                  </table>
                </>
              )}
              {reforecast.futureComparisons.length > 0 && (
                <>
                  <div className="field-label">{translateUi("Прошлый прогноз → обновлённый прогноз")}</div>
                  <table>
                    <thead><tr><th>{translateUi("Дата")}</th><th>forecast_v1</th><th>forecast_v2</th><th>Δ</th></tr></thead>
                    <tbody>{reforecast.futureComparisons.map((item) => <tr key={`future-${item.date}`}><td>{item.date}</td><td>{item.previousForecast}</td><td>{item.updatedForecast}</td><td>{item.delta > 0 ? '+' : ''}{item.delta}</td></tr>)}</tbody>
                  </table>
                </>
              )}
              {!reforecast.actualObservations.length && !reforecast.futureComparisons.length && <p className="forecast-empty-history">{translateUi("У версий нет перекрывающихся дат для сравнения.")}</p>}
            </details>
          </section>
        )}

        {managerSignals.length > 0 && (
          <section className="forecast-manager-signals" aria-label={translateUi("Сигналы изменения прогноза")}>
            <div className="field-label">{translateUi("Сохранённые manager signals по этому срезу")}</div>
            {managerSignals.map((signal) => (
              <p key={signal.id}>
                {formatForecastDate(signal.createdAt.slice(0, 10))}{translateUi(": пиковое значение изменилось с")} {formatForecastNumber(signal.previousPeak)} {translateUi("на")} {formatForecastNumber(signal.updatedPeak)} {translateUi("обращений в день (Δ")} {signal.delta > 0 ? '+' : ''}{signal.delta}{translateUi(", порог MAE")} {formatForecastNumber(signal.threshold)}, {signal.policyVersion}).
              </p>
            ))}
          </section>
        )}

        {insufficientHistory && (
          <div className="forecast-insufficient" role="status">
            <strong>{translateUi("Недостаточно истории для сезонного прогноза.")}</strong>
            <span>{translateUi("Показана наблюдаемая история; будущие значения и ожидаемые пики не подставляются.")}</span>
            {backtest?.observed_days != null && backtest.required_days != null && <small>{translateUi("Активных дневных наблюдений:")} {backtest.observed_days} {translateUi("из")} {backtest.required_days} {translateUi("требуемых.")}</small>}
          </div>
        )}

        {!insufficientHistory && !forecastAvailable && <NoData message={translateUi("Прогноз для выбранного среза не предоставлен.")} />}

        {forecastAvailable && (
          <div className="forecast-volume-grid">
            <div><span>{translateUi("Ожидаемый объём за")} {horizon} {translateUi("дней")}</span><strong>{formatForecastNumber(expectedVolume)}</strong><small>{translateUi("обращений по Seasonal Naive")}</small></div>
            <div><span>{translateUi("Средняя дневная нагрузка")}</span><strong>{formatForecastNumber(averageDailyLoad)}</strong><small>{translateUi("обращений в день")}</small></div>
            <div><span>{translateUi("Пиковый дневной объём")}</span><strong>{formatForecastNumber(peakLoad)}</strong><small>{translateUi(relativePeakLoad == null ? 'относительный пик не выделен' : `${formatForecastNumber(relativePeakLoad)}× от среднего`)}</small></div>
          </div>
        )}

        {forecastAvailable && forecastBoundary && <p className="forecast-boundary-note">{translateUi("Граница прогноза:")} {formatForecastDate(forecastBoundary)}</p>}
        {chart && <DataChart option={chart} label={translateUi("Дневная история обращений и прогноз Seasonal Naive с границей будущего")} />}
        {!chart && !insufficientHistory && <p className="forecast-empty-history">{translateUi("История и точки прогноза для диаграммы отсутствуют.")}</p>}
        {forecastAvailable && (
          <details className="chart-values">
            <summary>{translateUi("Показать дневную историю и прогноз")}</summary>
            <table>
              <thead><tr><th>{translateUi("Дата")}</th><th>{translateUi("Тип ряда")}</th><th>{translateUi("Обращения")}</th></tr></thead>
              <tbody>
                {history.map((point) => <tr key={`history-${point.label}`}><td>{point.label}</td><td>{translateUi("История")}</td><td>{point.actual ?? '—'}</td></tr>)}
                {forecast.map((point) => <tr key={`forecast-${point.label}`}><td>{point.label}</td><td>{translateUi("Прогноз")}</td><td>{point.forecast ?? '—'}</td></tr>)}
              </tbody>
            </table>
          </details>
        )}
      </section>

      <div className="forecast-support-grid">
        <section className="panel forecast-peaks-panel">
          <PanelHeading title={translateUi("Ожидаемые пиковые дни")} />
          <p className="panel-note">{translateUi("Пиковый объём сравнивается со средним дневным прогнозом для этого же среза.")}</p>
          {forecastAvailable && peakDetails.length ? (
            <div className="forecast-peak-list">
              {peakDetails.slice(0, 5).map(({ date, point }) => (
                <div className="forecast-peak-row" key={date}>
                  <span><strong>{formatForecastDate(date)}</strong><small>{translateUi("дневной интервал")}</small></span>
                  <span><strong>{formatForecastNumber(point.forecast)}</strong><small>{translateUi("обращений")}</small></span>
                  <span><strong>{averageDailyLoad > 0 ? `${formatForecastNumber((point.forecast ?? 0) / averageDailyLoad)}×` : '—'}</strong><small>{translateUi("от среднего")}</small></span>
                </div>
              ))}
              {peakDetails.length > 5 && <p className="forecast-peak-more">{translateUi("Показаны 5 из")} {peakDetails.length} {translateUi("пиковых дней.")}</p>}
            </div>
          ) : (
            <p className="forecast-empty-history">{translateUi(forecastAvailable ? 'Положительные ожидаемые пики не выделены.' : 'Пики доступны только при достаточной истории.')}</p>
          )}
        </section>

        <section className="panel forecast-backtest-panel">
          <PanelHeading title={translateUi("Качество baseline на backtest")} />
          <p className="panel-note">{translateUi("Метрики оценивают Seasonal Naive на исторических дневных точках; это не гарантия будущей точности.")}</p>
          {forecastAvailable && backtestHasMetrics ? (
            <>
              <div className="forecast-backtest-grid">
                <div><span>MAE</span><strong>{formatForecastNumber(backtest?.mae)}</strong><small>{translateUi("обращений в день")}</small></div>
                <div><span>RMSE</span><strong>{formatForecastNumber(backtest?.rmse)}</strong><small>{translateUi("обращений в день")}</small></div>
                <div><span>WAPE</span><strong>{formatForecastRate(backtest?.wape)}</strong></div>
                <div><span>SMAPE</span><strong>{formatForecastRate(backtest?.smape)}</strong></div>
              </div>
              <p className="forecast-backtest-samples">{translateUi("Проверено точек:")} {backtest?.sample_count ?? '—'} {translateUi("· окон:")} {backtest?.window_count ?? '—'}</p>
            </>
          ) : (
            <p className="forecast-empty-history">
              {translateUi(isDemoBacktest
                ? 'Оценочные метрики для демо-базовой линии не предоставлены.'
                : insufficientHistory
                  ? 'Backtest не рассчитан из-за недостаточной истории.'
                  : 'Для выбранного среза нет backtest метрик.')}
            </p>
          )}
        </section>
      </div>

      <section className="panel forecast-capacity-panel" aria-label={translateUi("Расчёт риска мощности")}>
        <PanelHeading title={translateUi("Расчёт риска мощности")} />
        <div className="forecast-capacity-summary" role="status" data-capacity-status={capacityAssessment.status}>
          <strong>{capacityAssessment.status}</strong>
          <span>{translateUi("Расчёт недоступен: к системе не подключены подтверждённые операционные данные.")}</span>
        </div>
        <p className="panel-note">{translateUi("До появления всех необходимых источников система показывает только прогноз обращений и пиковые дни, без оценки нехватки персонала.")}</p>
        <ul className="forecast-capacity-inputs" aria-label={translateUi("Необходимые операционные данные")}>
          {capacityAssessment.missingInputs.map((input) => (
            <li key={input}>{forecastCapacityInputLabels[input]}</li>
          ))}
        </ul>
      </section>
    </div>
  )
}

export function CleanReportsPage({ filters, filterOptions }: { filters: DashboardFilters; filterOptions: DashboardData['filterOptions'] }) {
  const [downloading, setDownloading] = useState<'pdf' | 'xlsx' | null>(null)
  const [error, setError] = useState<string | null>(null)
  const optionLabel = (items: Array<{ id: string; label: string }>, selected: string | undefined, empty: string) =>
    translateUi(items.find((item) => item.id === selected)?.label ?? selected ?? empty)
  const filterSummary = [
    translateUi(({ '7d': '7 дней', '30d': '30 дней', '90d': '90 дней' } as Record<string, string>)[filters.range] ?? filters.range),
    optionLabel(filterOptions.regions, filters.regionId, 'все регионы'),
    optionLabel(filterOptions.topics, filters.topicId, 'все темы'),
    optionLabel(filterOptions.services, filters.serviceId, 'все службы'),
    optionLabel(filterOptions.statuses, filters.status, 'все статусы'),
    optionLabel(filterOptions.districts, filters.district, 'все районы'),
    optionLabel(filterOptions.channels, filters.channel, 'все каналы'),
  ].join(', ')
  const download = async (format: 'pdf' | 'xlsx') => {
    setDownloading(format)
    setError(null)
    try { await downloadReport(format, filters) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Не удалось скачать отчёт') }
    finally { setDownloading(null) }
  }
  return <div className="analytics-page"><section className="panel"><PanelHeading title={translateUi("Отчёты")} /><p className="panel-note">{translateUi("В выгрузку войдут данные за")} {filterSummary}.</p><div className="report-actions"><button className="button button-primary" disabled={downloading !== null} onClick={() => void download('pdf')}>{translateUi(downloading === 'pdf' ? 'Загружаем PDF…' : 'Скачать PDF')}</button><button className="button button-secondary" disabled={downloading !== null} onClick={() => void download('xlsx')}>{translateUi(downloading === 'xlsx' ? 'Загружаем XLSX…' : 'Скачать XLSX')}</button></div>{error && <p className="panel-note" role="alert">{translateUi(error)}</p>}</section></div>
}
