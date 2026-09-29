import { localeTag, translateUi } from '../uiSettings'
import { useState } from 'react'
import { runQueryIntent } from '../api/client'
import type { DashboardFilters, QueryIntentResult } from '../api/client'
import type { DashboardData } from '../types'
import { QueryIntentResultView } from '../components/QueryIntentResultView'
import { Icon } from '../components/Icon'
import { formatDecisionTime, formatRuntimeRate } from '../operator'
import { MetricCard, NoData, PanelHeading, AlertListItem, RegionRow, formatAnalyticsChange, type DrilldownHandler } from '../components/AnalyticsPrimitives'
import type { Route } from '../components/AppChrome'

export function OverviewPage({ data, filters, onNavigate, onDrilldown }: { data: DashboardData; filters: DashboardFilters; onNavigate: (route: Route) => void; onDrilldown: DrilldownHandler }) {
  const [query, setQuery] = useState('')
  const [queryResult, setQueryResult] = useState<QueryIntentResult | null>(null)
  const [queryError, setQueryError] = useState<string | null>(null)
  const [queryLoading, setQueryLoading] = useState(false)
  const total = data.overview.totalTickets
  const highPriority = data.overview.highPriorityTickets
  const metrics = data.operatorMetrics
  const queryTitle = queryResult ? ({
    count: 'Количество обращений',
    trend: 'Динамика обращений',
    compare_regions: 'Обращения по регионам',
    top_topics: 'Темы обращений',
    spikes: 'Всплески обращений',
    forecast: 'Прогноз обращений',
  } as Record<string, string>)[queryResult.intent] ?? 'Результат запроса' : ''
  const submitQuery = async () => {
    if (!query.trim()) return
    setQueryLoading(true)
    setQueryError(null)
    try {
      const result = await runQueryIntent(query, filters)
      setQueryResult(result)
    } catch (error) {
      setQueryResult(null)
      setQueryError(error instanceof Error ? error.message : 'API недоступен')
    } finally {
      setQueryLoading(false)
    }
  }
  return <div className="analytics-page">
    <div className="metrics-grid">
      <MetricCard label={translateUi("Обращений в выборке")} value={total.toLocaleString(localeTag())} change={formatAnalyticsChange(data.overview.changeAbs, data.overview.changePct)} detail={data.overview.previousTotalTickets == null ? translateUi('сравнение недоступно') : `${translateUi('предыдущий период:')} ${data.overview.previousTotalTickets.toLocaleString(localeTag())}`} tone="mint" icon="inbox" onClick={() => onDrilldown('overview', 'all', 'Все обращения')} />
      <MetricCard label={translateUi("Высокий приоритет")} value={String(highPriority)} change={translateUi("текущий срез")} detail={translateUi("по выбранным фильтрам")} tone="rose" icon="pulse" onClick={() => onDrilldown('overview', 'high_priority', 'Высокий приоритет')} />
      <MetricCard label={translateUi("Решения оператора")} value={String(data.overview.operatorDecisions)} change={`${data.overview.confirmedDecisions} ${translateUi('подтверждено')} · ${data.overview.correctedDecisions} ${translateUi('исправлено')}`} detail={translateUi("по данным решений")} tone="amber" icon="clock" />
      <MetricCard label={translateUi("Аномальные сигналы")} value={String(data.alerts.length)} change={translateUi("обнаружено системой")} detail={translateUi("текущий срез")} tone="blue" icon="bell" />
    </div>
    <section className="panel runtime-metrics-panel" aria-label={translateUi("Время решений и качество обратной связи")}>
      <PanelHeading title={translateUi("Время решений и качество обратной связи")} />
      <p className="panel-note">{translateUi("Метрики считаются по решениям и relation feedback операторов; «—» означает, что для показателя пока нет событий в выборке.")}</p>
      <div className="runtime-metrics-grid">
        <MetricCard label={translateUi("Время до первого решения")} value={formatDecisionTime(metrics.operatorDecisionTimeMinutes)} change={metrics.operatorDecisionTimeSamples ? `${metrics.operatorDecisionTimeSamples} ${translateUi('решений')}` : translateUi('нет решений')} detail={translateUi("Среднее от создания обращения")} tone="blue" icon="clock" />
        <MetricCard label={translateUi("Исправление темы")} value={formatRuntimeRate(metrics.classificationCorrectionRate)} change={metrics.classificationDecisions ? `${metrics.classificationCorrections} / ${metrics.classificationDecisions}` : translateUi('нет классификаций')} detail={translateUi("Доля изменённых тем")} tone="amber" icon="edit" />
        <MetricCard label={translateUi("Исправление маршрута")} value={formatRuntimeRate(metrics.routingCorrectionRate)} change={metrics.routingDecisions ? `${metrics.routingCorrections} / ${metrics.routingDecisions}` : translateUi('нет решений с маршрутом')} detail={translateUi("Доля изменённых служб")} tone="rose" icon="arrow" />
        <MetricCard label={translateUi("Исправление приоритета")} value={formatRuntimeRate(metrics.priorityCorrectionRate)} change={metrics.priorityDecisions ? `${metrics.priorityCorrections} / ${metrics.priorityDecisions}` : translateUi('нет решений с приоритетом')} detail={translateUi("Доля изменённого приоритета")} tone="amber" icon="pulse" />
        <MetricCard label={translateUi("Полезность сходства")} value={formatRuntimeRate(metrics.similarityUsefulness)} change={metrics.similarityFeedbackCount ? `${metrics.similarityFeedbackCount} ${translateUi('оценок')}` : translateUi('нет оценок')} detail={translateUi("Подтверждения среди оценок связи")} tone="mint" icon="search" />
        <MetricCard label={translateUi("Precision дубликатов")} value={formatRuntimeRate(metrics.duplicatePrecision)} change={metrics.duplicateFeedbackCount ? `${metrics.duplicateFeedbackCount} ${translateUi('оценок')}` : translateUi('нет оценок')} detail={translateUi("Подтверждения среди проверок дубликата")} tone="blue" icon="check" />
      </div>
    </section>
    <div className="analytics-grid overview-grid">
      <section className="panel span-two"><PanelHeading title={translateUi("Поток обращений")} action={translateUi("Временная динамика")} onClick={() => onNavigate('/situation/time-series')} />{data.timeSeries.length ? <div className="region-table"><div className="region-table-head"><span>{translateUi("Дата")}</span><span>{translateUi("Обращения")}</span><span>{translateUi("Закрыто")}</span><span>{translateUi("Доля")}</span></div>{data.timeSeries.slice(-7).map((point) => <button className="region-table-row drilldown-row" key={point.date} onClick={() => onDrilldown('date', point.date, `Дата: ${point.date}`)}><strong>{point.date}</strong><span>{point.tickets}</span><span>{point.resolved}</span><span>{point.tickets ? `${Math.round(point.resolved / point.tickets * 100)}%` : '—'}</span></button>)}</div> : <NoData message={translateUi("За выбранный период нет обращений.")} />}</section>
      <section className="panel"><PanelHeading title={translateUi("Темы")} action={translateUi("Все темы")} onClick={() => onNavigate('/situation/topics')} />{data.topics.length ? <div className="topic-bars">{data.topics.slice(0, 5).map((topic) => <button className="topic-bar-row drilldown-row" key={topic.name} onClick={() => onDrilldown('topic', topic.id, `Тема: ${topic.name}`)}><div className="topic-bar-label"><span>{translateUi(topic.name)}</span><strong>{topic.value}%</strong></div><div className="bar-track"><span style={{ width: String(topic.value * 2.7) + '%', background: topic.color }} /></div></button>)}</div> : <NoData message={translateUi("Нет данных по темам.")} />}</section>
      <section className="panel"><PanelHeading title={translateUi("Сигналы")} action={translateUi("Открыть все")} onClick={() => onNavigate('/situation/alerts')} />{data.alerts.length ? <div className="alert-list">{data.alerts.map((alert) => <AlertListItem alert={alert} key={alert.id} onDrilldown={onDrilldown} />)}</div> : <NoData message={translateUi("Нет подключённых сигналов.")} />}</section>
      <section className="panel span-two region-panel"><PanelHeading title={translateUi("Регионы")} action={translateUi("Все регионы")} onClick={() => onNavigate('/situation/regions')} />{data.regions.length ? <div className="region-table"><div className="region-table-head"><span>{translateUi("Регион")}</span><span>{translateUi("Текущий период")}</span><span>{translateUi("Предыдущий период")}</span><span>{translateUi("Изменение")}</span></div>{data.regions.map((region) => <RegionRow region={region} key={region.name} onDrilldown={onDrilldown} />)}</div> : <NoData message={translateUi("Нет данных по регионам.")} />}</section>
      <section className="panel query-panel">
        <div className="eyebrow"><span className="eyebrow-line" />{translateUi("Спросить данные")}</div>
        <h3>{translateUi("Ответ по обращениям")}</h3>
        <p>{translateUi("Можно спросить о количестве, динамике, регионах, темах, всплесках или прогнозе.")}</p>
        <div className="query-input"><Icon name="search" size={16} /><input aria-label={translateUi("Вопрос по данным")} placeholder={translateUi("Сколько обращений по регионам?")} value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') void submitQuery() }} /><button aria-label={translateUi("Выполнить поиск")} onClick={() => void submitQuery()} disabled={queryLoading}><Icon name="arrow" size={16} /></button></div>
        {queryError && <p className="query-error" role="alert">{translateUi("Не удалось получить ответ:")} {translateUi(queryError)}</p>}
        {queryResult && <QueryIntentResultView result={queryResult} filterOptions={data.filterOptions} title={queryTitle} />}
      </section>
    </div>
  </div>
}
