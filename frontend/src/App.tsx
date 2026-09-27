import { useCallback, useEffect, useMemo, useRef, useState, type ReactElement } from 'react'
import { acknowledgeAlert, closeAlert, closeLearningCycle, createLearningCycle, loadAnalyticsDrilldown, loadCandidateEvaluation, loadDashboard, loadRelatedTicketDetail, promoteCandidate, rejectCandidate, reportUrl, runQueryIntent, submitDecision, submitRelationFeedback, subscribeToAlertChanges } from './api/client'
import type { AnalyticsDrilldownTicket, DashboardFilters, DrilldownDimension, QueryIntentResult } from './api/client'
import type { Alert, ApiSource, DashboardData, DatasetProvenance, ForecastPoint, LearningCycle, ModelStatus, Priority, RegionMetric, RelatedTicketDetail, RelationSuggestionSnapshot, RuleProvenance, Ticket, TopicMetric } from './types'
import { AuditLogPage } from './components/AuditLogPage'
import { DataChart } from './components/DataChart'
import { QueryIntentResultView } from './components/QueryIntentResultView'
import type { EChartsOption } from 'echarts'
import { languageLabel, languageReviewNotice } from './language'
import { confidenceStateLabel, confidenceStateNotice, normalizeConfidenceState } from './classification'
import { formatDecisionTime, formatRuntimeRate } from './operator'
import { formatExplainabilityFact } from './routing'

type Route =
  | '/operator'
  | '/situation/overview'
  | '/situation/regions'
  | '/situation/topics'
  | '/situation/time-series'
  | '/situation/alerts'
  | '/situation/forecast'
  | '/situation/reports'
  | '/situation/learning'
  | '/situation/models'
  | '/situation/audit'

type IconName = 'inbox' | 'pulse' | 'grid' | 'map' | 'tag' | 'trend' | 'bell' | 'forecast' | 'file' | 'cycle' | 'model' | 'search' | 'settings' | 'help' | 'chevron' | 'arrow' | 'check' | 'edit' | 'external' | 'download' | 'more' | 'clock' | 'close'
type DrilldownHandler = (dimension: DrilldownDimension, value: string | undefined, label: string) => void
type DrilldownState = { label: string; items: AnalyticsDrilldownTicket[]; total: number } | null
type RelatedTicketPanelState = {
  ticketId: string
  matchedFactors: string[]
  loading: boolean
  detail?: RelatedTicketDetail
  error?: string
}
const PREVIEW_STAGE_LABELS: Record<string, string> = {
  language: 'определение языка',
  classification: 'классификация',
  routing: 'маршрутизация',
  priority: 'определение приоритета',
  retrieval: 'поиск похожих обращений',
  duplicate_repeat: 'поиск повторов и дубликатов',
  response_template: 'шаблон ответа',
}

function datasetProvenanceNotice(provenance: DatasetProvenance | undefined): string | undefined {
  if (!provenance) return undefined

  const parts: string[] = []
  if (provenance.synthetic_ticket_count > 0 && provenance.real_ticket_count > 0) {
    parts.push(`В подключённых данных есть синтетические записи (${provenance.synthetic_ticket_count}) и реальные записи (${provenance.real_ticket_count}).`)
  } else if (provenance.synthetic_ticket_count > 0) {
    parts.push(`Синтетический набор: ${provenance.synthetic_ticket_count} обращений. Это демонстрационные данные, не операционная статистика заказчика.`)
  } else if (provenance.real_ticket_count > 0) {
    parts.push(`Реальные записи: ${provenance.real_ticket_count}.`)
  }
  if (provenance.unassigned_ticket_count > 0) {
    parts.push(`Без связи с набором данных: ${provenance.unassigned_ticket_count}.`)
  }
  if (provenance.quarantined_row_count > 0) {
    parts.push(`В карантине: ${provenance.quarantined_row_count}.`)
  }

  return parts.length > 0 ? parts.join(' ') : undefined
}

const icons: Record<IconName, string> = {
  inbox: 'M4 5h16v14H4z M4 8h16 M8 12h3',
  pulse: 'M3 12h3l2-6 4 12 2-6h7',
  grid: 'M4 4h6v6H4z M14 4h6v6h-6z M4 14h6v6H4z M14 14h6v6h-6z',
  map: 'M4 6l5-2 6 2 5-2v14l-5 2-6-2-5 2z M9 4v14 M15 6v14',
  tag: 'M4 5v6l9 9 7-7-9-9H4z M8 8h.01',
  trend: 'M4 18l5-5 3 3 8-9 M15 7h5v5',
  bell: 'M6 17h12l-1.5-2v-4a4.5 4.5 0 0 0-9 0v4z M10 20h4',
  forecast: 'M4 17l4-5 3 2 5-7 4 3 M4 20h16 M4 4v16',
  file: 'M6 3h8l4 4v14H6z M14 3v5h4 M9 13h6 M9 17h4',
  cycle: 'M5 8a7 7 0 0 1 12-2l2 2 M19 16a7 7 0 0 1-12 2l-2-2 M19 6v4h-4 M5 18v-4h4',
  model: 'M5 4h14v16H5z M8 8h8 M8 12h8 M8 16h5',
  search: 'M11 18a7 7 0 1 1 0-14 7 7 0 0 1 0 14z M16 16l4 4',
  settings: 'M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7z M19 13v-2l-2-.6a7 7 0 0 0-.7-1.7l.9-1.8-1.4-1.4-1.8.9a7 7 0 0 0-1.7-.7L12.7 4h-2l-.6 1.7a7 7 0 0 0-1.7.7l-1.8-.9-1.4 1.4.9 1.8a7 7 0 0 0-.7 1.7L4 11v2l1.7.6a7 7 0 0 0 .7 1.7l-.9 1.8 1.4 1.4 1.8-.9a7 7 0 0 0 1.7.7l.6 1.7h2l.6-1.7a7 7 0 0 0 1.7-.7l1.8.9 1.4-1.4-.9-1.8a7 7 0 0 0 .7-1.7z',
  help: 'M9.5 9a2.5 2.5 0 1 1 4.1 1.9c-.9.7-1.6 1.2-1.6 2.6 M12 17h.01 M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z',
  chevron: 'M7 10l5 5 5-5',
  arrow: 'M5 12h14 M13 6l6 6-6 6',
  check: 'M5 12l4 4L19 6',
  edit: 'M4 20h4L19 9l-4-4L4 16z M13 6l4 4',
  external: 'M14 5h5v5 M19 5l-8 8 M18 13v5H5V5h5',
  download: 'M12 4v11 M8 11l4 4 4-4 M5 20h14',
  more: 'M6 12h.01 M12 12h.01 M18 12h.01',
  clock: 'M12 7v5l3 2 M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z',
  close: 'M6 6l12 12 M18 6L6 18',
}

function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  return <svg aria-hidden="true" className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d={icons[name]} /></svg>
}

const routeTitles: Record<Route, { eyebrow: string; title: string; description: string }> = {
  '/operator': { eyebrow: 'Рабочее место', title: 'Входящие обращения', description: 'Решения оператора, которым можно доверять' },
  '/situation/overview': { eyebrow: 'Центр ситуации', title: 'Обзор потока', description: 'Что происходит с обращениями прямо сейчас' },
  '/situation/regions': { eyebrow: 'Центр ситуации', title: 'Регионы', description: 'Где меняется нагрузка и появляется риск' },
  '/situation/topics': { eyebrow: 'Центр ситуации', title: 'Темы обращений', description: 'Распределение спроса по таксономии Pulse' },
  '/situation/time-series': { eyebrow: 'Центр ситуации', title: 'Временная динамика', description: 'Ритм обращений за последние 30 дней' },
  '/situation/alerts': { eyebrow: 'Центр ситуации', title: 'Оповещения', description: 'Сигналы, которые требуют внимания команды' },
  '/situation/forecast': { eyebrow: 'Центр ситуации', title: 'Прогноз', description: 'Ожидаемая нагрузка на ближайшие дни' },
  '/situation/reports': { eyebrow: 'Центр ситуации', title: 'Отчёты', description: 'Срезы для руководителей и рабочих встреч' },
  '/situation/learning': { eyebrow: 'Центр ситуации', title: 'Цикл обучения', description: 'Как обратная связь становится улучшением модели' },
  '/situation/models': { eyebrow: 'Центр ситуации', title: 'Статус моделей', description: 'Версии, метрики и решение о продвижении' },
  '/situation/audit': { eyebrow: 'Администрирование', title: 'Журнал аудита', description: 'Кто, когда и с каким объектом выполнял действие' },
}

const operatorNav = [{ label: 'Входящие', route: '/operator' as Route, icon: 'inbox' as IconName }]
const situationNav = [
  { label: 'Обзор', route: '/situation/overview' as Route, icon: 'grid' as IconName },
  { label: 'Регионы', route: '/situation/regions' as Route, icon: 'map' as IconName },
  { label: 'Темы', route: '/situation/topics' as Route, icon: 'tag' as IconName },
  { label: 'Временная динамика', route: '/situation/time-series' as Route, icon: 'trend' as IconName },
  { label: 'Оповещения', route: '/situation/alerts' as Route, icon: 'bell' as IconName },
  { label: 'Прогноз', route: '/situation/forecast' as Route, icon: 'forecast' as IconName },
  { label: 'Отчёты', route: '/situation/reports' as Route, icon: 'file' as IconName },
  { label: 'Цикл обучения', route: '/situation/learning' as Route, icon: 'cycle' as IconName },
  { label: 'Статус моделей', route: '/situation/models' as Route, icon: 'model' as IconName },
  { label: 'Журнал аудита', route: '/situation/audit' as Route, icon: 'clock' as IconName },
]

function routeFromHash(): Route {
  const path = window.location.hash.replace(/^#/, '') || '/operator'
  return (Object.keys(routeTitles).includes(path) ? path : '/operator') as Route
}

function navigate(route: Route) {
  window.location.hash = route
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`
}

function relationFeedbackType(relation: Ticket['similar'][number]['relation']): 'DUPLICATE' | 'REPEAT' | 'SIMILAR' {
  if (relation === 'Возможный дубликат') return 'DUPLICATE'
  if (relation === 'Возможное повторное обращение') return 'REPEAT'
  return 'SIMILAR'
}

function App() {
  const [route, setRoute] = useState<Route>(routeFromHash)
  const [data, setData] = useState<DashboardData | null>(null)
  const [source, setSource] = useState<ApiSource>('demo')
  const [apiError, setApiError] = useState<string | undefined>()
  const [loading, setLoading] = useState(true)
  const [toast, setToast] = useState<string | null>(null)
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const [filters, setFilters] = useState<DashboardFilters>({ range: '7d' })
  const [drilldown, setDrilldown] = useState<DrilldownState>(null)
  const [drilldownLoading, setDrilldownLoading] = useState(false)

  const refreshDashboard = useCallback(async () => {
    setLoading(true)
    try {
      const result = await loadDashboard(filters)
      setData(result.data)
      setSource(result.source)
      setApiError(result.error)
    } catch (error: unknown) {
      setData(null)
      setSource('api')
      setApiError(error instanceof Error ? error.message : 'API недоступен')
    } finally {
      setLoading(false)
    }
  }, [filters])

  useEffect(() => {
    const onHashChange = () => {
      setRoute(routeFromHash())
      window.scrollTo(0, 0)
    }
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  useEffect(() => {
    if (route === '/situation/audit') return
    let active = true
    setLoading(true)
    loadDashboard(filters).then((result) => {
      if (!active) return
      setData(result.data)
      setSource(result.source)
      setApiError(result.error)
      setLoading(false)
    }).catch((error: unknown) => {
      if (!active) return
      setData(null)
      setSource('api')
      setApiError(error instanceof Error ? error.message : 'API недоступен')
      setLoading(false)
    })
    return () => { active = false }
  }, [filters, route])

  useEffect(() => {
    if (!route.startsWith('/situation') || route === '/situation/audit' || source !== 'api') return
    let active = true
    const unsubscribe = subscribeToAlertChanges(() => {
      loadDashboard(filters).then((result) => {
        if (!active) return
        setData(result.data)
        setApiError(result.error)
      }).catch((error: unknown) => {
        if (active) setApiError(error instanceof Error ? error.message : 'Не удалось обновить оповещения')
      })
    })
    return () => { active = false; unsubscribe() }
  }, [route, source, filters])

  const openDrilldown = useCallback(async (dimension: DrilldownDimension, value: string | undefined, label: string) => {
    setDrilldownLoading(true)
    try {
      const result = await loadAnalyticsDrilldown(dimension, value, filters)
      setDrilldown({ label, items: result.items, total: result.total })
    } catch (error) {
      setDrilldown({ label, items: [], total: 0 })
      setApiError(error instanceof Error ? error.message : 'Не удалось открыть исходные обращения')
    } finally {
      setDrilldownLoading(false)
    }
  }, [filters])

  useEffect(() => {
    if (!toast) return
    const timer = window.setTimeout(() => setToast(null), 4000)
    return () => window.clearTimeout(timer)
  }, [toast])

  const title = routeTitles[route]
  const showToast = useCallback((message: string) => setToast(message), [])
  const provenanceNotice = source === 'api'
    ? datasetProvenanceNotice(data?.datasetProvenance)
    : undefined

  return (
    <div className="app-shell">
      <Sidebar route={route} mobileOpen={mobileNavOpen} onNavigate={(next) => { navigate(next); setMobileNavOpen(false) }} />
      <main className="main-shell">
        <Topbar onOpenNav={() => setMobileNavOpen(true)} onOpenAlerts={() => navigate('/situation/alerts')} hasAlerts={Boolean(data?.alerts.length)} />
        {source === 'demo' && route !== '/situation/audit' && <div className="demo-banner"><span className="status-dot" /> Демо-данные · API подключится автоматически, когда backend будет доступен <span className="demo-banner-detail">{apiError ? `(${apiError})` : ''}</span></div>}
        {provenanceNotice && (
          <div className="demo-banner">
            <span className="status-dot" />
            {provenanceNotice}
          </div>
        )}
        {source === 'api' && apiError && route !== '/situation/audit' && <div className="error-banner"><span className="status-dot" /> API недоступен · {apiError}</div>}
        <div className="page-wrap">
          <PageHeader {...title} route={route} filters={filters} onFiltersChange={setFilters} filterOptions={data?.filterOptions} />
          {route === '/situation/audit' ? <AuditLogPage /> : loading ? <LoadingState /> : data ? <RouteContent route={route} data={data} onDataChange={setData} onRefresh={refreshDashboard} onToast={showToast} filters={filters} onFiltersChange={setFilters} onDrilldown={openDrilldown} drilldown={drilldown} drilldownLoading={drilldownLoading} /> : <ErrorState onRetry={() => window.location.reload()} />}
        </div>
      </main>
      {toast && <div role="status" aria-live="polite" className="toast"><span className="toast-check"><Icon name="check" size={15} /></span>{toast}<button aria-label="Закрыть уведомление" className="icon-button toast-close" onClick={() => setToast(null)}><Icon name="close" size={15} /></button></div>}
    </div>
  )
}

function Sidebar({ route, mobileOpen, onNavigate }: { route: Route; mobileOpen: boolean; onNavigate: (route: Route) => void }) {
  return <>
    {mobileOpen && <button className="sidebar-scrim" aria-label="Закрыть меню" onClick={() => onNavigate(route)} />}
    <aside className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`} aria-label="Основная навигация">
      <div className="brand-lockup">
        <div className="brand-mark"><Icon name="pulse" size={22} /></div>
        <div><div className="brand-name">pulse <span>109</span></div><div className="brand-caption">ситуационный центр</div></div>
      </div>
      <div className="sidebar-section-label">Операционная линия</div>
      <nav className="nav-list">
        {operatorNav.map((item) => <NavItem key={item.route} item={item} active={route === item.route} onNavigate={onNavigate} />)}
      </nav>
      <div className="sidebar-section-label situation-label">Центр ситуации</div>
      <nav className="nav-list">
        {situationNav.map((item) => <NavItem key={item.route} item={item} active={route === item.route} onNavigate={onNavigate} />)}
      </nav>
    </aside>
  </>
}

function NavItem({ item, active, onNavigate }: { item: { label: string; route: Route; icon: IconName; badge?: string }; active: boolean; onNavigate: (route: Route) => void }) {
  return <button className={`nav-item ${active ? 'nav-item-active' : ''}`} onClick={() => onNavigate(item.route)} aria-current={active ? 'page' : undefined}><Icon name={item.icon} size={17} /><span>{item.label}</span>{item.badge && <span className="nav-badge">{item.badge}</span>}</button>
}

function Topbar({ onOpenNav, onOpenAlerts, hasAlerts }: { onOpenNav: () => void; onOpenAlerts: () => void; hasAlerts: boolean }) {
  return <header className="topbar">
    <button className="mobile-menu-button icon-button" aria-label="Открыть меню" onClick={onOpenNav}><span className="menu-lines" /></button>
    <div className="topbar-title">Pulse 109</div>
    <button className="notification-button" aria-label="Открыть оповещения" onClick={onOpenAlerts}><Icon name="bell" size={18} />{hasAlerts && <span />}</button>
  </header>
}

function PageHeader({ eyebrow, title, description, route, filters, onFiltersChange, filterOptions }: { eyebrow: string; title: string; description: string; route: Route; filters: DashboardFilters; onFiltersChange: (filters: DashboardFilters) => void; filterOptions?: DashboardData['filterOptions'] }) {
  const situation = route !== '/operator'
  const showPeriodButtons = route !== '/situation/forecast'
  return <div className="page-header">
    <div><div className="eyebrow"><span className="eyebrow-line" />{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>
    {route !== '/operator' && route !== '/situation/audit' && <div className="period-control">
      {showPeriodButtons && <div className="period-buttons">{[['7d', '7 дней'], ['30d', '30 дней'], ['90d', '90 дней']].map(([value, label]) => <button key={value} className={`period-button ${filters.range === value ? 'active' : ''}`} onClick={() => onFiltersChange({ ...filters, range: value })}>{label}</button>)}</div>}
      {situation && filterOptions && <div className="analytics-filter-selects">
        <label><span>Регион</span><select value={filters.regionId ?? ''} onChange={(event) => onFiltersChange({ ...filters, regionId: event.target.value || undefined })}><option value="">Все регионы</option>{filterOptions.regions.map((option) => <option value={option.id} key={option.id}>{option.label}</option>)}</select></label>
        <label><span>Тема</span><select value={filters.topicId ?? ''} onChange={(event) => onFiltersChange({ ...filters, topicId: event.target.value || undefined })}><option value="">Все темы</option>{filterOptions.topics.map((option) => <option value={option.id} key={option.id}>{option.label}</option>)}</select></label>
        {filterOptions.services.length > 0 && <label><span>Служба</span><select value={filters.serviceId ?? ''} onChange={(event) => onFiltersChange({ ...filters, serviceId: event.target.value || undefined })}><option value="">Все службы</option>{filterOptions.services.map((option) => <option value={option.id} key={option.id}>{option.label}</option>)}</select></label>}
        {filterOptions.statuses.length > 0 && <label><span>Статус</span><select value={filters.status ?? ''} onChange={(event) => onFiltersChange({ ...filters, status: event.target.value || undefined })}><option value="">Все статусы</option>{filterOptions.statuses.map((option) => <option value={option.id} key={option.id}>{option.label}</option>)}</select></label>}
        {filterOptions.districts.length > 0 && <label><span>Район</span><select value={filters.district ?? ''} onChange={(event) => onFiltersChange({ ...filters, district: event.target.value || undefined })}><option value="">Все районы</option>{filterOptions.districts.map((option) => <option value={option.id} key={option.id}>{option.label}</option>)}</select></label>}
        {filterOptions.channels.length > 0 && <label><span>Канал</span><select value={filters.channel ?? ''} onChange={(event) => onFiltersChange({ ...filters, channel: event.target.value || undefined })}><option value="">Все каналы</option>{filterOptions.channels.map((option) => <option value={option.id} key={option.id}>{option.label}</option>)}</select></label>}
      </div>}
    </div>}
  </div>
}

function RouteContent({ route, data, onDataChange, onRefresh, onToast, filters, onFiltersChange, onDrilldown, drilldown, drilldownLoading }: { route: Exclude<Route, '/situation/audit'>; data: DashboardData; onDataChange: (data: DashboardData) => void; onRefresh: () => Promise<void>; onToast: (message: string) => void; filters: DashboardFilters; onFiltersChange: (filters: DashboardFilters) => void; onDrilldown: DrilldownHandler; drilldown: DrilldownState; drilldownLoading: boolean }) {
  let content: ReactElement
  switch (route) {
    case '/operator': content = <OperatorPage tickets={data.tickets} overview={data.overview} taxonomy={data.filterOptions} onDataChange={(tickets) => onDataChange({ ...data, tickets })} onToast={onToast} />; break
    case '/situation/overview': content = <OverviewPage data={data} filters={filters} onNavigate={navigate} onDrilldown={onDrilldown} />; break
    case '/situation/regions': content = <CleanRegionsPage regions={data.regions} onDrilldown={onDrilldown} />; break
    case '/situation/topics': content = <CleanTopicsPage topics={data.topics} onDrilldown={onDrilldown} />; break
    case '/situation/time-series': content = <CleanTimeSeriesPage timeSeries={data.timeSeries} onDrilldown={onDrilldown} />; break
    case '/situation/alerts': content = <CleanAlertsPage alerts={data.alerts} onToast={onToast} onDrilldown={onDrilldown} />; break
    case '/situation/forecast': content = <CleanForecastPage forecast={data.forecast} history={data.forecastHistory ?? []} status={data.forecastStatus} modelVersion={data.forecastModelVersion} model={data.forecastModel} source={data.forecastSource} insufficientHistory={data.forecastInsufficientHistory ?? false} forecastStart={data.forecastStart} expectedPeaks={data.forecastExpectedPeaks ?? []} backtest={data.forecastBacktest} horizon={filters.forecastHorizon ?? 30} filters={filters} filterOptions={data.filterOptions} onHorizonChange={(horizon) => onFiltersChange({ ...filters, forecastHorizon: horizon })} />; break
    case '/situation/reports': content = <CleanReportsPage filters={filters} />; break
    case '/situation/learning': content = <CleanLearningPage learning={data.learning} onRefresh={onRefresh} onToast={onToast} />; break
    case '/situation/models': content = <CleanModelsPage models={data.models} />; break
  }
  return <>{content}{route !== '/operator' && <AnalyticsDrilldownPanel state={drilldown} loading={drilldownLoading} />}</>
}

function AnalyticsDrilldownPanel({ state, loading }: { state: DrilldownState; loading: boolean }) {
  if (loading) return <section className="panel drilldown-panel" aria-live="polite"><div className="panel-heading"><h2>Исходные обращения</h2></div><p className="panel-note">Загружаем обращения…</p></section>
  if (!state) return null
  return <section className="panel drilldown-panel" aria-live="polite"><div className="panel-heading"><h2>Обращения в выбранном срезе</h2><span className="drilldown-label">{state.label} · {state.total}</span></div><p className="panel-note">Исходный текст и контактные данные скрыты в аналитике.</p>{state.items.length ? <div className="drilldown-list">{state.items.map((ticket) => <article className="drilldown-ticket" key={ticket.id}><div><strong>{ticket.id}</strong><span>{ticket.region_name} · {ticket.topic_label}</span></div><small>{ticket.created_at} · {ticket.status} · {ticket.priority}</small></article>)}</div> : <p className="panel-note">За выбранный период обращений не найдено.</p>}</section>
}

function LoadingState() {
  return <div className="loading-grid" aria-label="Загрузка данных"><div className="skeleton skeleton-wide" /><div className="skeleton-row"><div className="skeleton" /><div className="skeleton" /><div className="skeleton" /></div><div className="skeleton skeleton-large" /></div>
}

function ErrorState({ onRetry }: { onRetry: () => void }) {
  return <div className="state-card"><div className="state-icon error-icon">!</div><h2>Не удалось загрузить рабочие данные</h2><p>Проверьте подключение к API и повторите попытку.</p><button className="button button-primary" onClick={onRetry}>Повторить</button></div>
}

function OperatorPage({ tickets, overview, taxonomy, onDataChange, onToast }: { tickets: Ticket[]; overview: DashboardData['overview']; taxonomy: DashboardData['filterOptions']; onDataChange: (tickets: Ticket[]) => void; onToast: (message: string) => void }) {
  const [selectedId, setSelectedId] = useState(tickets[0]?.id ?? '')
  const [query, setQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<'all' | 'new' | 'reviewed'>('all')
  const [priorityFilter, setPriorityFilter] = useState<'all' | Priority>('all')
  const [mobileDetailOpen, setMobileDetailOpen] = useState(false)
  const [relatedDetail, setRelatedDetail] = useState<RelatedTicketPanelState | null>(null)
  const relatedDetailRequestId = useRef(0)
  const selected = tickets.find((ticket) => ticket.id === selectedId) ?? tickets[0]

  const filtered = useMemo(() => tickets.filter((ticket) => {
    const needle = query.trim().toLowerCase()
    const matchesQuery = !needle || [ticket.id, ticket.originalText, ticket.topic, ticket.region].join(' ').toLowerCase().includes(needle)
    const matchesStatus = statusFilter === 'all' || (statusFilter === 'new' ? ticket.status === 'new' : ticket.status !== 'new')
    const matchesPriority = priorityFilter === 'all' || ticket.priority === priorityFilter
    return matchesQuery && matchesStatus && matchesPriority
  }), [tickets, query, statusFilter, priorityFilter])

  const updateTicket = async (ticketId: string, decision: { status: 'confirmed' | 'corrected'; topic?: string; service?: string; priority?: string }) => {
    const ticket = tickets.find((item) => item.id === ticketId)
    if (!ticket) return
    try {
      const result = await submitDecision(ticketId, decision, taxonomy)
      const updated = tickets.map((item) => {
        if (item.id !== ticketId) return item
        if (result.ticket) return result.ticket
        return {
          ...item,
          ...decision,
          priority: (decision.priority as Priority | undefined) ?? item.priority,
          ...(decision.status === 'corrected' ? {
            responseTemplate: '',
            responseTemplateApproved: false,
            responseTemplateSource: 'MANUAL_REQUIRED',
          } : {}),
        }
      })
      onDataChange(updated)
      const feedbackNotice = result.source === 'api' && result.learningFeedbackStatus === 'NO_ACTIVE_COLLECT_CYCLE'
        ? '; обратная связь не включена в цикл: сейчас нет активного COLLECT'
        : ''
      onToast(result.source === 'demo' ? `Решение по ${ticketId} изменено только на этом экране` : `${('warning' in result && result.warning) || `Решение по ${ticketId} сохранено`}${feedbackNotice}`)
    } catch (error) {
      onToast(`Не удалось сохранить решение: ${error instanceof Error ? error.message : 'ошибка API'}`)
    }
  }

  const updateRelation = async (ticketId: string, relatedTicketId: string, relation: 'DUPLICATE' | 'REPEAT' | 'SIMILAR' | 'UNRELATED', decision: 'CONFIRMED' | 'REJECTED', suggestion?: RelationSuggestionSnapshot) => {
    try {
      await submitRelationFeedback(ticketId, relatedTicketId, relation, decision, suggestion)
      onToast(`Обратная связь по ${relatedTicketId} сохранена: ${relation.toLowerCase()} · ${decision.toLowerCase()}`)
    } catch (error) {
      onToast(`Не удалось сохранить связь: ${error instanceof Error ? error.message : 'ошибка API'}`)
    }
  }

  const openRelatedTicket = async (ticketId: string, matchedFactors: string[]) => {
    const requestId = relatedDetailRequestId.current + 1
    relatedDetailRequestId.current = requestId
    setRelatedDetail({ ticketId, matchedFactors, loading: true })
    try {
      const detail = await loadRelatedTicketDetail(ticketId)
      if (relatedDetailRequestId.current !== requestId) return
      setRelatedDetail({ ticketId, matchedFactors, loading: false, detail })
    } catch (error) {
      if (relatedDetailRequestId.current !== requestId) return
      setRelatedDetail({
        ticketId,
        matchedFactors,
        loading: false,
        error: error instanceof Error ? error.message : 'ошибка API',
      })
    }
  }

  const closeRelatedTicket = () => {
    relatedDetailRequestId.current += 1
    setRelatedDetail(null)
  }

  const confirmationRate = overview.operatorDecisions ? Math.round((overview.confirmedDecisions / overview.operatorDecisions) * 100) : 0
  const decisionLatency = formatDecisionTime(overview.avgDecisionMinutes, 'нет решений')
  return <div className="operator-page">
    <div className="operator-summary"><div className="summary-item"><span className="summary-value">{filtered.length}</span><span className="summary-label">обращений в загруженной выборке</span><span className="summary-trend">по текущим фильтрам</span></div><div className="summary-item"><span className="summary-value">{confirmationRate}%</span><span className="summary-label">подтверждение без правок</span><span className="summary-trend">{overview.confirmedDecisions} из {overview.operatorDecisions}</span></div><div className="summary-item"><span className="summary-value">{decisionLatency}</span><span className="summary-label">среднее до решения</span><span className="summary-trend">по данным решений</span></div><div className="summary-item summary-signal"><span className="signal-wave"><i /><i /><i /><i /><i /></span><span><span className="summary-label">Система</span><span className="summary-sub">Рабочие данные доступны</span></span></div></div>
    <div className="workbench-grid">
      <section className="ticket-queue" aria-label="Очередь обращений">
        <div className="section-toolbar"><div><h2>Очередь на разбор <span className="count-pill">{filtered.length}</span></h2><p>Выберите обращение для проверки</p></div></div>
        <div className="filter-row"><div className="inline-search"><Icon name="search" size={16} /><input aria-label="Фильтр очереди" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Найти обращение" /></div><select aria-label="Фильтр по статусу" value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as typeof statusFilter)}><option value="all">Все статусы</option><option value="new">Новые</option><option value="reviewed">Разобранные</option></select><select aria-label="Фильтр по приоритету" value={priorityFilter} onChange={(event) => setPriorityFilter(event.target.value as typeof priorityFilter)}><option value="all">Все приоритеты</option><option value="Критический">Критический</option><option value="Высокий">Высокий</option><option value="Средний">Средний</option><option value="Низкий">Низкий</option></select></div>
        <div className="ticket-table-wrap"><table className="ticket-table"><thead><tr><th scope="col">Обращение</th><th scope="col">Тема</th><th scope="col">Уверенность</th><th scope="col">Регион</th><th scope="col">Приоритет</th><th scope="col"><span className="sr-only">Действия</span></th></tr></thead><tbody>{filtered.map((ticket) => <tr key={ticket.id} className={selected?.id === ticket.id ? 'selected-row' : ''} onClick={() => { setSelectedId(ticket.id); setMobileDetailOpen(true) }}><td><div className="ticket-id">{ticket.id}<span className={`channel-dot channel-${ticket.channel.replace(/[^a-zA-Z]/g, '').toLowerCase()}`} /></div><div className="ticket-preview">{ticket.originalText}</div><span className="ticket-time">{ticket.createdAt} · {languageLabel(ticket.language)}</span></td><td><span className="topic-cell">{ticket.topic}</span><span className="status-text">{ticket.status === 'new' ? 'Нужно решение' : ticket.status === 'confirmed' ? 'Подтверждено' : 'Исправлено'}</span></td><td><Confidence value={ticket.confidence} compact available={ticket.confidenceAvailable !== false} /></td><td><span className="region-cell">{ticket.region}</span></td><td><PriorityBadge priority={ticket.priority} /></td><td><button className="row-arrow icon-button" aria-label={`Открыть ${ticket.id}`} onClick={(event) => { event.stopPropagation(); setSelectedId(ticket.id); setMobileDetailOpen(true) }}><Icon name="arrow" size={17} /></button></td></tr>)}</tbody></table>{filtered.length === 0 && <div className="empty-state"><div className="state-icon">⌕</div><h3>Ничего не найдено</h3><p>Измените запрос или сбросьте фильтры.</p><button className="button button-quiet" onClick={() => { setQuery(''); setStatusFilter('all'); setPriorityFilter('all') }}>Сбросить фильтры</button></div>}</div>
      </section>
      {selected && <TicketDetail ticket={selected} taxonomy={taxonomy} open={mobileDetailOpen} onClose={() => setMobileDetailOpen(false)} onDecision={updateTicket} onRelationFeedback={updateRelation} onOpenRelated={openRelatedTicket} />}
    </div>
    {relatedDetail && <RelatedTicketDetailPanel state={relatedDetail} taxonomy={taxonomy} onClose={closeRelatedTicket} onRetry={() => { void openRelatedTicket(relatedDetail.ticketId, relatedDetail.matchedFactors) }} />}
  </div>
}

function TicketDetail({ ticket, taxonomy, open, onClose, onDecision, onRelationFeedback, onOpenRelated }: { ticket: Ticket; taxonomy: DashboardData['filterOptions']; open: boolean; onClose: () => void; onDecision: (ticketId: string, decision: { status: 'confirmed' | 'corrected'; topic?: string; service?: string; priority?: string }) => Promise<void>; onRelationFeedback: (ticketId: string, relatedTicketId: string, relation: 'DUPLICATE' | 'REPEAT' | 'SIMILAR' | 'UNRELATED', decision: 'CONFIRMED' | 'REJECTED', suggestion?: RelationSuggestionSnapshot) => Promise<void>; onOpenRelated: (ticketId: string, matchedFactors: string[]) => void }) {
  const [correctionOpen, setCorrectionOpen] = useState(false)
  const [replyDraftMode, setReplyDraftMode] = useState<'closed' | 'template' | 'manual'>('closed')
  const [replyDraft, setReplyDraft] = useState('')
  const [templateIgnored, setTemplateIgnored] = useState(false)
  const [topic, setTopic] = useState(ticket.topic)
  const [service, setService] = useState(ticket.service)
  const [priority, setPriority] = useState<Priority>(ticket.priority)
  const confidenceState = normalizeConfidenceState(ticket.confidenceState, ticket.confidence, ticket.confidenceAvailable !== false)
  const confidenceAvailable = ticket.confidenceAvailable !== false && confidenceState !== 'UNAVAILABLE'
  const hasOperatorDecision = ticket.confirmedDecisionAvailable ?? (ticket.status !== 'new')
  const demoRecommendationFallback = ticket.confirmedDecisionAvailable === undefined && ticket.status === 'new'
  const recommendedService = ticket.recommendedService ?? (demoRecommendationFallback ? ticket.service : undefined)
  const recommendedPriority = ticket.recommendedPriority ?? (demoRecommendationFallback ? ticket.priority : undefined)
  const recommendedServiceProvenance = ticket.recommendedServiceProvenance
    ?? (demoRecommendationFallback ? ticket.serviceProvenance : undefined)
  const recommendedPriorityProvenance = ticket.recommendedPriorityProvenance
    ?? (demoRecommendationFallback ? ticket.priorityProvenance : undefined)
  const hasApprovedTemplate = ticket.responseTemplateApproved === true && ticket.responseTemplateSource === 'APPROVED_TEMPLATE'
  const preview = ticket.assistPreview
  const retrievalStage = preview?.stages.find((stage) => stage.name === 'retrieval')
  const emptyHistoryMessage = !preview
    ? 'История обращений не загружена.'
    : retrievalStage && retrievalStage.status !== 'completed'
      ? 'Поиск по истории недоступен. Проверьте связанные обращения вручную.'
      : 'Подходящие похожие обращения, дубликаты и повторы не найдены.'
  const languageNotice = languageReviewNotice(ticket.language)
  const incompleteStages = preview?.stages
    .filter((stage) => ['unavailable', 'skipped', 'unknown'].includes(stage.status))
    .map((stage) => PREVIEW_STAGE_LABELS[stage.name] ?? stage.name) ?? []
  const modelVersions = preview ? Object.entries(preview.model_versions).map(([name, version]) => `${name}: ${version}`).join(' · ') : ''
  const topicOptions = useMemo(() => {
    const options: Array<[string, string]> = [
      ...taxonomy.topics.map((item) => [item.id, item.label] as [string, string]),
      [ticket.topic, ticket.topic],
      ...ticket.alternatives.map((item) => [item.topic, item.topic] as [string, string]),
    ]
    return Array.from(new Map(options.map((option) => [option[1], option])).values())
  }, [taxonomy.topics, ticket.topic, ticket.alternatives])
  const serviceOptions = useMemo(() => {
    const options: Array<[string, string]> = [
      ...taxonomy.services.map((item) => [item.id, item.label] as [string, string]),
      [ticket.service, ticket.service],
    ]
    return Array.from(new Map(options.map((option) => [option[1], option])).values())
  }, [taxonomy.services, ticket.service])
  useEffect(() => {
    setTopic(ticket.topic)
    setService(ticket.service)
    setPriority(ticket.priority)
    setCorrectionOpen(ticket.status === 'new' && (confidenceState === 'LOW_CONFIDENCE' || confidenceState === 'UNAVAILABLE'))
    setReplyDraftMode('closed')
    setReplyDraft('')
    setTemplateIgnored(false)
  }, [ticket.id, ticket.topic, ticket.service, ticket.priority, ticket.status, ticket.responseTemplateId, ticket.responseTemplateVersion, ticket.responseTemplateSource, ticket.responseTemplate, confidenceState])

  return <aside className={`ticket-detail ${open ? 'ticket-detail-open' : ''}`} aria-label={`Детали обращения ${ticket.id}`}>
    <div className="detail-header"><div><div className="detail-overline"><span className={`status-indicator ${ticket.status}`} />{ticket.status === 'new' ? 'Требует решения' : ticket.status === 'confirmed' ? 'Подтверждено' : 'Исправлено'}</div><h2>{ticket.id}</h2></div><button className="icon-button detail-close" aria-label="Закрыть детали" onClick={onClose}><Icon name="close" size={18} /></button></div>
    <div className="detail-scroll">
      <div className="original-text-block"><div className="field-label">Оригинальный текст <span className="language-chip">{languageLabel(ticket.language)}</span></div><p>«{ticket.originalText}»</p><div className="source-line">{ticket.channel} · {ticket.createdAt} · {ticket.region}{ticket.externalRef && ` · № ${ticket.externalRef}`}</div></div>
      {languageNotice && <p className="panel-note" role="status">{languageNotice}</p>}
      {preview?.needs_review && (preview.status === 'partial' || incompleteStages.length > 0 || confidenceState === 'CONFIDENT') && <p className="panel-note" role="status">{preview.status === 'partial' ? 'Предпросмотр неполный.' : incompleteStages.length > 0 ? 'Часть функций недоступна.' : 'Рекомендацию нужно проверить.'} {incompleteStages.length > 0 && `Недоступно: ${incompleteStages.join(', ')}. `}Подтвердите тему, службу и приоритет после ручной проверки. Запрос {preview.request_id} · {preview.latency_ms.toFixed(0)} мс{modelVersions ? ` · ${modelVersions}` : ''}.</p>}
      <div className="detail-section">
        <div className="field-label">Модель предложила</div>
        <div className="prediction-row">
          <div>
            <div className="prediction-topic">{ticket.predictedTopic ?? ticket.topic}</div>
            <div className={`classification-state classification-${confidenceState.toLowerCase()}`}>
              {confidenceStateNotice(confidenceState)}
            </div>
          </div>
          <Confidence value={ticket.confidence} state={confidenceState} available={confidenceAvailable} />
        </div>
        {confidenceState === 'UNCERTAIN' && (
          <div className="alternatives">
            <span className="field-label">Возможные альтернативы</span>
            {ticket.alternatives.length ? ticket.alternatives.map((alternative) => (
              <div className="alternative-row" key={alternative.topic}>
                <span>{alternative.topic}</span>
                <span>{formatPercent(alternative.confidence)}</span>
              </div>
            )) : <p className="panel-note">Альтернативы не получены. Проверьте тему вручную.</p>}
          </div>
        )}
        <details className="model-meta">
          <summary>Технические сведения</summary>
          <span>Оценка модели: {confidenceAvailable ? formatPercent(ticket.confidence) : 'нет данных'}</span>
          {ticket.modelVersion && <span>Модель: {ticket.modelVersion}</span>}
        </details>
      </div>
      {correctionOpen && (
        <div className="correction-panel">
          <div className="correction-heading">
            <strong>Выберите тему вручную</strong>
            <button className="icon-button" aria-label="Закрыть форму исправления" onClick={() => setCorrectionOpen(false)}>
              <Icon name="close" size={15} />
            </button>
          </div>
          <label>Тема
            <select value={topic} onChange={(event) => setTopic(event.target.value)}>
              {topicOptions.map(([id, label]) => <option value={label} key={id}>{label}</option>)}
              <option value="Другая тема">Другая тема</option>
            </select>
          </label>
          <label>Служба
            <select value={service} onChange={(event) => setService(event.target.value)}>
              {serviceOptions.map(([id, label]) => <option value={label} key={id}>{label}</option>)}
              <option value="Другая служба">Другая служба</option>
            </select>
          </label>
          <label>Приоритет
            <select value={priority} onChange={(event) => setPriority(event.target.value as Priority)}>
              <option>Высокий</option>
              <option>Критический</option>
              <option>Средний</option>
              <option>Низкий</option>
              <option>Не определён</option>
            </select>
          </label>
          <button className="button button-primary full-width" onClick={() => onDecision(ticket.id, { status: 'corrected', topic, service, priority })}>
            <Icon name="check" size={16} />Сохранить исправление
          </button>
        </div>
      )}
      <div className="detail-section">
        <div className="field-label">Рекомендация Pulse · маршрутизация и приоритет</div>
        <div className="routing-grid">
          <div className="routing-field">
            <span>Служба</span>
            <strong>{recommendedService ?? 'Не предоставлена'}</strong>
            <RoutingProvenance
              provenance={recommendedServiceProvenance}
              fallbackReason="Источник рекомендованной службы не сохранён; проверьте вручную"
            />
          </div>
          <div className="routing-field">
            <span>Приоритет</span>
            <PriorityBadge priority={recommendedPriority ?? 'Не определён'} />
            <RoutingProvenance
              provenance={recommendedPriorityProvenance}
              fallbackReason="Источник рекомендованного приоритета не сохранён; проверьте вручную"
            />
          </div>
        </div>
        <p className="panel-note"><strong>Основание маршрутизации:</strong> {ticket.routingReason?.trim() || 'Не предоставлено; проверьте службу и приоритет вручную.'}</p>
      </div>
      {hasOperatorDecision && (
        <div className="detail-section">
          <div className="field-label">{ticket.status === 'corrected' ? 'Исправление оператора' : 'Подтверждённое решение оператора'}</div>
          <div className="routing-grid">
            <div className="routing-field">
              <span>Служба</span>
              <strong>{ticket.service}</strong>
              <RoutingProvenance
                provenance={ticket.serviceProvenance}
                fallbackReason="Источник подтверждённой службы не сохранён; проверьте решение вручную"
              />
            </div>
            <div className="routing-field">
              <span>Приоритет</span>
              <PriorityBadge priority={ticket.priority} />
              <RoutingProvenance
                provenance={ticket.priorityProvenance}
                fallbackReason="Источник подтверждённого приоритета не сохранён; проверьте решение вручную"
              />
            </div>
          </div>
        </div>
      )}
      {!hasOperatorDecision && ticket.status !== 'new' && (
        <p className="panel-note" role="status">Статус записи отмечен как разобранный, но данные подтверждённого решения не предоставлены.</p>
      )}
      <div className="detail-section">
        <div className="field-label">Ответ оператору <span className="language-chip">{hasApprovedTemplate ? (ticket.responseTemplateVersion ? `Утверждённый · v${ticket.responseTemplateVersion}` : 'Утверждённый шаблон') : 'Ручной ответ'}</span></div>
        {hasApprovedTemplate && !templateIgnored && (
          <div className="response-template">
            <p>{ticket.responseTemplate}</p>
            <div className="response-template-actions">
              <button className="button button-quiet" onClick={() => { setReplyDraft(ticket.responseTemplate); setReplyDraftMode('template') }}>Использовать шаблон</button>
              <button className="text-button" onClick={() => { setTemplateIgnored(true); setReplyDraftMode('closed'); setReplyDraft('') }}>Игнорировать</button>
            </div>
          </div>
        )}
        {!hasApprovedTemplate && replyDraftMode === 'closed' && (
          <p className="panel-note" role="status">
            {ticket.responseTemplateSource === 'UNAVAILABLE' && ticket.responseTemplate.trim()
              ? ticket.responseTemplate
              : ticket.status === 'new'
                ? 'Утверждённый шаблон появится после подтверждения темы и службы. Ответ можно написать вручную.'
                : 'Для этого решения нет утверждённого шаблона. Составьте ответ вручную.'}
          </p>
        )}
        {templateIgnored && replyDraftMode === 'closed' && (
          <button className="text-button" onClick={() => setTemplateIgnored(false)}>Показать утверждённый шаблон</button>
        )}
        {replyDraftMode === 'closed' && (
          <button className="button button-quiet" onClick={() => { setReplyDraft(''); setReplyDraftMode('manual') }}>Написать вручную</button>
        )}
        {replyDraftMode !== 'closed' && (
          <div className="response-draft">
            <label className="field-label" htmlFor={`reply-draft-${ticket.id}`}>Черновик ответа</label>
            <textarea id={`reply-draft-${ticket.id}`} aria-label="Черновик ответа" value={replyDraft} onChange={(event) => setReplyDraft(event.target.value)} rows={5} />
            <p className="panel-note">Черновик доступен только на этом экране. Отправки в CRM нет.</p>
            <button className="text-button" onClick={() => { setReplyDraftMode('closed'); setReplyDraft('') }}>Закрыть черновик</button>
          </div>
        )}
      </div>
      <div className="detail-section">
        <div className="section-inline-heading">
          <div className="field-label">История и связанные обращения <span className="count-pill">{ticket.similar.length}</span></div>
          <span className="field-label">проверьте связь</span>
        </div>
        <div className="similar-list">
          {ticket.similar.length ? ticket.similar.map((item) => {
            const relation = relationFeedbackType(item.relation)
            return <div className="similar-item" key={item.id}>
              <div className="similar-main-column">
                <button className="similar-main similar-open" aria-label={`Открыть оригинал ${item.id}`} onClick={() => onOpenRelated(item.id, item.matchedFactors ?? [])}>
                  <strong>{item.id}</strong><span>{item.title}</span><Icon name="arrow" size={14} />
                </button>
                <span className="similar-factors">{item.matchedFactors?.length ? item.matchedFactors.join(' · ') : 'Совпадающие признаки не предоставлены'}</span>
                {item.suggestion
                  ? <small>Отбор top-K: score {formatPercent(item.similarity)} · порог {formatPercent(item.suggestion.threshold)}</small>
                  : <small>Порог отбора top-K не сохранён</small>}
              </div>
              <div className="similar-meta">
                <span className={`relation-badge ${item.relation === 'Возможный дубликат' ? 'relation-duplicate' : item.relation === 'Возможное повторное обращение' ? 'relation-repeat' : ''}`}>{item.relation}</span>
                <span>Создано {item.createdAt}</span>
                <span>{formatPercent(item.similarity)}</span>
                <button className="text-button" onClick={() => onRelationFeedback(ticket.id, item.id, relation, 'CONFIRMED', item.suggestion)}>Подтвердить</button>
                <button className="text-button" onClick={() => onRelationFeedback(ticket.id, item.id, relation, 'REJECTED', item.suggestion)}>Отклонить</button>
              </div>
            </div>
          }) : <p className="panel-note">{emptyHistoryMessage}</p>}
        </div>
      </div>
    </div>
    {!correctionOpen && <div className="detail-actions"><button className="button button-primary" onClick={() => onDecision(ticket.id, { status: 'confirmed' })}><Icon name="check" size={16} />Подтвердить</button><button className="button button-secondary" onClick={() => setCorrectionOpen(true)}><Icon name="edit" size={16} />Исправить</button></div>}
  </aside>
}

function RelatedTicketDetailPanel({ state, taxonomy, onClose, onRetry }: { state: RelatedTicketPanelState; taxonomy: DashboardData['filterOptions']; onClose: () => void; onRetry: () => void }) {
  const detail = state.detail
  const decision = detail?.latestDecision
  const decisionTopic = decision
    ? taxonomy.topics.find((item) => item.id === decision.confirmedTopicId)?.label ?? decision.confirmedTopicId
    : ''
  const decisionAction = decision?.action.toLowerCase() === 'confirm'
    ? 'Подтверждение предложенной темы'
    : decision?.action.toLowerCase() === 'correct'
      ? 'Исправление темы оператором'
      : decision?.action

  return <div className="related-ticket-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <section className="related-ticket-panel" role="dialog" aria-modal="true" aria-labelledby="related-ticket-title">
      <header className="related-ticket-header">
        <div><span className="field-label">Контекст для проверки</span><h2 id="related-ticket-title">Связанное обращение</h2><span className="related-ticket-id">{state.ticketId}</span></div>
        <button className="icon-button" aria-label="Закрыть связанное обращение" onClick={onClose}><Icon name="close" size={18} /></button>
      </header>
      {state.loading ? <div className="related-ticket-state" role="status">Загружаю карточку обращения…</div> : state.error ? <div className="related-ticket-state" role="alert"><p>Не удалось загрузить карточку: {state.error}</p><button className="button button-secondary" onClick={onRetry}>Повторить</button></div> : detail && <div className="related-ticket-scroll">
        <section className="related-ticket-section">
          <div className="field-label">Оригинальный текст</div>
          <p className="related-ticket-original">«{detail.originalText}»</p>
          <p className="source-line">{detail.channel} · {detail.region}{detail.externalRef && ` · № ${detail.externalRef}`}</p>
        </section>
        <section className="related-ticket-section">
          <div className="field-label">Тема и статус в записи</div>
          <dl className="related-ticket-meta">
            <div><dt>Создано</dt><dd>{detail.createdAt}</dd></div>
            <div><dt>Закрыто</dt><dd>{detail.closedAt ?? 'Время закрытия не указано'}</dd></div>
            <div><dt>Тема</dt><dd>{detail.topic}</dd></div>
            <div><dt>Статус записи</dt><dd>{detail.status}</dd></div>
          </dl>
        </section>
        <section className="related-ticket-section">
          <div className="field-label">Последнее доступное действие оператора</div>
          {decision ? <dl className="related-ticket-meta">
            <div><dt>Действие</dt><dd>{decisionAction || 'Действие оператора зафиксировано'}</dd></div>
            <div><dt>Подтверждённая тема</dt><dd>{decisionTopic || 'Тема не указана'}</dd></div>
            {decision.service && <div><dt>Служба в решении</dt><dd>{decision.service}</dd></div>}
            {decision.priority && <div><dt>Приоритет в решении</dt><dd>{decision.priority}</dd></div>}
            {decision.createdAt && <div><dt>Время действия</dt><dd>{decision.createdAt}</dd></div>}
          </dl> : <p className="panel-note">Операторское действие не зафиксировано. Исход обращения неизвестен.</p>}
        </section>
        <section className="related-ticket-section">
          <div className="field-label">Проверяемые признаки сходства</div>
          {state.matchedFactors.length ? <div className="related-factor-list">{state.matchedFactors.map((factor) => <span className="relation-badge" key={factor}>{factor}</span>)}</div> : <p className="panel-note">Совпадение по теме или региону не подтверждено доступными полями.</p>}
        </section>
        <p className="related-ticket-note">Это возможное сходство для проверки оператором. Оно не подтверждает, что объект или проблема те же, и не создаёт связь автоматически.</p>
      </div>}
    </section>
  </div>
}

function Confidence({ value, state, compact = false, available = true }: { value: number; state?: Ticket['confidenceState']; compact?: boolean; available?: boolean }) {
  const confidenceState = normalizeConfidenceState(state, value, available)
  return <div className={`confidence ${compact ? 'confidence-compact' : ''}`}><div className="confidence-track"><span style={{ width: available ? `${value * 100}%` : '0%' }} /></div><strong>{available ? formatPercent(value) : '—'}</strong>{!compact && <small>{available ? confidenceStateLabel(confidenceState) : 'нет данных'}</small>}</div>
}

function PriorityBadge({ priority }: { priority: Priority }) {
  const level = priority === 'Критический' ? 'critical' : priority === 'Высокий' ? 'high' : priority === 'Средний' ? 'medium' : priority === 'Низкий' ? 'low' : 'unknown'
  return <span className={`priority-badge priority-${level}`}><span />{priority === 'Не определён' ? 'Не определён' : priority}</span>
}

function RoutingProvenance({ provenance, fallbackReason }: { provenance?: RuleProvenance; fallbackReason: string }) {
  const value = provenance ?? { source: 'MANUAL' as const, version: null, reason: fallbackReason }
  const sourceLabel = value.source === 'OFFICIAL'
    ? 'Официальное правило'
    : value.source === 'LABEL_HISTORY'
      ? 'Историческая метка'
      : 'Ручной источник / решение'
  return (
    <span className={`routing-provenance routing-source-${value.source.toLowerCase()}`}>
      <strong>{value.source} · {sourceLabel}{value.version ? ` · версия ${value.version}` : ''}</strong>
      <span>{value.reason}</span>
      {value.factsUsed?.length
        ? <span>Использованные факты: {value.factsUsed.map(formatExplainabilityFact).join(' · ')}</span>
        : <span>Факты выбора не сохранены</span>}
    </span>
  )
}

function CleanRegionsPage({ regions, onDrilldown }: { regions: RegionMetric[]; onDrilldown: DrilldownHandler }) {
  if (!regions.length) return <div className="analytics-page"><NoData message="Нет данных по регионам." /></div>
  return <div className="analytics-page"><section className="panel"><PanelHeading title="Нагрузка по регионам" /><div className="region-table"><div className="region-table-head"><span>Регион</span><span>Текущий период</span><span>Предыдущий период</span><span>Изменение</span></div>{regions.map((region) => <button className="region-table-row drilldown-row" key={region.name} onClick={() => onDrilldown('region', region.id, `Регион: ${region.name}`)}><strong>{region.name}</strong><span>{region.tickets.toLocaleString('ru-RU')}</span><span>{region.previousTickets?.toLocaleString('ru-RU') ?? '—'}</span><span>{formatAnalyticsChange(region.changeAbs, region.change)}</span></button>)}</div></section></div>
}

function CleanTopicsPage({ topics, onDrilldown }: { topics: TopicMetric[]; onDrilldown: DrilldownHandler }) {
  if (!topics.length) return <div className="analytics-page"><NoData message="Нет данных по темам." /></div>
  return <div className="analytics-page"><section className="panel"><PanelHeading title="Распределение по темам" /><div className="topic-bars">{topics.map((topic) => <button className="topic-bar-row drilldown-row" key={topic.name} onClick={() => onDrilldown('topic', topic.id, `Тема: ${topic.name}`)}><div className="topic-bar-label"><span>{topic.name}</span><strong>{topic.value}% · {topic.tickets?.toLocaleString('ru-RU') ?? '—'}</strong></div>{topic.previousTickets != null && <small className="topic-period-context">Предыдущий период: {topic.previousTickets.toLocaleString('ru-RU')} · изменение {formatAnalyticsChange(topic.changeAbs, topic.change)}</small>}<div className="bar-track"><span style={{ width: String(topic.value) + '%', background: topic.color }} /></div></button>)}</div></section></div>
}

function chartBaseOption(dates: string[]): EChartsOption {
  return {
    animation: false,
    tooltip: { trigger: 'axis' },
    grid: { left: 42, right: 16, top: 42, bottom: 34 },
    xAxis: { type: 'category', data: dates, boundaryGap: false, axisLabel: { color: '#899c95', formatter: (date: string) => date.slice(5) }, axisLine: { lineStyle: { color: '#40514b' } } },
    yAxis: { type: 'value', min: 0, axisLabel: { color: '#899c95' }, splitLine: { lineStyle: { color: 'rgba(214,236,225,.1)' } } },
  }
}

function CleanTimeSeriesPage({ timeSeries, onDrilldown }: { timeSeries: DashboardData['timeSeries']; onDrilldown: DrilldownHandler }) {
  if (!timeSeries.length) return <div className="analytics-page"><NoData message="За выбранный период нет обращений." /></div>
  const dates = timeSeries.map((point) => point.date)
  const option: EChartsOption = {
    ...chartBaseOption(dates),
    legend: { data: ['Обращения', 'Закрыто'], top: 0, textStyle: { color: '#a9bbb2' } },
    series: [
      { name: 'Обращения', type: 'line' as const, data: timeSeries.map((point) => point.tickets), smooth: false, showSymbol: false, lineStyle: { width: 2 }, itemStyle: { color: '#8cf0c8' } },
      { name: 'Закрыто', type: 'line' as const, data: timeSeries.map((point) => point.resolved), smooth: false, showSymbol: false, lineStyle: { width: 2 }, itemStyle: { color: '#a7d9ff' } },
    ],
  }
  return <div className="analytics-page"><section className="panel"><PanelHeading title="Временная динамика" /><DataChart option={option} label="Обращения и закрытые обращения по дням" /><div className="region-table region-table-full"><div className="region-table-head"><span>Дата</span><span>Обращения</span><span>Закрыто</span><span>Доля закрытия</span></div>{timeSeries.map((point) => <button className="region-table-row region-table-row-full drilldown-row" key={point.date} onClick={() => onDrilldown('date', point.date, `Дата: ${point.date}`)}><strong>{point.date}</strong><span>{point.tickets}</span><span>{point.resolved}</span><span>{point.tickets ? `${Math.round(point.resolved / point.tickets * 100)}%` : '—'}</span></button>)}</div></section></div>
}

function formatAlertNumber(value: number | undefined): string {
  return value == null ? '—' : new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 2 }).format(value)
}

function formatAlertTime(value: string | undefined): string {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat('ru-RU', { dateStyle: 'medium', timeStyle: 'short' }).format(date)
}

function alertPeriodLabel(alert: Alert): string {
  if (alert.periodStart && alert.periodEnd) {
    return `${formatAlertTime(alert.periodStart)} — ${formatAlertTime(alert.periodEnd)}`
  }
  return formatAlertTime(alert.createdAt ?? alert.detectedAt)
}

function alertChartOption(alert: Alert): EChartsOption | undefined {
  const historicalCounts = [...(alert.historyCounts ?? [])].reverse()
  if (!historicalCounts.length) return undefined

  const periodDays = alert.periodDays ?? 7
  const labels = [
    ...historicalCounts.map((_, index) => `Неделя −${historicalCounts.length - index}`),
    `Последние ${periodDays} дн.`,
  ]
  const counts = [...historicalCounts, alert.currentCount ?? alert.affectedTickets]
  const series: NonNullable<EChartsOption['series']> = [
    {
      name: 'Обращения',
      type: 'line',
      data: counts,
      showSymbol: true,
      lineStyle: { width: 2 },
      itemStyle: { color: '#8cf0c8' },
    },
  ]
  if (alert.baseline != null) {
    series.push({
      name: 'Обычный уровень',
      type: 'line',
      data: counts.map(() => alert.baseline),
      showSymbol: false,
      lineStyle: { width: 1, type: 'dashed' },
      itemStyle: { color: '#a7d9ff' },
    })
  }
  return {
    animation: false,
    tooltip: { trigger: 'axis' },
    legend: { data: series.map((item) => item.name).filter((name): name is string => Boolean(name)), top: 0, textStyle: { color: '#a9bbb2' } },
    grid: { left: 42, right: 16, top: 42, bottom: 34 },
    xAxis: { type: 'category', data: labels, axisLabel: { color: '#899c95' }, axisLine: { lineStyle: { color: '#40514b' } } },
    yAxis: { type: 'value', min: 0, axisLabel: { color: '#899c95' }, splitLine: { lineStyle: { color: 'rgba(214,236,225,.1)' } } },
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

function CleanAlertsPage({ alerts, onToast, onDrilldown }: { alerts: Alert[]; onToast: (message: string) => void; onDrilldown: DrilldownHandler }) {
  const [items, setItems] = useState(alerts)
  const [showHistory, setShowHistory] = useState(false)
  const [selectedAlertId, setSelectedAlertId] = useState(alerts[0]?.id)
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

  const chart = selectedAlert ? alertChartOption(selectedAlert) : undefined
  const sourceTicketIds = selectedAlert?.linkedTicketIds ?? []
  const currentCount = selectedAlert?.currentCount ?? selectedAlert?.affectedTickets ?? 0
  const periodDays = selectedAlert?.periodDays ?? 7

  return (
    <div className="analytics-page">
      <div className="alerts-layout">
        <section className="panel alert-list-panel">
          <PanelHeading title="Требует внимания" />
          <div className="alert-filter-row" role="group" aria-label="Фильтр истории оповещений">
            <button className={`filter-chip ${!showHistory ? 'active' : ''}`} aria-pressed={!showHistory} onClick={() => setShowHistory(false)}>
              Требуют внимания <span>{attentionItems.length}</span>
            </button>
            <button className={`filter-chip ${showHistory ? 'active' : ''}`} aria-pressed={showHistory} onClick={() => setShowHistory(true)}>
              История <span>{historyItems.length}</span>
            </button>
          </div>
          {visibleItems.length ? (
            <div className="alerts-table">
              {visibleItems.map((alert) => (
                <div className={`alert-row ${selectedAlert?.id === alert.id ? 'alert-row-active' : ''}`} key={alert.id}>
                  <button className="alert-drilldown" aria-pressed={selectedAlert?.id === alert.id} onClick={() => setSelectedAlertId(alert.id)}>
                    <span className={`alert-dot alert-${alert.severity}`} />
                    <span className="alert-row-main">
                      <strong>{alert.title}</strong>
                      <span>{alert.region} · {alert.topic}</span>
                      <small>{alertPeriodLabel(alert)} · {alert.status}</small>
                    </span>
                    <span className="alert-count">{(alert.currentCount ?? alert.affectedTickets).toLocaleString('ru-RU')}<small>обращений</small></span>
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <NoData message={showHistory ? 'Закрытых оповещений пока нет.' : 'Нет сигналов, требующих внимания.'} />
          )}
        </section>

        {selectedAlert ? (
          <section className="panel alert-detail-panel signal-card" aria-label="Карточка сигнала">
            <div className="signal-card-topline">
              <span className={`alert-severity-badge alert-severity-${selectedAlert.severity}`}>
                {selectedAlert.severity === 'critical' ? 'Высокий приоритет' : selectedAlert.severity === 'watch' ? 'Повышенный приоритет' : 'Обычный приоритет'}
              </span>
              <span className="signal-status">{selectedAlert.status}</span>
            </div>
            <h2>{selectedAlert.title}</h2>
            <p>{selectedAlert.description}</p>
            <div className="signal-location">
              <span><small>Регион</small><strong>{selectedAlert.region}</strong></span>
              <span><small>Тема</small><strong>{selectedAlert.topic}</strong></span>
              <span><small>Период</small><strong>{alertPeriodLabel(selectedAlert)}</strong></span>
            </div>

            <div className="alert-facts">
              <div><span>Текущий уровень · {periodDays} дн.</span><strong>{currentCount.toLocaleString('ru-RU')}</strong></div>
              <div><span>Обычный уровень · {periodDays} дн.</span><strong>{formatAlertNumber(selectedAlert.baseline)}</strong></div>
              <div><span>Отклонение от baseline</span><strong>{formatAlertNumber(selectedAlert.deviation)}</strong></div>
              <div><span>Версия detector</span><strong>{selectedAlert.detectorVersion ?? 'не сохранена'}</strong></div>
            </div>

            {chart ? (
              <div className="alert-detail-chart">
                <div className="field-label">Динамика обращений и обычный уровень</div>
                <DataChart option={chart} label={`Динамика обращений по теме ${selectedAlert.topic} в регионе ${selectedAlert.region}`} />
              </div>
            ) : (
              <p className="signal-missing-evidence">Истории недостаточно для графика; сохранённое detector evidence не содержит периодных значений.</p>
            )}

            <div className="signal-reasons">
              <div className="field-label">Почему сработал detector</div>
              <ul>{alertTriggerDescriptions(selectedAlert).map((reason) => <li key={reason}>{reason}</li>)}</ul>
              {selectedAlert.robustZ != null && <small>Robust z: {formatAlertNumber(selectedAlert.robustZ)} · отношение к baseline: {formatAlertNumber(selectedAlert.ratio)}×</small>}
            </div>

            <div className="signal-sources">
              <div className="field-label">Исходные обращения · {sourceTicketIds.length}</div>
              {sourceTicketIds.length ? (
                <ul>{sourceTicketIds.slice(0, 5).map((ticketId) => <li key={ticketId}>№ {ticketId}</li>)}</ul>
              ) : (
                <p>Для этого alert не сохранены связанные обращения.</p>
              )}
              {sourceTicketIds.length > 5 && <small>И ещё {sourceTicketIds.length - 5}</small>}
            </div>

            <div className="signal-actions">
              <button className="signal-open-tickets" disabled={!sourceTicketIds.length} onClick={() => onDrilldown('alert', selectedAlert.id, `Обращения по сигналу: ${selectedAlert.title}`)}>
                Открыть обращения
              </button>
              {!showHistory && selectedAlert.status === 'Новый' && <button className="text-button" onClick={() => update(selectedAlert, 'ack')}>Принять</button>}
              {!showHistory && selectedAlert.status !== 'Закрыт' && <button className="text-button" onClick={() => update(selectedAlert, 'close')}>Закрыть</button>}
            </div>
          </section>
        ) : (
          <section className="panel alert-detail-panel"><NoData message="Выберите сигнал или запись истории." /></section>
        )}
      </div>
    </div>
  )
}

function formatForecastNumber(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value)
    ? '—'
    : new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 1 }).format(value)
}

function formatForecastDate(value: string): string {
  const date = new Date(`${value}T12:00:00`)
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat('ru-RU', { day: 'numeric', month: 'short', year: 'numeric' }).format(date)
}

function formatForecastRate(value: number | null | undefined): string {
  return typeof value !== 'number' || !Number.isFinite(value)
    ? '—'
    : new Intl.NumberFormat('ru-RU', { style: 'percent', maximumFractionDigits: 1 }).format(value)
}

function forecastChartOption(history: ForecastPoint[], forecast: ForecastPoint[], forecastStart?: string): EChartsOption | undefined {
  const hasHistory = history.length > 0
  const hasForecast = forecast.length > 0
  if (!hasHistory && !hasForecast) return undefined

  const labels = [...history, ...forecast].map((point) => point.label)
  const historicalValues = [
    ...history.map((point) => point.actual ?? null),
    ...forecast.map(() => null),
  ]
  const futureValues = [
    ...history.map(() => null),
    ...forecast.map((point) => point.forecast ?? null),
  ]
  const series: NonNullable<EChartsOption['series']> = []
  if (hasHistory) {
    series.push({
      name: 'История',
      type: 'line',
      data: historicalValues,
      showSymbol: false,
      lineStyle: { width: 2 },
      itemStyle: { color: '#8cf0c8' },
    })
  }
  if (hasForecast) {
    series.push({
      name: 'Прогноз',
      type: 'line',
      data: futureValues,
      showSymbol: false,
      lineStyle: { width: 2, type: 'dashed' },
      areaStyle: { color: 'rgba(167, 217, 255, .12)' },
      itemStyle: { color: '#a7d9ff' },
      markLine: forecastStart ? {
        symbol: 'none',
        lineStyle: { color: '#f8d488', type: 'dashed', width: 1 },
        label: { color: '#f8d488', formatter: 'Начало прогноза' },
        data: [{ xAxis: forecastStart }],
      } : undefined,
    })
  }

  return {
    ...chartBaseOption(labels),
    animation: false,
    tooltip: { trigger: 'axis' },
    legend: { data: series.map((item) => item.name).filter((name): name is string => Boolean(name)), top: 0, textStyle: { color: '#a9bbb2' } },
    grid: { left: 45, right: 18, top: 43, bottom: 48 },
    xAxis: { type: 'category', data: labels, axisLabel: { color: '#899c95', interval: 'auto', hideOverlap: true }, axisLine: { lineStyle: { color: '#40514b' } } },
    yAxis: { type: 'value', min: 0, axisLabel: { color: '#899c95' }, splitLine: { lineStyle: { color: 'rgba(214,236,225,.1)' } } },
    series,
  }
}

function CleanForecastPage({ forecast, history, status, modelVersion, model, source, insufficientHistory, forecastStart, expectedPeaks, backtest, horizon, filters, filterOptions, onHorizonChange }: {
  forecast: ForecastPoint[]
  history: ForecastPoint[]
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
  const forecastAvailable = !insufficientHistory && (status === 'OK' || status === 'DEMO_ONLY') && forecast.length > 0
  const chart = forecastChartOption(history, forecastAvailable ? forecast : [], forecastStart ?? forecast[0]?.label)
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
    id ? options.find((option) => option.id === id)?.label ?? id : 'Все'
  const activeFilters = [
    `Регион: ${optionLabel(filterOptions.regions, filters.regionId)}`,
    `Тема: ${optionLabel(filterOptions.topics, filters.topicId)}`,
    `Служба: ${optionLabel(filterOptions.services, filters.serviceId)}`,
    `Статус: ${optionLabel(filterOptions.statuses, filters.status)}`,
    `Район: ${optionLabel(filterOptions.districts, filters.district)}`,
    `Канал: ${optionLabel(filterOptions.channels, filters.channel)}`,
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
          <PanelHeading title="Прогноз нагрузки" />
          <div className="forecast-horizon-control" role="group" aria-label="Горизонт прогноза">
            {([30, 60, 90] as const).map((days) => (
              <button key={days} className={`forecast-horizon-button ${horizon === days ? 'active' : ''}`} aria-pressed={horizon === days} onClick={() => onHorizonChange(days)}>
                {days} дней
              </button>
            ))}
          </div>
        </div>
        <div className="forecast-provenance">
          <span><small>Модель</small><strong>{modelName}</strong></span>
          <span><small>Версия</small><strong>{modelVersion ?? 'не указана'}</strong></span>
          <span><small>Состояние</small><strong>{statusLabel}</strong></span>
          <span><small>Источник</small><strong>{sourceLabel}</strong></span>
        </div>
        <p className="forecast-filter-summary">Применённый срез: {activeFilters.join(' · ')}</p>
        <p className="panel-note">Используется дневной ряд за доступную часть окна до 366 дней. Почасовая детализация для текущего ряда недоступна.</p>

        {insufficientHistory && (
          <div className="forecast-insufficient" role="status">
            <strong>Недостаточно истории для сезонного прогноза.</strong>
            <span>Показана наблюдаемая история; будущие значения и ожидаемые пики не подставляются.</span>
            {backtest?.observed_days != null && backtest.required_days != null && <small>Активных дневных наблюдений: {backtest.observed_days} из {backtest.required_days} требуемых.</small>}
          </div>
        )}

        {!insufficientHistory && !forecastAvailable && <NoData message="Прогноз для выбранного среза не предоставлен." />}

        {forecastAvailable && (
          <div className="forecast-volume-grid">
            <div><span>Ожидаемый объём за {horizon} дней</span><strong>{formatForecastNumber(expectedVolume)}</strong><small>обращений по Seasonal Naive</small></div>
            <div><span>Средняя дневная нагрузка</span><strong>{formatForecastNumber(averageDailyLoad)}</strong><small>обращений в день</small></div>
            <div><span>Пиковый дневной объём</span><strong>{formatForecastNumber(peakLoad)}</strong><small>{relativePeakLoad == null ? 'относительный пик не выделен' : `${formatForecastNumber(relativePeakLoad)}× от среднего`}</small></div>
          </div>
        )}

        {forecastAvailable && forecastBoundary && <p className="forecast-boundary-note">Граница прогноза: {formatForecastDate(forecastBoundary)}</p>}
        {chart && <DataChart option={chart} label="Дневная история обращений и прогноз Seasonal Naive с границей будущего" />}
        {!chart && !insufficientHistory && <p className="forecast-empty-history">История и точки прогноза для диаграммы отсутствуют.</p>}
        {forecastAvailable && (
          <details className="chart-values">
            <summary>Показать дневную историю и прогноз</summary>
            <table>
              <thead><tr><th>Дата</th><th>Тип ряда</th><th>Обращения</th></tr></thead>
              <tbody>
                {history.map((point) => <tr key={`history-${point.label}`}><td>{point.label}</td><td>История</td><td>{point.actual ?? '—'}</td></tr>)}
                {forecast.map((point) => <tr key={`forecast-${point.label}`}><td>{point.label}</td><td>Прогноз</td><td>{point.forecast ?? '—'}</td></tr>)}
              </tbody>
            </table>
          </details>
        )}
      </section>

      <div className="forecast-support-grid">
        <section className="panel forecast-peaks-panel">
          <PanelHeading title="Ожидаемые пиковые дни" />
          <p className="panel-note">Пиковый объём сравнивается со средним дневным прогнозом для этого же среза.</p>
          {forecastAvailable && peakDetails.length ? (
            <div className="forecast-peak-list">
              {peakDetails.slice(0, 5).map(({ date, point }) => (
                <div className="forecast-peak-row" key={date}>
                  <span><strong>{formatForecastDate(date)}</strong><small>дневной интервал</small></span>
                  <span><strong>{formatForecastNumber(point.forecast)}</strong><small>обращений</small></span>
                  <span><strong>{averageDailyLoad > 0 ? `${formatForecastNumber((point.forecast ?? 0) / averageDailyLoad)}×` : '—'}</strong><small>от среднего</small></span>
                </div>
              ))}
              {peakDetails.length > 5 && <p className="forecast-peak-more">Показаны 5 из {peakDetails.length} пиковых дней.</p>}
            </div>
          ) : (
            <p className="forecast-empty-history">{forecastAvailable ? 'Положительные ожидаемые пики не выделены.' : 'Пики доступны только при достаточной истории.'}</p>
          )}
        </section>

        <section className="panel forecast-backtest-panel">
          <PanelHeading title="Качество baseline на backtest" />
          <p className="panel-note">Метрики оценивают Seasonal Naive на исторических дневных точках; это не гарантия будущей точности.</p>
          {forecastAvailable && backtestHasMetrics ? (
            <>
              <div className="forecast-backtest-grid">
                <div><span>MAE</span><strong>{formatForecastNumber(backtest?.mae)}</strong><small>обращений в день</small></div>
                <div><span>RMSE</span><strong>{formatForecastNumber(backtest?.rmse)}</strong><small>обращений в день</small></div>
                <div><span>WAPE</span><strong>{formatForecastRate(backtest?.wape)}</strong></div>
                <div><span>SMAPE</span><strong>{formatForecastRate(backtest?.smape)}</strong></div>
              </div>
              <p className="forecast-backtest-samples">Проверено точек: {backtest?.sample_count ?? '—'} · окон: {backtest?.window_count ?? '—'}</p>
            </>
          ) : (
            <p className="forecast-empty-history">
              {isDemoBacktest
                ? 'Оценочные метрики для демо-базовой линии не предоставлены.'
                : insufficientHistory
                  ? 'Backtest не рассчитан из-за недостаточной истории.'
                  : 'Для выбранного среза нет backtest метрик.'}
            </p>
          )}
        </section>
      </div>
    </div>
  )
}

function CleanReportsPage({ filters }: { filters: DashboardFilters }) {
  const filterSummary = [filters.range, filters.regionId ?? 'все регионы', filters.topicId ?? 'все темы', filters.serviceId ?? 'все службы', filters.status ?? 'все статусы', filters.district ?? 'все районы', filters.channel ?? 'все каналы'].join(', ')
  return <div className="analytics-page"><section className="panel"><PanelHeading title="Отчёты" /><p className="panel-note">В выгрузку войдут данные за {filterSummary}.</p><div className="report-actions"><a className="button button-primary" href={reportUrl('pdf', filters)}>Скачать PDF</a><a className="button button-secondary" href={reportUrl('xlsx', filters)}>Скачать XLSX</a></div></section></div>
}

function evaluationMetric(value: unknown, percentage = false): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'недостаточно данных'
  return percentage ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)
}

function evaluationGateLabel(key: string): string {
  const labels: Record<string, string> = {
    promotion_policy: 'Версия promotion policy поддерживается',
    offline_sample_count: 'Размер offline выборки',
    shadow_sample_count: 'Размер shadow выборки',
    macro_f1_non_inferiority: 'Macro-F1 candidate относительно production',
    critical_class_regressions: 'Регрессии по классам',
    shadow_correction_rate_delta: 'Изменение correction rate в shadow',
    candidate_shadow_inference_failures: 'Ошибки candidate inference',
    synthetic_evidence: 'Происхождение evidence',
    candidate_evaluation_job: 'Фоновая оценка candidate',
  }
  return labels[key] ?? key
}

function evaluationGateStatus(status: string): string {
  switch (status) {
    case 'PASSED': return 'Пройден'
    case 'FAILED': return 'Не пройден'
    case 'PENDING': return 'Выполняется'
    default: return 'Недостаточно данных'
  }
}

function promotionThresholdLabel(key: string): string {
  const labels: Record<string, string> = {
    minimum_offline_samples: 'Минимум offline записей',
    minimum_shadow_samples: 'Минимум shadow решений',
    maximum_macro_f1_regression: 'Допустимое снижение macro-F1',
    maximum_class_f1_regression: 'Допустимое снижение F1 класса',
    maximum_shadow_correction_rate_delta: 'Допустимый рост correction rate',
    maximum_shadow_inference_failures: 'Допустимые ошибки inference',
  }
  return labels[key] ?? key
}

function CleanLearningPage({ learning, onRefresh, onToast }: { learning: LearningCycle; onRefresh: () => Promise<void>; onToast: (message: string) => void }) {
  const [evaluation, setEvaluation] = useState<Awaited<ReturnType<typeof loadCandidateEvaluation>> | null>(null)
  const [evaluationError, setEvaluationError] = useState<string | null>(null)
  const [busy, setBusy] = useState<'create' | 'close' | 'evaluation' | 'promote' | 'reject' | null>(null)
  const [note, setNote] = useState('')

  useEffect(() => {
    let active = true
    let timer: number | undefined
    setEvaluation(null)
    setEvaluationError(null)
    if (!['EVALUATE', 'DECISION'].includes(learning.stage) || learning.id === 'нет данных') return () => { active = false }
    const load = async () => {
      try {
        const result = await loadCandidateEvaluation()
        if (!active) return
        setEvaluation(result)
        if (learning.stage === 'DECISION' && result.status === 'PENDING') {
          timer = window.setTimeout(() => void load(), 4000)
        }
      } catch (error: unknown) {
        if (active) setEvaluationError(error instanceof Error ? error.message : 'Оценка пока недоступна')
      }
    }
    void load()
    return () => {
      active = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [learning.id, learning.stage, learning.updatedAt])

  const runAction = async (action: 'create' | 'close' | 'evaluation' | 'promote' | 'reject') => {
    setBusy(action)
    try {
      if (action === 'create') {
        await createLearningCycle()
        onToast('Открыт новый цикл COLLECT с настроенным окном сбора')
      } else if (action === 'close') {
        const result = await closeLearningCycle(learning.id)
        onToast(result.state === 'INSUFFICIENT_FEEDBACK'
          ? `Сбор закрыт: ${result.cycle.feedback_count}/${result.cycle.min_feedback_count} валидных записей, кандидат не создан`
          : result.state === 'DECISION'
            ? 'Окно shadow evaluation закрыто; Data/ML evaluation поставлена в очередь'
            : 'Сбор обратной связи закрыт; обучение поставлено в очередь')
      } else if (action === 'evaluation') {
        setEvaluation(await loadCandidateEvaluation())
        onToast('Оценка candidate перечитана из backend')
      } else if (action === 'promote' || action === 'reject') {
        if (!window.confirm(action === 'promote' ? 'Продвинуть candidate в production?' : 'Отклонить candidate?')) return
        const result = action === 'promote' ? await promoteCandidate(note) : await rejectCandidate(note)
        onToast(result.state === 'PROMOTED' ? 'Candidate продвинут в production' : 'Candidate отклонён; production не изменён')
        setNote('')
      }
      await onRefresh()
    } catch (error) {
      onToast(`Не удалось выполнить действие: ${error instanceof Error ? error.message : 'ошибка API'}`)
    } finally {
      setBusy(null)
    }
  }

  if (learning.id === 'нет данных') return <div className="analytics-page"><section className="panel"><NoData message="Активного цикла обучения нет." /><button className="button button-primary" disabled={busy !== null} onClick={() => void runAction('create')}>{busy === 'create' ? 'Создаём…' : 'Открыть цикл COLLECT'}</button></section></div>

  const offlineStatus = evaluation?.offline_evaluation.status
  const readyToReview = learning.stage === 'DECISION'
    && evaluation?.status === 'COMPLETED'
    && evaluation.decision === 'PENDING_HUMAN_DECISION'
    && !evaluation.synthetic
    && evaluation.gates.length > 0
    && evaluation.gates.every((gate) => gate.status === 'PASSED')
  const candidateF1 = evaluation?.offline_evaluation.metrics.macro_f1
  const productionF1 = evaluation?.baseline_evaluation.metrics.macro_f1
  const classMetrics = evaluation?.offline_evaluation.metrics.per_class_f1 ?? {}
  const productionClassMetrics = evaluation?.offline_evaluation.metrics.production_per_class_f1
    ?? evaluation?.baseline_evaluation.metrics.per_class_f1
    ?? {}
  const classChanges = evaluation?.offline_evaluation.metrics.per_class_changes ?? {}
  const classSupport = evaluation?.offline_evaluation.metrics.per_class_support ?? {}
  const classLabels = [...new Set([
    ...Object.keys(classMetrics),
    ...Object.keys(productionClassMetrics),
    ...Object.keys(classChanges),
  ])].sort()
  const criticalRegressions = evaluation
    ? [...new Set([
      ...evaluation.offline_evaluation.critical_regressions,
      ...evaluation.shadow_evaluation.critical_regressions,
    ])]
    : []
  const collectEndTimestamp = Date.parse(learning.collectEndsAt)
  const collectEndReached = Number.isFinite(collectEndTimestamp) && collectEndTimestamp <= Date.now()
  const canCloseCollect = learning.manualCloseEnabled || collectEndReached
  const evaluationEndTimestamp = learning.evaluationEndsAt ? Date.parse(learning.evaluationEndsAt) : Number.NaN
  const evaluationEndReached = Number.isFinite(evaluationEndTimestamp) && evaluationEndTimestamp <= Date.now()
  const canCloseEvaluation = learning.manualCloseEnabled || evaluationEndReached
  const collectWindowLabel = learning.collectStartedAt && learning.collectEndsAt
    ? `${new Date(learning.collectStartedAt).toLocaleString('ru-RU')} — ${new Date(learning.collectEndsAt).toLocaleString('ru-RU')}`
    : 'не задано'
  const evaluationWindowLabel = learning.evaluationStartedAt && learning.evaluationEndsAt
    ? `${new Date(learning.evaluationStartedAt).toLocaleString('ru-RU')} — ${new Date(learning.evaluationEndsAt).toLocaleString('ru-RU')}`
    : learning.evaluationStartedAt
      ? `${new Date(learning.evaluationStartedAt).toLocaleString('ru-RU')} — конец не задан`
      : 'ещё не началось'
  return <div className="analytics-page">
    <section className="panel">
      <PanelHeading title={'Цикл ' + learning.id} />
      <div className="dataset-stat"><span>Состояние</span><strong>{learning.stage}</strong></div>
      <div className="dataset-stat"><span>Обратная связь</span><strong>{learning.feedbackCount}</strong></div>
      <div className="dataset-stat"><span>Порог обратной связи</span><strong>{learning.minFeedbackCount}</strong></div>
      <div className="dataset-stat"><span>Окно COLLECT</span><strong>{collectWindowLabel}</strong></div>
      <div className="dataset-stat"><span>Окно shadow evaluation</span><strong>{evaluationWindowLabel}</strong></div>
      <div className="dataset-stat"><span>Shadow predictions</span><strong>{learning.shadowPredictionCount ?? 0}</strong></div>
      <div className="dataset-stat"><span>Ошибки shadow inference</span><strong>{learning.shadowInferenceFailures ?? 0}</strong></div>
      <div className="dataset-stat"><span>Связанные решения оператора</span><strong>{learning.shadowOperatorDecisionCount ?? 0}</strong></div>
      <div className="dataset-stat"><span>Blind A/B signal</span><strong>{learning.blindAbEnabled ? 'включён' : 'отключён'}</strong></div>
      <div className="dataset-stat"><span>Production baseline</span><strong>{learning.productionModelVersion ?? 'не зафиксирована'}</strong></div>
      <div className="dataset-stat"><span>Frozen evaluation dataset</span><strong>{learning.frozenEvaluationDatasetVersion ?? 'не настроен'}</strong></div>
      {learning.candidateDatasetChecksum && <div className="dataset-stat"><span>Candidate dataset SHA-256</span><strong>{learning.candidateDatasetChecksum}</strong></div>}
      <div className="dataset-stat"><span>Датасет</span><strong>{learning.dataset}</strong></div>
      <div className="dataset-stat"><span>Кандидат</span><strong>{learning.candidate}</strong></div>
      {learning.decisionNote && <p className="panel-note learning-decision-note">Решение: {learning.decisionNote}</p>}
      <div className="learning-actions" aria-label="Действия reviewer">
        {learning.stage === 'COLLECT' && learning.id !== 'нет данных' && <button className="button button-primary" disabled={busy !== null || !canCloseCollect} onClick={() => void runAction('close')}>{busy === 'close' ? 'Закрываем…' : 'Закрыть цикл'}</button>}
        {learning.stage === 'COLLECT' && learning.id !== 'нет данных' && !canCloseCollect && <p className="panel-note">Сбор завершится автоматически по окончании окна COLLECT.</p>}
        {learning.stage === 'EVALUATE' && learning.id !== 'нет данных' && <button className="button button-primary" disabled={busy !== null || !canCloseEvaluation} onClick={() => void runAction('close')}>{busy === 'close' ? 'Закрываем…' : 'Закрыть окно evaluation'}</button>}
        {learning.stage === 'EVALUATE' && learning.id !== 'нет данных' && !canCloseEvaluation && <p className="panel-note">Shadow evaluation завершится автоматически по окончании окна.</p>}
        {!learning.blindAbEnabled && <p className="panel-note">Слепое сравнение A/B отключено; предпочтения не собираются.</p>}
        {['PROMOTED', 'REJECTED', 'INSUFFICIENT_FEEDBACK', 'DATASET_BUILD_FAILED', 'TRAINING_FAILED'].includes(learning.stage) && <button className="button button-primary" disabled={busy !== null} onClick={() => void runAction('create')}>{busy === 'create' ? 'Создаём…' : 'Открыть цикл COLLECT'}</button>}
        {['EVALUATE', 'DECISION'].includes(learning.stage) && <button className="button button-secondary" disabled={busy !== null} onClick={() => void runAction('evaluation')}>{busy === 'evaluation' ? 'Читаем…' : 'Показать evaluation'}</button>}
        {learning.stage === 'DECISION' && <>
          <input className="learning-note" aria-label="Комментарий reviewer" placeholder="Комментарий к решению (необязательно)" value={note} onChange={(event) => setNote(event.target.value)} disabled={busy !== null} />
          <button className="button button-primary" disabled={busy !== null || !readyToReview} onClick={() => void runAction('promote')}>{busy === 'promote' ? 'Продвигаем…' : 'Promote'}</button>
          <button className="button button-quiet" disabled={busy !== null} onClick={() => void runAction('reject')}>{busy === 'reject' ? 'Отклоняем…' : 'Reject'}</button>
        </>}
      </div>
    </section>
    {['EVALUATE', 'DECISION'].includes(learning.stage) && <section className="panel learning-evaluation">
      <PanelHeading title="Evaluation candidate" />
      {evaluationError && <p className="panel-note">Оценка пока недоступна: {evaluationError}. Повторите запрос после восстановления backend.</p>}
      {!evaluation && !evaluationError && <p className="panel-note">Загружаем evaluation из backend…</p>}
      {evaluation && <>
        <div className="dataset-stat"><span>Статус оценки</span><strong>{evaluation.status === 'PENDING' ? 'В очереди или выполняется' : evaluation.status === 'FAILED' ? 'Фоновая оценка завершилась ошибкой' : 'Оценка завершена'}</strong></div>
        <div className="dataset-stat"><span>Решение policy</span><strong>{evaluation.decision === 'PENDING_HUMAN_DECISION' ? 'Все gates пройдены; ожидает решения reviewer' : evaluation.decision === 'FAIL' ? 'Есть проваленные gates' : evaluation.decision === 'PASS' ? 'Policy пройдена; ожидает действия reviewer' : 'Недостаточно evidence'}</strong></div>
        <div className="dataset-stat"><span>Статус evidence</span><strong>{String(offlineStatus ?? 'нет данных')}</strong></div>
        <div className="learning-evaluation-grid">
          <article className="learning-evaluation-metric">
            <span>Candidate macro-F1</span>
            <strong>{evaluationMetric(candidateF1)}</strong>
            <small>{evaluation.offline_evaluation.sample_count} frozen offline записей</small>
          </article>
          <article className="learning-evaluation-metric">
            <span>Production macro-F1</span>
            <strong>{evaluationMetric(productionF1)}</strong>
            <small>{evaluation.baseline_evaluation.sample_count} offline записей</small>
          </article>
          <article className="learning-evaluation-metric">
            <span>Shadow agreement</span>
            <strong>{evaluationMetric(evaluation.shadow_evaluation.agreement_with_confirmed, true)}</strong>
            <small>{evaluation.shadow_evaluation.sample_count} подтверждённых решений</small>
          </article>
          <article className="learning-evaluation-metric">
            <span>Изменение correction rate</span>
            <strong>{evaluationMetric(evaluation.shadow_evaluation.correction_rate_delta, true)}</strong>
            <small>candidate минус production</small>
          </article>
        </div>
        <div className="dataset-stat"><span>Candidate</span><strong>{evaluation.candidate_model_version}</strong></div>
        <div className="dataset-stat"><span>Baseline production</span><strong>{evaluation.production_model_version}</strong></div>
        <div className="dataset-stat"><span>Frozen evaluation dataset</span><strong>{evaluation.offline_evaluation.dataset_version}</strong></div>
        <div className="dataset-stat"><span>Promotion policy</span><strong>{evaluation.policy_version}</strong></div>
        {Object.entries(evaluation.promotion_policy.thresholds).length > 0 && <div className="learning-policy-thresholds">
          <h3>Пороги до оценки candidate</h3>
          <ul>{Object.entries(evaluation.promotion_policy.thresholds).map(([key, value]) => <li key={key}>
            <span>{promotionThresholdLabel(key)}</span>
            <strong>{key.includes('regression') || key.includes('delta') ? evaluationMetric(value, true) : value}</strong>
          </li>)}</ul>
        </div>}
        <div className="learning-gates">
          <h3>Promotion gates</h3>
          <ul>{evaluation.gates.map((item) => <li className={`learning-gate learning-gate-${item.status.toLowerCase()}`} key={item.key}>
            <span><strong>{evaluationGateLabel(item.key)}</strong>{item.reason && <small>{item.reason}</small>}</span>
            <span className="learning-gate-result">
              <strong>{evaluationGateStatus(item.status)}</strong>
              {(item.observed !== undefined || item.threshold !== undefined) && <small>{item.observed ?? '—'} / {item.threshold ?? '—'}</small>}
            </span>
          </li>)}</ul>
        </div>
        <div className="learning-gates">
          <h3>Critical regressions</h3>
          {criticalRegressions.length > 0
            ? <ul>{criticalRegressions.map((name) => <li className="learning-regression" key={name}><strong>{name}</strong></li>)}</ul>
            : offlineStatus === 'COMPLETED' && evaluation.offline_evaluation.sample_count > 0
              ? <p className="panel-note">Критические регрессии не обнаружены.</p>
              : <p className="panel-note">Не оценивались: offline evidence недостаточно.</p>}
        </div>
        <div className="learning-gates">
          <h3>Per-class F1</h3>
          {classLabels.length > 0
            ? <div className="learning-class-table-wrap"><table className="learning-class-table">
              <thead><tr><th>Класс</th><th>Production</th><th>Candidate</th><th>Изменение</th><th>Support</th></tr></thead>
              <tbody>{classLabels.map((label) => <tr key={label}>
                <th scope="row">{label}</th>
                <td>{evaluationMetric(productionClassMetrics[label])}</td>
                <td>{evaluationMetric(classMetrics[label])}</td>
                <td>{evaluationMetric(classChanges[label])}</td>
                <td>{classSupport[label] ?? '—'}</td>
              </tr>)}</tbody>
            </table></div>
            : <p className="panel-note">Per-class metrics появятся после получения достаточного offline evidence.</p>}
        </div>
        <div className="dataset-stat"><span>Blind A/B</span><strong>{evaluation.shadow_evaluation.blind_ab === 'ENABLED' ? 'включён' : 'отключён'}</strong></div>
        {evaluation.synthetic && <p className="panel-note">Evidence синтетический и не допускается для production promotion.</p>}
      </>}
    </section>}
  </div>
}

function CleanModelsPage({ models }: { models: ModelStatus[] }) {
  if (!models.length) return <div className="analytics-page"><NoData message="Реестр моделей не предоставлен API." /></div>
  return <div className="analytics-page"><section className="panel"><PanelHeading title="Реестр моделей" /><div className="models-table">{models.map((model) => <div className="models-row" key={model.version}><strong>{model.name}</strong><code>{model.version}</code><span>{model.status}</span><span><strong>{model.metricValue}</strong><small>{model.metric}</small></span><span>{model.updatedAt}</span></div>)}</div></section></div>
}


function OverviewPage({ data, filters, onNavigate, onDrilldown }: { data: DashboardData; filters: DashboardFilters; onNavigate: (route: Route) => void; onDrilldown: DrilldownHandler }) {
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
      <MetricCard label="Обращений в выборке" value={total.toLocaleString('ru-RU')} change={formatAnalyticsChange(data.overview.changeAbs, data.overview.changePct)} detail={data.overview.previousTotalTickets == null ? 'сравнение недоступно' : `предыдущий период: ${data.overview.previousTotalTickets.toLocaleString('ru-RU')}`} tone="mint" icon="inbox" onClick={() => onDrilldown('overview', 'all', 'Все обращения')} />
      <MetricCard label="Высокий приоритет" value={String(highPriority)} change="текущий срез" detail="по выбранным фильтрам" tone="rose" icon="pulse" onClick={() => onDrilldown('overview', 'high_priority', 'Высокий приоритет')} />
      <MetricCard label="Решения оператора" value={String(data.overview.operatorDecisions)} change={`${data.overview.confirmedDecisions} подтверждено · ${data.overview.correctedDecisions} исправлено`} detail="по данным решений" tone="amber" icon="clock" />
      <MetricCard label="Аномальные сигналы" value={String(data.alerts.length)} change="обнаружено системой" detail="текущий срез" tone="blue" icon="bell" />
    </div>
    <section className="panel runtime-metrics-panel" aria-label="Время решений и качество обратной связи">
      <PanelHeading title="Время решений и качество обратной связи" />
      <p className="panel-note">Метрики считаются по решениям и relation feedback операторов; «—» означает, что для показателя пока нет событий в выборке.</p>
      <div className="runtime-metrics-grid">
        <MetricCard label="Время до первого решения" value={formatDecisionTime(metrics.operatorDecisionTimeMinutes)} change={metrics.operatorDecisionTimeSamples ? `${metrics.operatorDecisionTimeSamples} решений` : 'нет решений'} detail="Среднее от создания обращения" tone="blue" icon="clock" />
        <MetricCard label="Исправление темы" value={formatRuntimeRate(metrics.classificationCorrectionRate)} change={metrics.classificationDecisions ? `${metrics.classificationCorrections} / ${metrics.classificationDecisions}` : 'нет классификаций'} detail="Доля изменённых тем" tone="amber" icon="edit" />
        <MetricCard label="Исправление маршрута" value={formatRuntimeRate(metrics.routingCorrectionRate)} change={metrics.routingDecisions ? `${metrics.routingCorrections} / ${metrics.routingDecisions}` : 'нет решений с маршрутом'} detail="Доля изменённых служб" tone="rose" icon="arrow" />
        <MetricCard label="Исправление приоритета" value={formatRuntimeRate(metrics.priorityCorrectionRate)} change={metrics.priorityDecisions ? `${metrics.priorityCorrections} / ${metrics.priorityDecisions}` : 'нет решений с приоритетом'} detail="Доля изменённого приоритета" tone="amber" icon="pulse" />
        <MetricCard label="Полезность сходства" value={formatRuntimeRate(metrics.similarityUsefulness)} change={metrics.similarityFeedbackCount ? `${metrics.similarityFeedbackCount} оценок` : 'нет оценок'} detail="Подтверждения среди оценок связи" tone="mint" icon="search" />
        <MetricCard label="Precision дубликатов" value={formatRuntimeRate(metrics.duplicatePrecision)} change={metrics.duplicateFeedbackCount ? `${metrics.duplicateFeedbackCount} оценок` : 'нет оценок'} detail="Подтверждения среди проверок дубликата" tone="blue" icon="check" />
      </div>
    </section>
    <div className="analytics-grid overview-grid">
      <section className="panel span-two"><PanelHeading title="Поток обращений" action="Временная динамика" onClick={() => onNavigate('/situation/time-series')} />{data.timeSeries.length ? <div className="region-table"><div className="region-table-head"><span>Дата</span><span>Обращения</span><span>Закрыто</span><span>Доля</span></div>{data.timeSeries.slice(-7).map((point) => <button className="region-table-row drilldown-row" key={point.date} onClick={() => onDrilldown('date', point.date, `Дата: ${point.date}`)}><strong>{point.date}</strong><span>{point.tickets}</span><span>{point.resolved}</span><span>{point.tickets ? `${Math.round(point.resolved / point.tickets * 100)}%` : '—'}</span></button>)}</div> : <NoData message="За выбранный период нет обращений." />}</section>
      <section className="panel"><PanelHeading title="Темы" action="Все темы" onClick={() => onNavigate('/situation/topics')} />{data.topics.length ? <div className="topic-bars">{data.topics.slice(0, 5).map((topic) => <button className="topic-bar-row drilldown-row" key={topic.name} onClick={() => onDrilldown('topic', topic.id, `Тема: ${topic.name}`)}><div className="topic-bar-label"><span>{topic.name}</span><strong>{topic.value}%</strong></div><div className="bar-track"><span style={{ width: String(topic.value * 2.7) + '%', background: topic.color }} /></div></button>)}</div> : <NoData message="Нет данных по темам." />}</section>
      <section className="panel"><PanelHeading title="Сигналы" action="Открыть все" onClick={() => onNavigate('/situation/alerts')} />{data.alerts.length ? <div className="alert-list">{data.alerts.map((alert) => <AlertListItem alert={alert} key={alert.id} onDrilldown={onDrilldown} />)}</div> : <NoData message="Нет подключённых сигналов." />}</section>
      <section className="panel span-two region-panel"><PanelHeading title="Регионы" action="Все регионы" onClick={() => onNavigate('/situation/regions')} />{data.regions.length ? <div className="region-table"><div className="region-table-head"><span>Регион</span><span>Текущий период</span><span>Предыдущий период</span><span>Изменение</span></div>{data.regions.map((region) => <RegionRow region={region} key={region.name} onDrilldown={onDrilldown} />)}</div> : <NoData message="Нет данных по регионам." />}</section>
      <section className="panel query-panel">
        <div className="eyebrow"><span className="eyebrow-line" />Спросить данные</div>
        <h3>Ответ по обращениям</h3>
        <p>Можно спросить о количестве, динамике, регионах, темах, всплесках или прогнозе.</p>
        <div className="query-input"><Icon name="search" size={16} /><input aria-label="Вопрос по данным" placeholder="Сколько обращений по регионам?" value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') void submitQuery() }} /><button aria-label="Выполнить поиск" onClick={() => void submitQuery()} disabled={queryLoading}><Icon name="arrow" size={16} /></button></div>
        {queryError && <p className="query-error" role="alert">Не удалось получить ответ: {queryError}</p>}
        {queryResult && <QueryIntentResultView result={queryResult} filterOptions={data.filterOptions} title={queryTitle} />}
      </section>
    </div>
  </div>
}

function MetricCard({ label, value, change, detail, tone, icon, onClick }: { label: string; value: string; change: string; detail: string; tone: string; icon: IconName; onClick?: () => void }) {
  return <article className={`metric-card metric-${tone}${onClick ? ' drilldown-row' : ''}`} role={onClick ? 'button' : undefined} tabIndex={onClick ? 0 : undefined} onClick={onClick} onKeyDown={onClick ? (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onClick() } } : undefined}><div className="metric-top"><span>{label}</span><span className="metric-icon"><Icon name={icon} size={17} /></span></div><div className="metric-value">{value}</div><div className="metric-bottom"><span className="metric-change">{change}</span><span>{detail}</span></div></article>
}

function NoData({ message }: { message: string }) {
  return <div className="state-card"><div className="state-icon">—</div><p>{message}</p></div>
}

function PanelHeading({ title, action, onClick }: { title: string; action?: string; onClick?: () => void }) {
  return <div className="panel-heading"><h2>{title}</h2>{action && <button className="text-button" onClick={onClick}>{action}<Icon name="arrow" size={14} /></button>}</div>
}

function AlertListItem({ alert, onDrilldown }: { alert: Alert; onDrilldown: DrilldownHandler }) {
  return <button className="alert-list-item drilldown-row" onClick={() => onDrilldown('alert', alert.id, `Оповещение: ${alert.title}`)}><span className={`alert-dot alert-${alert.severity}`} /><span><strong>{alert.title}</strong><span>{alert.region} · {alert.affectedTickets} обращений</span></span><Icon name="arrow" size={14} /></button>
}

function RegionRow({ region, onDrilldown }: { region: RegionMetric; onDrilldown?: DrilldownHandler }) {
  const row = <><strong>{region.name}</strong><span>{region.tickets.toLocaleString('ru-RU')}</span><span>{region.previousTickets?.toLocaleString('ru-RU') ?? '—'}</span><span>{formatAnalyticsChange(region.changeAbs, region.change)}</span></>
  if (!onDrilldown) return <div className="region-table-row">{row}</div>
  return <button className="region-table-row drilldown-row" onClick={() => onDrilldown('region', region.id, `Регион: ${region.name}`)}>{row}</button>
}

function formatAnalyticsChange(changeAbs: number | undefined, changePct: number | undefined): string {
  if (changeAbs == null) return '—'
  const absolute = `${changeAbs > 0 ? '+' : ''}${changeAbs.toLocaleString('ru-RU')}`
  if (changePct == null) return absolute
  const percent = `${changePct > 0 ? '+' : ''}${changePct.toLocaleString('ru-RU', { maximumFractionDigits: 1 })}%`
  return `${absolute} (${percent})`
}











export default App
