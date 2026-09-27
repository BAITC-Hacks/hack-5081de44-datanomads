import { useCallback, useEffect, useMemo, useState, type ReactElement } from 'react'
import { acknowledgeAlert, closeAlert, closeLearningCycle, loadAnalyticsDrilldown, loadCandidateEvaluation, loadDashboard, promoteCandidate, rejectCandidate, reportUrl, runQueryIntent, submitDecision, submitRelationFeedback, subscribeToAlertChanges } from './api/client'
import type { BackendTicket, DashboardFilters, DrilldownDimension, QueryIntentResult } from './api/client'
import type { Alert, ApiSource, DashboardData, DatasetProvenance, ForecastPoint, LearningCycle, ModelStatus, Priority, RegionMetric, RuleProvenance, Ticket, TopicMetric } from './types'
import { DataChart } from './components/DataChart'
import type { EChartsOption } from 'echarts'
import { languageLabel, languageReviewNotice } from './language'
import { confidenceStateLabel, confidenceStateNotice, normalizeConfidenceState } from './classification'

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

type IconName = 'inbox' | 'pulse' | 'grid' | 'map' | 'tag' | 'trend' | 'bell' | 'forecast' | 'file' | 'cycle' | 'model' | 'search' | 'settings' | 'help' | 'chevron' | 'arrow' | 'check' | 'edit' | 'external' | 'download' | 'more' | 'clock' | 'close'
type DrilldownHandler = (dimension: DrilldownDimension, value: string | undefined, label: string) => void
type DrilldownState = { label: string; items: BackendTicket[]; total: number } | null
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
  }, [filters])

  useEffect(() => {
    if (!route.startsWith('/situation') || source !== 'api') return
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
        {source === 'demo' && <div className="demo-banner"><span className="status-dot" /> Демо-данные · API подключится автоматически, когда backend будет доступен <span className="demo-banner-detail">{apiError ? `(${apiError})` : ''}</span></div>}
        {provenanceNotice && (
          <div className="demo-banner">
            <span className="status-dot" />
            {provenanceNotice}
          </div>
        )}
        {source === 'api' && apiError && <div className="error-banner"><span className="status-dot" /> API недоступен · {apiError}</div>}
        <div className="page-wrap">
          <PageHeader {...title} route={route} filters={filters} onFiltersChange={setFilters} filterOptions={data?.filterOptions} />
          {loading ? <LoadingState /> : data ? <RouteContent route={route} data={data} onDataChange={setData} onRefresh={refreshDashboard} onToast={showToast} filters={filters} onDrilldown={openDrilldown} drilldown={drilldown} drilldownLoading={drilldownLoading} /> : <ErrorState onRetry={() => window.location.reload()} />}
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
  return <div className="page-header"><div><div className="eyebrow"><span className="eyebrow-line" />{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{route === '/operator' ? null : <div className="period-control"><div className="period-buttons">{[['7d', '7 дней'], ['30d', '30 дней'], ['90d', '90 дней']].map(([value, label]) => <button key={value} className={`period-button ${filters.range === value ? 'active' : ''}`} onClick={() => onFiltersChange({ ...filters, range: value })}>{label}</button>)}</div>{situation && filterOptions && <div className="analytics-filter-selects"><label><span>Регион</span><select value={filters.regionId ?? ''} onChange={(event) => onFiltersChange({ ...filters, regionId: event.target.value || undefined })}><option value="">Все регионы</option>{filterOptions.regions.map((option) => <option value={option.id} key={option.id}>{option.label}</option>)}</select></label><label><span>Тема</span><select value={filters.topicId ?? ''} onChange={(event) => onFiltersChange({ ...filters, topicId: event.target.value || undefined })}><option value="">Все темы</option>{filterOptions.topics.map((option) => <option value={option.id} key={option.id}>{option.label}</option>)}</select></label>{filterOptions.services.length > 0 && <label><span>Служба</span><select value={filters.serviceId ?? ''} onChange={(event) => onFiltersChange({ ...filters, serviceId: event.target.value || undefined })}><option value="">Все службы</option>{filterOptions.services.map((option) => <option value={option.id} key={option.id}>{option.label}</option>)}</select></label>}{filterOptions.statuses.length > 0 && <label><span>Статус</span><select value={filters.status ?? ''} onChange={(event) => onFiltersChange({ ...filters, status: event.target.value || undefined })}><option value="">Все статусы</option>{filterOptions.statuses.map((option) => <option value={option.id} key={option.id}>{option.label}</option>)}</select></label>}{filterOptions.districts.length > 0 && <label><span>Район</span><select value={filters.district ?? ''} onChange={(event) => onFiltersChange({ ...filters, district: event.target.value || undefined })}><option value="">Все районы</option>{filterOptions.districts.map((option) => <option value={option.id} key={option.id}>{option.label}</option>)}</select></label>}{filterOptions.channels.length > 0 && <label><span>Канал</span><select value={filters.channel ?? ''} onChange={(event) => onFiltersChange({ ...filters, channel: event.target.value || undefined })}><option value="">Все каналы</option>{filterOptions.channels.map((option) => <option value={option.id} key={option.id}>{option.label}</option>)}</select></label>}</div>}</div>}</div>
}

function RouteContent({ route, data, onDataChange, onRefresh, onToast, filters, onDrilldown, drilldown, drilldownLoading }: { route: Route; data: DashboardData; onDataChange: (data: DashboardData) => void; onRefresh: () => Promise<void>; onToast: (message: string) => void; filters: DashboardFilters; onDrilldown: DrilldownHandler; drilldown: DrilldownState; drilldownLoading: boolean }) {
  let content: ReactElement
  switch (route) {
    case '/operator': content = <OperatorPage tickets={data.tickets} overview={data.overview} taxonomy={data.filterOptions} onDataChange={(tickets) => onDataChange({ ...data, tickets })} onToast={onToast} />; break
    case '/situation/overview': content = <OverviewPage data={data} filters={filters} onNavigate={navigate} onDrilldown={onDrilldown} />; break
    case '/situation/regions': content = <CleanRegionsPage regions={data.regions} onDrilldown={onDrilldown} />; break
    case '/situation/topics': content = <CleanTopicsPage topics={data.topics} onDrilldown={onDrilldown} />; break
    case '/situation/time-series': content = <CleanTimeSeriesPage timeSeries={data.timeSeries} onDrilldown={onDrilldown} />; break
    case '/situation/alerts': content = <CleanAlertsPage alerts={data.alerts} onToast={onToast} onDrilldown={onDrilldown} />; break
    case '/situation/forecast': content = <CleanForecastPage forecast={data.forecast} status={data.forecastStatus} modelVersion={data.forecastModelVersion} />; break
    case '/situation/reports': content = <CleanReportsPage filters={filters} />; break
    case '/situation/learning': content = <CleanLearningPage learning={data.learning} onRefresh={onRefresh} onToast={onToast} />; break
    case '/situation/models': content = <CleanModelsPage models={data.models} />; break
  }
  return <>{content}{route !== '/operator' && <AnalyticsDrilldownPanel state={drilldown} loading={drilldownLoading} />}</>
}

function AnalyticsDrilldownPanel({ state, loading }: { state: DrilldownState; loading: boolean }) {
  if (loading) return <section className="panel drilldown-panel" aria-live="polite"><div className="panel-heading"><h2>Исходные обращения</h2></div><p className="panel-note">Загружаем обращения…</p></section>
  if (!state) return null
  return <section className="panel drilldown-panel" aria-live="polite"><div className="panel-heading"><h2>Исходные обращения</h2><span className="drilldown-label">{state.label} · {state.total}</span></div>{state.items.length ? <div className="drilldown-list">{state.items.map((ticket) => <article className="drilldown-ticket" key={ticket.id}><div><strong>{ticket.id}</strong><span>{ticket.region_name} · {ticket.topic_label}</span></div><p>{ticket.text}</p><small>{ticket.created_at} · {ticket.status} · {ticket.priority}</small></article>)}</div> : <p className="panel-note">За выбранный период обращений не найдено.</p>}</section>
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
            responseTemplate: 'Для исправленного решения шаблон недоступен. Составьте ответ вручную.',
            responseTemplateApproved: false,
            responseTemplateSource: undefined,
          } : {}),
        }
      })
      onDataChange(updated)
      onToast(result.source === 'demo' ? `Решение по ${ticketId} изменено только на этом экране` : ('warning' in result && result.warning) || `Решение по ${ticketId} сохранено`)
    } catch (error) {
      onToast(`Не удалось сохранить решение: ${error instanceof Error ? error.message : 'ошибка API'}`)
    }
  }

  const updateRelation = async (ticketId: string, relatedTicketId: string, relation: 'DUPLICATE' | 'REPEAT' | 'SIMILAR' | 'UNRELATED', decision: 'CONFIRMED' | 'REJECTED') => {
    try {
      await submitRelationFeedback(ticketId, relatedTicketId, relation, decision)
      onToast(`Связь ${relatedTicketId} сохранена: ${relation.toLowerCase()}`)
    } catch (error) {
      onToast(`Не удалось сохранить связь: ${error instanceof Error ? error.message : 'ошибка API'}`)
    }
  }

  const confirmationRate = overview.operatorDecisions ? Math.round((overview.confirmedDecisions / overview.operatorDecisions) * 100) : 0
  const decisionLatency = overview.avgDecisionMinutes && overview.avgDecisionMinutes > 0 ? `${Math.round(overview.avgDecisionMinutes)} мин` : 'нет решений'
  return <div className="operator-page">
    <div className="operator-summary"><div className="summary-item"><span className="summary-value">{filtered.length}</span><span className="summary-label">обращений в загруженной выборке</span><span className="summary-trend">по текущим фильтрам</span></div><div className="summary-item"><span className="summary-value">{confirmationRate}%</span><span className="summary-label">подтверждение без правок</span><span className="summary-trend">{overview.confirmedDecisions} из {overview.operatorDecisions}</span></div><div className="summary-item"><span className="summary-value">{decisionLatency}</span><span className="summary-label">среднее до решения</span><span className="summary-trend">по данным решений</span></div><div className="summary-item summary-signal"><span className="signal-wave"><i /><i /><i /><i /><i /></span><span><span className="summary-label">Система</span><span className="summary-sub">Рабочие данные доступны</span></span></div></div>
    <div className="workbench-grid">
      <section className="ticket-queue" aria-label="Очередь обращений">
        <div className="section-toolbar"><div><h2>Очередь на разбор <span className="count-pill">{filtered.length}</span></h2><p>Выберите обращение для проверки</p></div></div>
        <div className="filter-row"><div className="inline-search"><Icon name="search" size={16} /><input aria-label="Фильтр очереди" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Найти обращение" /></div><select aria-label="Фильтр по статусу" value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as typeof statusFilter)}><option value="all">Все статусы</option><option value="new">Новые</option><option value="reviewed">Разобранные</option></select><select aria-label="Фильтр по приоритету" value={priorityFilter} onChange={(event) => setPriorityFilter(event.target.value as typeof priorityFilter)}><option value="all">Все приоритеты</option><option value="Критический">Критический</option><option value="Высокий">Высокий</option><option value="Средний">Средний</option><option value="Низкий">Низкий</option></select></div>
        <div className="ticket-table-wrap"><table className="ticket-table"><thead><tr><th scope="col">Обращение</th><th scope="col">Тема</th><th scope="col">Уверенность</th><th scope="col">Регион</th><th scope="col">Приоритет</th><th scope="col"><span className="sr-only">Действия</span></th></tr></thead><tbody>{filtered.map((ticket) => <tr key={ticket.id} className={selected?.id === ticket.id ? 'selected-row' : ''} onClick={() => { setSelectedId(ticket.id); setMobileDetailOpen(true) }}><td><div className="ticket-id">{ticket.id}<span className={`channel-dot channel-${ticket.channel.replace(/[^a-zA-Z]/g, '').toLowerCase()}`} /></div><div className="ticket-preview">{ticket.originalText}</div><span className="ticket-time">{ticket.createdAt} · {languageLabel(ticket.language)}</span></td><td><span className="topic-cell">{ticket.topic}</span><span className="status-text">{ticket.status === 'new' ? 'Нужно решение' : ticket.status === 'confirmed' ? 'Подтверждено' : 'Исправлено'}</span></td><td><Confidence value={ticket.confidence} compact available={ticket.confidenceAvailable !== false} /></td><td><span className="region-cell">{ticket.region}</span></td><td><PriorityBadge priority={ticket.priority} /></td><td><button className="row-arrow icon-button" aria-label={`Открыть ${ticket.id}`} onClick={(event) => { event.stopPropagation(); setSelectedId(ticket.id); setMobileDetailOpen(true) }}><Icon name="arrow" size={17} /></button></td></tr>)}</tbody></table>{filtered.length === 0 && <div className="empty-state"><div className="state-icon">⌕</div><h3>Ничего не найдено</h3><p>Измените запрос или сбросьте фильтры.</p><button className="button button-quiet" onClick={() => { setQuery(''); setStatusFilter('all'); setPriorityFilter('all') }}>Сбросить фильтры</button></div>}</div>
      </section>
      {selected && <TicketDetail ticket={selected} taxonomy={taxonomy} open={mobileDetailOpen} onClose={() => setMobileDetailOpen(false)} onDecision={updateTicket} onRelationFeedback={updateRelation} onOpenRelated={(relatedId) => { setSelectedId(relatedId); setMobileDetailOpen(true) }} />}
    </div>
  </div>
}

function TicketDetail({ ticket, taxonomy, open, onClose, onDecision, onRelationFeedback, onOpenRelated }: { ticket: Ticket; taxonomy: DashboardData['filterOptions']; open: boolean; onClose: () => void; onDecision: (ticketId: string, decision: { status: 'confirmed' | 'corrected'; topic?: string; service?: string; priority?: string }) => Promise<void>; onRelationFeedback: (ticketId: string, relatedTicketId: string, relation: 'DUPLICATE' | 'REPEAT' | 'SIMILAR' | 'UNRELATED', decision: 'CONFIRMED' | 'REJECTED') => Promise<void>; onOpenRelated: (ticketId: string) => void }) {
  const [correctionOpen, setCorrectionOpen] = useState(false)
  const [templateOpen, setTemplateOpen] = useState(false)
  const [topic, setTopic] = useState(ticket.topic)
  const [service, setService] = useState(ticket.service)
  const [priority, setPriority] = useState<Priority>(ticket.priority)
  const confidenceState = normalizeConfidenceState(ticket.confidenceState, ticket.confidence, ticket.confidenceAvailable !== false)
  const confidenceAvailable = ticket.confidenceAvailable !== false && confidenceState !== 'UNAVAILABLE'
  const hasTemplate = Boolean(ticket.responseTemplateSource && ticket.responseTemplateSource !== 'UNAVAILABLE')
  const preview = ticket.assistPreview
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
    setTemplateOpen(false)
  }, [ticket.id, ticket.topic, ticket.service, ticket.priority, ticket.status, confidenceState])

  return <aside className={`ticket-detail ${open ? 'ticket-detail-open' : ''}`} aria-label={`Детали обращения ${ticket.id}`}>
    <div className="detail-header"><div><div className="detail-overline"><span className={`status-indicator ${ticket.status}`} />{ticket.status === 'new' ? 'Требует решения' : ticket.status === 'confirmed' ? 'Подтверждено' : 'Исправлено'}</div><h2>{ticket.id}</h2></div><button className="icon-button detail-close" aria-label="Закрыть детали" onClick={onClose}><Icon name="close" size={18} /></button></div>
    <div className="detail-scroll">
      <div className="original-text-block"><div className="field-label">Оригинальный текст <span className="language-chip">{languageLabel(ticket.language)}</span></div><p>«{ticket.originalText}»</p><div className="source-line">{ticket.channel} · {ticket.createdAt} · {ticket.region}</div></div>
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
        <div className="field-label">Маршрутизация и приоритет</div>
        <div className="routing-grid">
          <div className="routing-field">
            <span>Служба</span>
            <strong>{ticket.service}</strong>
            <RoutingProvenance
              provenance={ticket.serviceProvenance}
              fallbackReason="Демонстрационная рекомендация; официальный источник не предоставлен"
            />
          </div>
          <div className="routing-field">
            <span>Приоритет</span>
            <PriorityBadge priority={ticket.priority} />
            <RoutingProvenance
              provenance={ticket.priorityProvenance}
              fallbackReason="Демонстрационное значение; официальное правило не предоставлено"
            />
          </div>
        </div>
      </div>
      <div className="detail-section">
        <div className="field-label">Ответ оператору {ticket.status !== 'new' && <span className="language-chip">{ticket.responseTemplateApproved ? 'Утверждённый' : hasTemplate ? 'Демо-черновик' : 'Нет шаблона'}</span>}</div>
        {ticket.status === 'new' ? (
          <p className="panel-note">{ticket.responseTemplateSource === 'UNAVAILABLE' ? ticket.responseTemplate : 'Подтвердите или исправьте тему и службу, чтобы увидеть ответ.'}</p>
        ) : hasTemplate ? (
          <div className={`response-template ${templateOpen ? 'response-template-open' : ''}`}><p>{ticket.responseTemplate}</p><button className="text-button" onClick={() => setTemplateOpen((value) => !value)}><Icon name="external" size={14} />{templateOpen ? 'Свернуть шаблон' : 'Открыть шаблон'}</button></div>
        ) : (
          <p className="panel-note">{ticket.responseTemplate}</p>
        )}
      </div>
      <div className="detail-section"><div className="section-inline-heading"><div className="field-label">Похожие обращения <span className="count-pill">{ticket.similar.length}</span></div><span className="field-label">подтвердите связь</span></div><div className="similar-list">{ticket.similar.map((item) => <div className="similar-item" key={item.id}><button className="similar-main similar-open" onClick={() => onOpenRelated(item.id)}><strong>{item.id}</strong><span>{item.title}</span><Icon name="arrow" size={14} /></button><div className="similar-meta"><span className={`relation-badge ${item.relation === 'Дубликат' ? 'relation-duplicate' : item.relation === 'Повтор' ? 'relation-repeat' : ''}`}>{item.relation}</span><span>{formatPercent(item.similarity)}</span><button className="text-button" onClick={() => onRelationFeedback(ticket.id, item.id, item.relation === 'Дубликат' ? 'DUPLICATE' : item.relation === 'Повтор' ? 'REPEAT' : 'SIMILAR', 'CONFIRMED')}>Подтвердить</button><button className="text-button" onClick={() => onRelationFeedback(ticket.id, item.id, 'UNRELATED', 'REJECTED')}>Отклонить</button></div></div>)}</div></div>
    </div>
    {!correctionOpen && <div className="detail-actions"><button className="button button-primary" onClick={() => onDecision(ticket.id, { status: 'confirmed' })}><Icon name="check" size={16} />Подтвердить</button><button className="button button-secondary" onClick={() => setCorrectionOpen(true)}><Icon name="edit" size={16} />Исправить</button></div>}
  </aside>
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
    </span>
  )
}

function CleanRegionsPage({ regions, onDrilldown }: { regions: RegionMetric[]; onDrilldown: DrilldownHandler }) {
  if (!regions.length) return <div className="analytics-page"><NoData message="Нет данных по регионам." /></div>
  return <div className="analytics-page"><section className="panel"><PanelHeading title="Нагрузка по регионам" /><div className="region-table region-table-full"><div className="region-table-head"><span>Регион</span><span>Обращения</span><span>Динамика</span><span>Состояние</span></div>{regions.map((region) => <button className="region-table-row region-table-row-full drilldown-row" key={region.name} onClick={() => onDrilldown('region', region.id, `Регион: ${region.name}`)}><strong>{region.name}</strong><span>{region.tickets.toLocaleString('ru-RU')}</span><span>{region.change == null ? '—' : String(region.change) + '%'}</span><span>{region.risk ?? 'нет данных'}</span></button>)}</div></section></div>
}

function CleanTopicsPage({ topics, onDrilldown }: { topics: TopicMetric[]; onDrilldown: DrilldownHandler }) {
  if (!topics.length) return <div className="analytics-page"><NoData message="Нет данных по темам." /></div>
  return <div className="analytics-page"><section className="panel"><PanelHeading title="Распределение по темам" /><div className="topic-bars">{topics.map((topic) => <button className="topic-bar-row drilldown-row" key={topic.name} onClick={() => onDrilldown('topic', topic.id, `Тема: ${topic.name}`)}><div className="topic-bar-label"><span>{topic.name}</span><strong>{topic.value}%</strong></div><div className="bar-track"><span style={{ width: String(topic.value) + '%', background: topic.color }} /></div></button>)}</div></section></div>
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

function CleanAlertsPage({ alerts, onToast, onDrilldown }: { alerts: Alert[]; onToast: (message: string) => void; onDrilldown: DrilldownHandler }) {
  const [items, setItems] = useState(alerts)
  useEffect(() => setItems(alerts), [alerts])
  if (!items.length) return <div className="analytics-page"><NoData message="Нет подключённых оповещений." /></div>
  const update = async (alert: Alert, action: 'ack' | 'close') => {
    try {
      const updated = action === 'ack' ? await acknowledgeAlert(alert.id) : await closeAlert(alert.id)
      const status = updated.status.toLowerCase() === 'acknowledged' ? 'В работе' : updated.status.toLowerCase() === 'closed' ? 'Закрыт' : alert.status
      setItems((current) => current.map((item) => item.id === alert.id ? { ...item, status } : item))
      onToast(`Оповещение ${alert.id}: состояние подтверждено backend`)
    } catch (error) {
      onToast(`Не удалось обновить оповещение: ${error instanceof Error ? error.message : 'ошибка API'}`)
    }
  }
  return <div className="analytics-page"><section className="panel"><PanelHeading title="Оповещения" /><div className="alerts-table">{items.map((alert) => <div className="alert-row" key={alert.id}><button className="alert-drilldown" onClick={() => onDrilldown('alert', alert.id, `Оповещение: ${alert.title}`)}><span className={"alert-dot alert-" + alert.severity} /><span className="alert-row-main"><strong>{alert.title}</strong><span>{alert.description}</span><small>{alert.detectedAt} · {alert.region}</small></span><span className="alert-count">{alert.affectedTickets}</span></button><span className="alert-actions">{alert.status === 'Новый' && <button className="text-button" onClick={() => update(alert, 'ack')}>Принять</button>}{alert.status !== 'Закрыт' && <button className="text-button" onClick={() => update(alert, 'close')}>Закрыть</button>}</span></div>)}</div></section></div>
}

function CleanForecastPage({ forecast, status, modelVersion }: { forecast: ForecastPoint[]; status?: string; modelVersion?: string }) {
  if (!forecast.length) return <div className="analytics-page"><NoData message="Для прогноза пока недостаточно истории обращений." /></div>
  const option: EChartsOption = {
    ...chartBaseOption(forecast.map((point) => point.label)),
    series: [{ name: 'Прогноз обращений', type: 'line' as const, data: forecast.map((point) => point.forecast ?? null), showSymbol: false, lineStyle: { width: 2 }, areaStyle: { color: 'rgba(167, 217, 255, .12)' }, itemStyle: { color: '#a7d9ff' } }],
  }
  return <div className="analytics-page"><section className="panel"><PanelHeading title="Прогноз нагрузки" /><p className="panel-note">Расчёт по истории обращений · состояние {status ?? 'unknown'} · версия {modelVersion ?? 'не указана'}</p><DataChart option={option} label="Прогноз количества обращений по дням" /><details className="chart-values"><summary>Показать значения по дням</summary><table><thead><tr><th>Дата</th><th>Обращения</th></tr></thead><tbody>{forecast.map((point) => <tr key={point.label}><td>{point.label}</td><td>{point.forecast ?? '—'}</td></tr>)}</tbody></table></details></section></div>
}

function CleanReportsPage({ filters }: { filters: DashboardFilters }) {
  const filterSummary = [filters.range, filters.regionId ?? 'все регионы', filters.topicId ?? 'все темы', filters.serviceId ?? 'все службы', filters.status ?? 'все статусы', filters.district ?? 'все районы', filters.channel ?? 'все каналы'].join(', ')
  return <div className="analytics-page"><section className="panel"><PanelHeading title="Отчёты" /><p className="panel-note">В выгрузку войдут данные за {filterSummary}.</p><div className="report-actions"><a className="button button-primary" href={reportUrl('pdf', filters)}>Скачать PDF</a><a className="button button-secondary" href={reportUrl('xlsx', filters)}>Скачать XLSX</a></div></section></div>
}

function CleanLearningPage({ learning, onRefresh, onToast }: { learning: LearningCycle; onRefresh: () => Promise<void>; onToast: (message: string) => void }) {
  const [evaluation, setEvaluation] = useState<Awaited<ReturnType<typeof loadCandidateEvaluation>> | null>(null)
  const [evaluationError, setEvaluationError] = useState<string | null>(null)
  const [busy, setBusy] = useState<'close' | 'evaluation' | 'promote' | 'reject' | null>(null)
  const [note, setNote] = useState('')

  useEffect(() => {
    let active = true
    setEvaluation(null)
    setEvaluationError(null)
    if (!['EVALUATE', 'DECISION'].includes(learning.stage) || learning.id === 'нет данных') return () => { active = false }
    void loadCandidateEvaluation().then((result) => {
      if (active) setEvaluation(result)
    }).catch((error: unknown) => {
      if (active) setEvaluationError(error instanceof Error ? error.message : 'Оценка пока недоступна')
    })
    return () => { active = false }
  }, [learning.id, learning.stage, learning.updatedAt])

  if (learning.id === 'нет данных') return <div className="analytics-page"><NoData message="Цикл обучения не предоставлен API." /></div>

  const runAction = async (action: 'close' | 'evaluation' | 'promote' | 'reject') => {
    setBusy(action)
    try {
      if (action === 'close') {
        await closeLearningCycle(learning.id)
        onToast('Сбор обратной связи закрыт; background job поставлена в очередь')
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

  const offlineStatus = evaluation?.offline_metrics?.status
  const readyToReview = evaluation?.decision === 'READY_TO_REVIEW'
  return <div className="analytics-page">
    <section className="panel">
      <PanelHeading title={'Цикл ' + learning.id} />
      <div className="dataset-stat"><span>Состояние</span><strong>{learning.stage}</strong></div>
      <div className="dataset-stat"><span>Обратная связь</span><strong>{learning.feedbackCount}</strong></div>
      <div className="dataset-stat"><span>Датасет</span><strong>{learning.dataset}</strong></div>
      <div className="dataset-stat"><span>Кандидат</span><strong>{learning.candidate}</strong></div>
      {learning.decisionNote && <p className="panel-note learning-decision-note">Решение: {learning.decisionNote}</p>}
      <div className="learning-actions" aria-label="Действия reviewer">
        {learning.stage === 'COLLECT' && <button className="button button-primary" disabled={busy !== null} onClick={() => void runAction('close')}>{busy === 'close' ? 'Закрываем…' : 'Закрыть цикл'}</button>}
        {['EVALUATE', 'DECISION'].includes(learning.stage) && <button className="button button-secondary" disabled={busy !== null} onClick={() => void runAction('evaluation')}>{busy === 'evaluation' ? 'Читаем…' : 'Показать evaluation'}</button>}
        {['EVALUATE', 'DECISION'].includes(learning.stage) && <>
          <input className="learning-note" aria-label="Комментарий reviewer" placeholder="Комментарий к решению (необязательно)" value={note} onChange={(event) => setNote(event.target.value)} disabled={busy !== null} />
          <button className="button button-primary" disabled={busy !== null || !readyToReview} onClick={() => void runAction('promote')}>{busy === 'promote' ? 'Продвигаем…' : 'Promote'}</button>
          <button className="button button-quiet" disabled={busy !== null} onClick={() => void runAction('reject')}>{busy === 'reject' ? 'Отклоняем…' : 'Reject'}</button>
        </>}
      </div>
    </section>
    {['EVALUATE', 'DECISION'].includes(learning.stage) && <section className="panel learning-evaluation">
      <PanelHeading title="Evaluation candidate" />
      {evaluationError && <p className="panel-note">Оценка пока недоступна: {evaluationError}. После завершения background job нажмите «Показать evaluation».</p>}
      {!evaluation && !evaluationError && <p className="panel-note">Загружаем evaluation из backend…</p>}
      {evaluation && <>
        <div className="dataset-stat"><span>Решение policy</span><strong>{evaluation.decision}</strong></div>
        <div className="dataset-stat"><span>Статус evidence</span><strong>{String(offlineStatus ?? 'нет данных')}</strong></div>
        <div className="dataset-stat"><span>Sample size</span><strong>{evaluation.sample_size}</strong></div>
        <div className="dataset-stat"><span>Promotion policy</span><strong>{evaluation.promotion_policy_version}</strong></div>
        <pre className="learning-evaluation-json">{JSON.stringify({ offline_metrics: evaluation.offline_metrics, shadow_metrics: evaluation.shadow_metrics, critical_regressions: evaluation.critical_regressions }, null, 2)}</pre>
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
  const queryRows = queryResult?.rows ?? queryResult?.result?.points ?? []
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
      <MetricCard label="Обращений в выборке" value={String(total)} change="текущий срез" detail="по выбранным фильтрам" tone="mint" icon="inbox" />
      <MetricCard label="Высокий приоритет" value={String(highPriority)} change="текущий срез" detail="по выбранным фильтрам" tone="rose" icon="pulse" />
      <MetricCard label="Решения оператора" value={String(data.overview.operatorDecisions)} change={`${data.overview.confirmedDecisions} подтверждено · ${data.overview.correctedDecisions} исправлено`} detail="по данным решений" tone="amber" icon="clock" />
      <MetricCard label="Аномальные сигналы" value={String(data.alerts.length)} change="обнаружено системой" detail="текущий срез" tone="blue" icon="bell" />
    </div>
    <div className="analytics-grid overview-grid">
      <section className="panel span-two"><PanelHeading title="Поток обращений" action="Временная динамика" onClick={() => onNavigate('/situation/time-series')} />{data.timeSeries.length ? <div className="region-table"><div className="region-table-head"><span>Дата</span><span>Обращения</span><span>Закрыто</span><span>Доля</span></div>{data.timeSeries.slice(-7).map((point) => <button className="region-table-row drilldown-row" key={point.date} onClick={() => onDrilldown('date', point.date, `Дата: ${point.date}`)}><strong>{point.date}</strong><span>{point.tickets}</span><span>{point.resolved}</span><span>{point.tickets ? `${Math.round(point.resolved / point.tickets * 100)}%` : '—'}</span></button>)}</div> : <NoData message="За выбранный период нет обращений." />}</section>
      <section className="panel"><PanelHeading title="Темы" action="Все темы" onClick={() => onNavigate('/situation/topics')} />{data.topics.length ? <div className="topic-bars">{data.topics.slice(0, 5).map((topic) => <button className="topic-bar-row drilldown-row" key={topic.name} onClick={() => onDrilldown('topic', topic.id, `Тема: ${topic.name}`)}><div className="topic-bar-label"><span>{topic.name}</span><strong>{topic.value}%</strong></div><div className="bar-track"><span style={{ width: String(topic.value * 2.7) + '%', background: topic.color }} /></div></button>)}</div> : <NoData message="Нет данных по темам." />}</section>
      <section className="panel"><PanelHeading title="Сигналы" action="Открыть все" onClick={() => onNavigate('/situation/alerts')} />{data.alerts.length ? <div className="alert-list">{data.alerts.map((alert) => <AlertListItem alert={alert} key={alert.id} />)}</div> : <NoData message="Нет подключённых сигналов." />}</section>
      <section className="panel span-two region-panel"><PanelHeading title="Регионы" action="Все регионы" onClick={() => onNavigate('/situation/regions')} />{data.regions.length ? <div className="region-table"><div className="region-table-head"><span>Регион</span><span>Обращения</span><span>Динамика</span><span>Состояние</span></div>{data.regions.map((region) => <RegionRow region={region} key={region.name} onDrilldown={onDrilldown} />)}</div> : <NoData message="Нет данных по регионам." />}</section>
      <section className="panel query-panel">
        <div className="eyebrow"><span className="eyebrow-line" />Спросить данные</div>
        <h3>Ответ по обращениям</h3>
        <p>Можно спросить о количестве, динамике, регионах, темах, всплесках или прогнозе.</p>
        <div className="query-input"><Icon name="search" size={16} /><input aria-label="Вопрос по данным" placeholder="Сколько обращений по регионам?" value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') void submitQuery() }} /><button aria-label="Выполнить поиск" onClick={() => void submitQuery()} disabled={queryLoading}><Icon name="arrow" size={16} /></button></div>
        {queryError && <p className="query-error" role="alert">Не удалось получить ответ: {queryError}</p>}
        {queryResult && <div className="query-results" role="status">
          <strong>{queryTitle}</strong>
          {queryResult.intent === 'count' ? <p>Обращений за выбранный период: {queryResult.number ?? 0}</p> : queryRows.length ? <ul>{queryRows.slice(0, 10).map((row, index) => <li key={`${row.label ?? row.period ?? row.date ?? 'row'}-${index}`}><span>{row.label ?? row.period ?? row.date ?? `Позиция ${index + 1}`}</span><strong>{row.count ?? row.tickets ?? 0}</strong></li>)}</ul> : <p>По выбранным фильтрам результатов нет.</p>}
        </div>}
      </section>
    </div>
  </div>
}

function MetricCard({ label, value, change, detail, tone, icon }: { label: string; value: string; change: string; detail: string; tone: string; icon: IconName }) {
  return <article className={`metric-card metric-${tone}`}><div className="metric-top"><span>{label}</span><span className="metric-icon"><Icon name={icon} size={17} /></span></div><div className="metric-value">{value}</div><div className="metric-bottom"><span className="metric-change">{change}</span><span>{detail}</span></div></article>
}

function NoData({ message }: { message: string }) {
  return <div className="state-card"><div className="state-icon">—</div><p>{message}</p></div>
}

function PanelHeading({ title, action, onClick }: { title: string; action?: string; onClick?: () => void }) {
  return <div className="panel-heading"><h2>{title}</h2>{action && <button className="text-button" onClick={onClick}>{action}<Icon name="arrow" size={14} /></button>}</div>
}

function AlertListItem({ alert }: { alert: Alert }) {
  return <div className="alert-list-item"><span className={`alert-dot alert-${alert.severity}`} /><div><strong>{alert.title}</strong><span>{alert.region} · {alert.affectedTickets} обращений</span></div><Icon name="arrow" size={14} /></div>
}

function RegionRow({ region, onDrilldown }: { region: RegionMetric; onDrilldown?: DrilldownHandler }) {
  const change = region.change == null ? '—' : String(region.change) + '%'
  const risk = region.risk ?? 'нет данных'
  if (!onDrilldown) return <div className="region-table-row"><strong>{region.name}</strong><span>{region.tickets.toLocaleString('ru-RU')}</span><span>{change}</span><span>{risk}</span></div>
  return <button className="region-table-row drilldown-row" onClick={() => onDrilldown('region', region.id, `Регион: ${region.name}`)}><strong>{region.name}</strong><span>{region.tickets.toLocaleString('ru-RU')}</span><span>{change}</span><span>{risk}</span></button>
}











export default App
