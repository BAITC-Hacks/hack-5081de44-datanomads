import { useCallback, useEffect, useMemo, useState } from 'react'
import { loadDashboard, submitDecision } from './api/client'
import type { Alert, ApiSource, DashboardData, ForecastPoint, LearningCycle, ModelStatus, Priority, RegionMetric, Ticket, TopicMetric } from './types'

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

const operatorNav = [{ label: 'Входящие', route: '/operator' as Route, icon: 'inbox' as IconName, badge: '24' }]
const situationNav = [
  { label: 'Обзор', route: '/situation/overview' as Route, icon: 'grid' as IconName },
  { label: 'Регионы', route: '/situation/regions' as Route, icon: 'map' as IconName },
  { label: 'Темы', route: '/situation/topics' as Route, icon: 'tag' as IconName },
  { label: 'Временная динамика', route: '/situation/time-series' as Route, icon: 'trend' as IconName },
  { label: 'Оповещения', route: '/situation/alerts' as Route, icon: 'bell' as IconName, badge: '3' },
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

function confidenceLabel(value: number) {
  if (value >= 0.9) return 'Высокая'
  if (value >= 0.75) return 'Средняя'
  return 'Низкая'
}

function App() {
  const [route, setRoute] = useState<Route>(routeFromHash)
  const [data, setData] = useState<DashboardData | null>(null)
  const [source, setSource] = useState<ApiSource>('demo')
  const [apiError, setApiError] = useState<string | undefined>()
  const [loading, setLoading] = useState(true)
  const [toast, setToast] = useState<string | null>(null)
  const [mobileNavOpen, setMobileNavOpen] = useState(false)

  useEffect(() => {
    const onHashChange = () => setRoute(routeFromHash())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  useEffect(() => {
    let active = true
    setLoading(true)
    loadDashboard().then((result) => {
      if (!active) return
      setData(result.data)
      setSource(result.source)
      setApiError(result.error)
      setLoading(false)
    })
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (!toast) return
    const timer = window.setTimeout(() => setToast(null), 4000)
    return () => window.clearTimeout(timer)
  }, [toast])

  const title = routeTitles[route]
  const showToast = useCallback((message: string) => setToast(message), [])

  return (
    <div className="app-shell">
      <Sidebar route={route} mobileOpen={mobileNavOpen} onNavigate={(next) => { navigate(next); setMobileNavOpen(false) }} />
      <main className="main-shell">
        <Topbar onOpenNav={() => setMobileNavOpen(true)} />
        {source === 'demo' && <div className="demo-banner"><span className="status-dot" /> Демо-данные · API подключится автоматически, когда backend будет доступен <span className="demo-banner-detail">{apiError ? `(${apiError})` : ''}</span></div>}
        <div className="page-wrap">
          <PageHeader {...title} route={route} onNavigate={navigate} />
          {loading ? <LoadingState /> : data ? <RouteContent route={route} data={data} onDataChange={setData} onToast={showToast} /> : <ErrorState onRetry={() => window.location.reload()} />}
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
      <div className="sidebar-bottom">
        <button className="sidebar-link"><Icon name="help" size={17} /><span>Помощь и правила</span></button>
        <button className="sidebar-link"><Icon name="settings" size={17} /><span>Настройки</span></button>
        <div className="user-card"><div className="avatar">АМ</div><div className="user-meta"><strong>Алия М.</strong><span>Оператор · Алматы</span></div><Icon name="more" size={18} /></div>
      </div>
    </aside>
  </>
}

function NavItem({ item, active, onNavigate }: { item: { label: string; route: Route; icon: IconName; badge?: string }; active: boolean; onNavigate: (route: Route) => void }) {
  return <button className={`nav-item ${active ? 'nav-item-active' : ''}`} onClick={() => onNavigate(item.route)} aria-current={active ? 'page' : undefined}><Icon name={item.icon} size={17} /><span>{item.label}</span>{item.badge && <span className="nav-badge">{item.badge}</span>}</button>
}

function Topbar({ onOpenNav }: { onOpenNav: () => void }) {
  return <header className="topbar">
    <button className="mobile-menu-button icon-button" aria-label="Открыть меню" onClick={onOpenNav}><span className="menu-lines" /></button>
    <div className="topbar-search"><Icon name="search" size={17} /><input aria-label="Поиск по обращениям" placeholder="Поиск по ID, тексту или региону" /><kbd>⌘ K</kbd></div>
    <div className="topbar-actions"><button className="icon-button" aria-label="Помощь"><Icon name="help" size={18} /></button><button className="notification-button" aria-label="3 новых оповещения"><Icon name="bell" size={18} /><span /></button><div className="topbar-date"><span className="live-dot" />21 сентября 2026</div></div>
  </header>
}

function PageHeader({ eyebrow, title, description, route, onNavigate }: { eyebrow: string; title: string; description: string; route: Route; onNavigate: (route: Route) => void }) {
  return <div className="page-header"><div><div className="eyebrow"><span className="eyebrow-line" />{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{route === '/operator' ? <button className="button button-quiet"><Icon name="download" size={16} />Экспорт очереди</button> : <div className="period-control"><button className="period-button active">7 дней</button><button className="period-button">30 дней</button><button className="period-button">90 дней</button><button className="period-button icon-button" aria-label="Другой период"><Icon name="clock" size={15} /></button></div>}</div>
}

function RouteContent({ route, data, onDataChange, onToast }: { route: Route; data: DashboardData; onDataChange: (data: DashboardData) => void; onToast: (message: string) => void }) {
  switch (route) {
    case '/operator': return <OperatorPage tickets={data.tickets} onDataChange={(tickets) => onDataChange({ ...data, tickets })} onToast={onToast} />
    case '/situation/overview': return <OverviewPage data={data} onNavigate={navigate} />
    case '/situation/regions': return <RegionsPage regions={data.regions} />
    case '/situation/topics': return <TopicsPage topics={data.topics} />
    case '/situation/time-series': return <TimeSeriesPage />
    case '/situation/alerts': return <AlertsPage alerts={data.alerts} onToast={onToast} />
    case '/situation/forecast': return <ForecastPage forecast={data.forecast} />
    case '/situation/reports': return <ReportsPage onToast={onToast} />
    case '/situation/learning': return <LearningPage learning={data.learning} />
    case '/situation/models': return <ModelsPage models={data.models} />
  }
}

function LoadingState() {
  return <div className="loading-grid" aria-label="Загрузка данных"><div className="skeleton skeleton-wide" /><div className="skeleton-row"><div className="skeleton" /><div className="skeleton" /><div className="skeleton" /></div><div className="skeleton skeleton-large" /></div>
}

function ErrorState({ onRetry }: { onRetry: () => void }) {
  return <div className="state-card"><div className="state-icon error-icon">!</div><h2>Не удалось загрузить рабочие данные</h2><p>Проверьте подключение к API и повторите попытку.</p><button className="button button-primary" onClick={onRetry}>Повторить</button></div>
}

function OperatorPage({ tickets, onDataChange, onToast }: { tickets: Ticket[]; onDataChange: (tickets: Ticket[]) => void; onToast: (message: string) => void }) {
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
    const result = await submitDecision(ticketId, decision)
    const updated = tickets.map((item) => item.id === ticketId ? { ...item, ...decision, priority: (decision.priority as Priority | undefined) ?? item.priority } : item)
    onDataChange(updated)
    onToast(result.source === 'demo' ? `Решение по ${ticketId} сохранено в демо-режиме` : `Решение по ${ticketId} сохранено`)
  }

  return <div className="operator-page">
    <div className="operator-summary"><div className="summary-item"><span className="summary-value">24</span><span className="summary-label">новых сегодня</span><span className="summary-trend positive">+8,4%</span></div><div className="summary-item"><span className="summary-value">89%</span><span className="summary-label">подтверждено без правок</span><span className="summary-trend positive">+2,1 п.п.</span></div><div className="summary-item"><span className="summary-value">14 мин</span><span className="summary-label">медиана до решения</span><span className="summary-trend negative">+3 мин</span></div><div className="summary-item summary-signal"><span className="signal-wave"><i /><i /><i /><i /><i /></span><span><span className="summary-label">система в норме</span><span className="summary-sub">Последнее обновление 1 мин назад</span></span></div></div>
    <div className="workbench-grid">
      <section className="ticket-queue" aria-label="Очередь обращений">
        <div className="section-toolbar"><div><h2>Очередь на разбор <span className="count-pill">{filtered.length}</span></h2><p>Сначала — обращения с высоким влиянием</p></div><button className="icon-button" aria-label="Настроить очередь"><Icon name="settings" size={17} /></button></div>
        <div className="filter-row"><div className="inline-search"><Icon name="search" size={16} /><input aria-label="Фильтр очереди" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Найти обращение" /></div><select aria-label="Фильтр по статусу" value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as typeof statusFilter)}><option value="all">Все статусы</option><option value="new">Новые</option><option value="reviewed">Разобранные</option></select><select aria-label="Фильтр по приоритету" value={priorityFilter} onChange={(event) => setPriorityFilter(event.target.value as typeof priorityFilter)}><option value="all">Все приоритеты</option><option value="Высокий">Высокий</option><option value="Средний">Средний</option><option value="Низкий">Низкий</option></select></div>
        <div className="ticket-table-wrap"><table className="ticket-table"><thead><tr><th scope="col">Обращение</th><th scope="col">Тема</th><th scope="col">Уверенность</th><th scope="col">Регион</th><th scope="col">Приоритет</th><th scope="col"><span className="sr-only">Действия</span></th></tr></thead><tbody>{filtered.map((ticket) => <tr key={ticket.id} className={selected?.id === ticket.id ? 'selected-row' : ''} onClick={() => { setSelectedId(ticket.id); setMobileDetailOpen(true) }}><td><div className="ticket-id">{ticket.id}<span className={`channel-dot channel-${ticket.channel.replace(/[^a-zA-Z]/g, '').toLowerCase()}`} /></div><div className="ticket-preview">{ticket.originalText}</div><span className="ticket-time">{ticket.createdAt} · {ticket.language}</span></td><td><span className="topic-cell">{ticket.topic}</span><span className="status-text">{ticket.status === 'new' ? 'Нужно решение' : ticket.status === 'confirmed' ? 'Подтверждено' : 'Исправлено'}</span></td><td><Confidence value={ticket.confidence} compact /></td><td><span className="region-cell">{ticket.region}</span></td><td><PriorityBadge priority={ticket.priority} /></td><td><button className="row-arrow icon-button" aria-label={`Открыть ${ticket.id}`} onClick={(event) => { event.stopPropagation(); setSelectedId(ticket.id); setMobileDetailOpen(true) }}><Icon name="arrow" size={17} /></button></td></tr>)}</tbody></table>{filtered.length === 0 && <div className="empty-state"><div className="state-icon">⌕</div><h3>Ничего не найдено</h3><p>Измените запрос или сбросьте фильтры.</p><button className="button button-quiet" onClick={() => { setQuery(''); setStatusFilter('all'); setPriorityFilter('all') }}>Сбросить фильтры</button></div>}</div>
      </section>
      {selected && <TicketDetail ticket={selected} open={mobileDetailOpen} onClose={() => setMobileDetailOpen(false)} onDecision={updateTicket} />}
    </div>
  </div>
}

function TicketDetail({ ticket, open, onClose, onDecision }: { ticket: Ticket; open: boolean; onClose: () => void; onDecision: (ticketId: string, decision: { status: 'confirmed' | 'corrected'; topic?: string; service?: string; priority?: string }) => Promise<void> }) {
  const [correctionOpen, setCorrectionOpen] = useState(false)
  const [topic, setTopic] = useState(ticket.topic)
  const [service, setService] = useState(ticket.service)
  const [priority, setPriority] = useState<Priority>(ticket.priority)
  useEffect(() => { setTopic(ticket.topic); setService(ticket.service); setPriority(ticket.priority); setCorrectionOpen(false) }, [ticket.id, ticket.topic, ticket.service, ticket.priority])

  return <aside className={`ticket-detail ${open ? 'ticket-detail-open' : ''}`} aria-label={`Детали обращения ${ticket.id}`}>
    <div className="detail-header"><div><div className="detail-overline"><span className={`status-indicator ${ticket.status}`} />{ticket.status === 'new' ? 'Требует решения' : ticket.status === 'confirmed' ? 'Подтверждено' : 'Исправлено'}</div><h2>{ticket.id}</h2></div><button className="icon-button detail-close" aria-label="Закрыть детали" onClick={onClose}><Icon name="close" size={18} /></button></div>
    <div className="detail-scroll">
      <div className="original-text-block"><div className="field-label">Оригинальный текст <span className="language-chip">{ticket.language}</span></div><p>«{ticket.originalText}»</p><div className="source-line">{ticket.channel} · {ticket.createdAt} · {ticket.region}</div></div>
      <div className="detail-section"><div className="field-label">Модель предложила</div><div className="prediction-row"><div><div className="prediction-topic">{ticket.topic}</div><div className="confidence-copy">{confidenceLabel(ticket.confidence)} уверенность · модель cls-2026-09-18.4</div></div><Confidence value={ticket.confidence} /></div><div className="alternatives"><span className="field-label">Альтернативы</span>{ticket.alternatives.map((alternative) => <div className="alternative-row" key={alternative.topic}><span>{alternative.topic}</span><span>{formatPercent(alternative.confidence)}</span></div>)}</div></div>
      <div className="detail-section"><div className="field-label">Маршрутизация</div><div className="routing-grid"><div className="routing-field"><span>Рекомендуемая служба</span><strong>{ticket.service}</strong></div><div className="routing-field"><span>Приоритет</span><PriorityBadge priority={ticket.priority} /></div></div></div>
      <div className="detail-section"><div className="field-label">Рекомендуемый ответ</div><div className="response-template"><p>{ticket.responseTemplate}</p><button className="text-button"><Icon name="external" size={14} />Открыть шаблон</button></div></div>
      <div className="detail-section"><div className="section-inline-heading"><div className="field-label">Похожие обращения <span className="count-pill">{ticket.similar.length}</span></div><button className="text-button">Все похожие <Icon name="arrow" size={14} /></button></div><div className="similar-list">{ticket.similar.map((item) => <div className="similar-item" key={item.id}><div className="similar-main"><strong>{item.id}</strong><span>{item.title}</span></div><div className="similar-meta"><span className={`relation-badge ${item.relation === 'Дубликат' ? 'relation-duplicate' : item.relation === 'Повтор' ? 'relation-repeat' : ''}`}>{item.relation}</span><span>{formatPercent(item.similarity)}</span></div></div>)}</div></div>
      {correctionOpen && <div className="correction-panel"><div className="correction-heading"><strong>Исправить решение</strong><button className="icon-button" aria-label="Закрыть форму исправления" onClick={() => setCorrectionOpen(false)}><Icon name="close" size={15} /></button></div><label>Тема<select value={topic} onChange={(event) => setTopic(event.target.value)}><option>{ticket.topic}</option>{ticket.alternatives.map((alternative) => <option key={alternative.topic}>{alternative.topic}</option>)}<option>Другая тема</option></select></label><label>Служба<input value={service} onChange={(event) => setService(event.target.value)} /></label><label>Приоритет<select value={priority} onChange={(event) => setPriority(event.target.value as Priority)}><option>Высокий</option><option>Средний</option><option>Низкий</option></select></label><button className="button button-primary full-width" onClick={() => onDecision(ticket.id, { status: 'corrected', topic, service, priority })}><Icon name="check" size={16} />Сохранить исправление</button></div>}
    </div>
    {!correctionOpen && <div className="detail-actions"><button className="button button-primary" onClick={() => onDecision(ticket.id, { status: 'confirmed' })}><Icon name="check" size={16} />Подтвердить</button><button className="button button-secondary" onClick={() => setCorrectionOpen(true)}><Icon name="edit" size={16} />Исправить</button></div>}
  </aside>
}

function Confidence({ value, compact = false }: { value: number; compact?: boolean }) {
  return <div className={`confidence ${compact ? 'confidence-compact' : ''}`}><div className="confidence-track"><span style={{ width: `${value * 100}%` }} /></div><strong>{formatPercent(value)}</strong>{!compact && <small>{confidenceLabel(value)}</small>}</div>
}

function PriorityBadge({ priority }: { priority: Priority }) {
  return <span className={`priority-badge priority-${priority === 'Высокий' ? 'high' : priority === 'Средний' ? 'medium' : 'low'}`}><span />{priority}</span>
}

function OverviewPage({ data, onNavigate }: { data: DashboardData; onNavigate: (route: Route) => void }) {
  return <div className="analytics-page"><div className="metrics-grid"><MetricCard label="Всего обращений" value="6 476" change="+6,8%" detail="к прошлой неделе" tone="mint" icon="inbox" /><MetricCard label="В работе" value="1 184" change="−4,2%" detail="очередь сокращается" tone="blue" icon="pulse" /><MetricCard label="SLA первого ответа" value="92,6%" change="+1,9 п.п." detail="цель — 90%" tone="amber" icon="clock" /><MetricCard label="Аномальные сигналы" value="3" change="1 новый" detail="требуют внимания" tone="rose" icon="bell" /></div><div className="analytics-grid overview-grid"><section className="panel span-two"><PanelHeading title="Поток обращений" action="Временная динамика" onClick={() => onNavigate('/situation/time-series')} /><div className="chart-headline"><div><strong>6 476</strong><span>обращений за 7 дней</span></div><div className="chart-legend"><span><i className="legend-dot mint-dot" />факт</span><span><i className="legend-dot muted-dot" />предыдущая неделя</span></div></div><Sparkline large /></section><section className="panel"><PanelHeading title="Темы" action="Все темы" onClick={() => onNavigate('/situation/topics')} /><div className="topic-bars">{data.topics.slice(0, 5).map((topic) => <div className="topic-bar-row" key={topic.name}><div className="topic-bar-label"><span>{topic.name}</span><strong>{topic.value}%</strong></div><div className="bar-track"><span style={{ width: `${topic.value * 2.7}%`, background: topic.color }} /></div></div>)}</div></section><section className="panel"><PanelHeading title="Сигналы" action="Открыть все" onClick={() => onNavigate('/situation/alerts')} /><div className="alert-list">{data.alerts.map((alert) => <AlertListItem alert={alert} key={alert.id} />)}</div></section><section className="panel span-two region-panel"><PanelHeading title="Регионы" action="К карте регионов" onClick={() => onNavigate('/situation/regions')} /><div className="region-table"><div className="region-table-head"><span>Регион</span><span>Обращения</span><span>Динамика</span><span>Состояние</span></div>{data.regions.map((region) => <RegionRow region={region} key={region.name} />)}</div></section><section className="panel query-panel"><div className="eyebrow"><span className="eyebrow-line" />Спросить данные</div><h3>Найдите ответ в потоке</h3><p>Например: «Где больше всего повторных обращений?»</p><div className="query-input"><Icon name="search" size={16} /><input aria-label="Вопрос по данным" placeholder="Задайте вопрос на русском" /><button aria-label="Выполнить поиск"><Icon name="arrow" size={16} /></button></div><span className="query-note">Ответы строятся по проверенным срезам данных</span></section></div></div>
}

function MetricCard({ label, value, change, detail, tone, icon }: { label: string; value: string; change: string; detail: string; tone: string; icon: IconName }) {
  return <article className={`metric-card metric-${tone}`}><div className="metric-top"><span>{label}</span><span className="metric-icon"><Icon name={icon} size={17} /></span></div><div className="metric-value">{value}</div><div className="metric-bottom"><span className="metric-change">{change}</span><span>{detail}</span></div></article>
}

function PanelHeading({ title, action, onClick }: { title: string; action?: string; onClick?: () => void }) {
  return <div className="panel-heading"><h2>{title}</h2>{action && <button className="text-button" onClick={onClick}>{action}<Icon name="arrow" size={14} /></button>}</div>
}

function Sparkline({ large = false }: { large?: boolean }) {
  const points = large ? '0,99 36,92 72,95 108,73 144,80 180,66 216,72 252,45 288,56 324,38 360,44 396,20 432,31 468,12 504,25 540,7' : '0,35 22,29 44,32 66,20 88,25 110,12 132,17 154,4'
  return <svg className={`sparkline ${large ? 'sparkline-large' : ''}`} viewBox={large ? '0 0 540 115' : '0 0 154 40'} preserveAspectRatio="none" aria-label="График динамики"><defs><linearGradient id="pulse-gradient" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#8cf0c8" stopOpacity=".28" /><stop offset="1" stopColor="#8cf0c8" stopOpacity="0" /></linearGradient></defs>{large && <path d={`M ${points.replaceAll(' ', ' L ')} L 540,115 L 0,115 Z`} fill="url(#pulse-gradient)" />}<polyline points={points} fill="none" stroke="#8cf0c8" strokeWidth={large ? 2.5 : 2} strokeLinecap="round" strokeLinejoin="round" /></svg>
}

function AlertListItem({ alert }: { alert: Alert }) {
  return <div className="alert-list-item"><span className={`alert-dot alert-${alert.severity}`} /><div><strong>{alert.title}</strong><span>{alert.region} · {alert.affectedTickets} обращений</span></div><Icon name="arrow" size={14} /></div>
}

function RegionRow({ region }: { region: RegionMetric }) {
  return <div className="region-table-row"><strong>{region.name}</strong><span>{region.tickets.toLocaleString('ru-RU')}</span><span className={region.change >= 0 ? 'trend-up' : 'trend-down'}>{region.change >= 0 ? '+' : ''}{region.change}%</span><span className={`risk-state risk-${region.risk}`}><i />{region.risk === 'critical' ? 'Внимание' : region.risk === 'watch' ? 'Наблюдение' : 'Стабильно'}</span></div>
}

function RegionsPage({ regions }: { regions: RegionMetric[] }) {
  return <div className="analytics-page"><div className="metrics-grid"><MetricCard label="Регионов под наблюдением" value="20" change="3 сигнала" detail="за последние 24 часа" tone="mint" icon="map" /><MetricCard label="Максимальная динамика" value="+12,4%" change="г. Алматы" detail="к прошлой неделе" tone="rose" icon="trend" /><MetricCard label="Стабильная нагрузка" value="14" change="из 20 регионов" detail="без значимых сдвигов" tone="blue" icon="check" /></div><div className="panel"><PanelHeading title="Нагрузка по регионам" action="Скачать CSV" /><div className="region-table region-table-full"><div className="region-table-head"><span>Регион</span><span>Обращения</span><span>Динамика</span><span>Состояние</span><span>Тренд</span></div>{regions.map((region) => <div className="region-table-row region-table-row-full" key={region.name}><strong>{region.name}</strong><span>{region.tickets.toLocaleString('ru-RU')}</span><span className={region.change >= 0 ? 'trend-up' : 'trend-down'}>{region.change >= 0 ? '+' : ''}{region.change}%</span><span className={`risk-state risk-${region.risk}`}><i />{region.risk === 'critical' ? 'Внимание' : region.risk === 'watch' ? 'Наблюдение' : 'Стабильно'}</span><Sparkline /></div>)}</div></div></div>
}

function TopicsPage({ topics }: { topics: TopicMetric[] }) {
  return <div className="analytics-page"><div className="topics-layout"><section className="panel topic-distribution"><PanelHeading title="Распределение по темам" action="Таксономия v1.4" /><div className="donut-wrap"><div className="donut"><div><strong>100%</strong><span>обращений</span></div></div><div className="donut-legend">{topics.map((topic) => <div key={topic.name}><span className="legend-color" style={{ background: topic.color }} /><span>{topic.name}</span><strong>{topic.value}%</strong></div>)}</div></div></section><section className="panel topic-rank"><PanelHeading title="Изменения за неделю" /><div className="topic-rank-list">{topics.map((topic, index) => <div className="topic-rank-row" key={topic.name}><span className="rank-number">{String(index + 1).padStart(2, '0')}</span><span className="rank-color" style={{ background: topic.color }} /><strong>{topic.name}</strong><span className={topic.change >= 0 ? 'trend-up' : 'trend-down'}>{topic.change >= 0 ? '+' : ''}{topic.change}%</span><div className="rank-bar"><span style={{ width: `${Math.min(100, Math.abs(topic.change) * 8 + 14)}%`, background: topic.color }} /></div></div>)}</div></section></div></div>
}

function TimeSeriesPage() {
  return <div className="analytics-page"><section className="panel time-series-panel"><div className="panel-heading"><div><h2>Обращения по дням</h2><p className="panel-subtitle">Последние 30 дней · все регионы</p></div><div className="chart-controls"><button className="period-button active">Все</button><button className="period-button">Новые</button><button className="period-button">Повторные</button></div></div><div className="big-chart"><div className="y-axis"><span>900</span><span>600</span><span>300</span><span>0</span></div><div className="chart-area"><div className="chart-grid-lines"><i /><i /><i /><i /></div><svg viewBox="0 0 900 290" preserveAspectRatio="none" aria-label="Динамика обращений за 30 дней"><defs><linearGradient id="area-gradient" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#8cf0c8" stopOpacity=".32" /><stop offset="1" stopColor="#8cf0c8" stopOpacity="0" /></linearGradient></defs><path d="M0 205 L30 218 L60 195 L90 202 L120 172 L150 182 L180 160 L210 170 L240 146 L270 152 L300 130 L330 139 L360 117 L390 123 L420 100 L450 109 L480 88 L510 104 L540 81 L570 88 L600 63 L630 73 L660 51 L690 60 L720 35 L750 48 L780 29 L810 39 L840 22 L870 31 L900 15 L900 290 L0 290Z" fill="url(#area-gradient)" /><path d="M0 205 L30 218 L60 195 L90 202 L120 172 L150 182 L180 160 L210 170 L240 146 L270 152 L300 130 L330 139 L360 117 L390 123 L420 100 L450 109 L480 88 L510 104 L540 81 L570 88 L600 63 L630 73 L660 51 L690 60 L720 35 L750 48 L780 29 L810 39 L840 22 L870 31 L900 15" fill="none" stroke="#8cf0c8" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" /></svg><div className="x-axis"><span>23 авг</span><span>30 авг</span><span>06 сен</span><span>13 сен</span><span>20 сен</span></div></div></div></section><div className="metrics-grid"><MetricCard label="Пиковый день" value="824" change="18 сен" detail="+14% к обычному" tone="mint" icon="trend" /><MetricCard label="Среднее в день" value="648" change="+6,8%" detail="к предыдущему периоду" tone="blue" icon="pulse" /><MetricCard label="Повторные" value="11,4%" change="−0,8 п.п." detail="доля от потока" tone="amber" icon="cycle" /></div></div>
}

function AlertsPage({ alerts, onToast }: { alerts: Alert[]; onToast: (message: string) => void }) {
  const [selected, setSelected] = useState<Alert | null>(alerts[0] ?? null)
  return <div className="analytics-page"><div className="alerts-layout"><section className="panel alerts-list-panel"><PanelHeading title="Все оповещения" action="Настроить пороги" /><div className="alert-filter-row"><button className="filter-chip active">Все <span>{alerts.length}</span></button><button className="filter-chip">Новые <span>1</span></button><button className="filter-chip">В работе <span>1</span></button></div><div className="alerts-table">{alerts.map((alert) => <button className={`alert-row ${selected?.id === alert.id ? 'alert-row-active' : ''}`} key={alert.id} onClick={() => setSelected(alert)}><span className={`alert-dot alert-${alert.severity}`} /><span className="alert-row-main"><strong>{alert.title}</strong><span>{alert.description}</span><small>{alert.detectedAt} · {alert.region}</small></span><span className="alert-count">{alert.affectedTickets}<small>обращений</small></span><Icon name="chevron" size={16} /></button>)}</div></section>{selected && <section className="panel alert-detail-panel"><div className="detail-overline"><span className={`alert-dot alert-${selected.severity}`} />{selected.status}</div><h2>{selected.title}</h2><p>{selected.description}</p><div className="alert-facts"><div><span>Регион</span><strong>{selected.region}</strong></div><div><span>Тема</span><strong>{selected.topic}</strong></div><div><span>Обнаружено</span><strong>{selected.detectedAt}</strong></div><div><span>Затронуто</span><strong>{selected.affectedTickets} обращений</strong></div></div><div className="alert-detail-chart"><div className="field-label">Сигнал за 24 часа</div><Sparkline large /></div><button className="button button-primary full-width" onClick={() => onToast(`Открыта очередь по сигналу «${selected.title}»`)}>Открыть связанные обращения <Icon name="arrow" size={16} /></button></section>}</div></div>
}

function ForecastPage({ forecast }: { forecast: ForecastPoint[] }) {
  return <div className="analytics-page"><section className="panel forecast-panel"><div className="panel-heading"><div><h2>Прогноз нагрузки</h2><p className="panel-subtitle">Seasonal Naive · горизонт 5 дней · обновлено сегодня в 08:00</p></div><span className="model-status-chip"><i />модель в норме</span></div><div className="forecast-chart"><div className="y-axis"><span>900</span><span>600</span><span>300</span><span>0</span></div><div className="chart-area"><div className="chart-grid-lines"><i /><i /><i /><i /></div><div className="forecast-bars">{forecast.map((point, index) => <div className={`forecast-column ${point.forecast ? 'forecast-column-predicted' : ''}`} key={`${point.label}-${index}`}><div className="forecast-range" style={point.forecast ? { height: `${((point.high ?? 0) - (point.low ?? 0)) / 3.2}px`, bottom: `${(point.low ?? 0) / 3.2}px` } : undefined} /><div className="forecast-bar" style={{ height: `${((point.actual ?? point.forecast ?? 0) / 900) * 230}px` }} /><span>{point.label}</span></div>)}</div></div></div><div className="forecast-legend"><span><i className="legend-dot mint-dot" />фактические обращения</span><span><i className="legend-dot blue-dot" />прогноз</span><span><i className="legend-band" />диапазон неопределённости</span></div></section><div className="forecast-callout"><div className="callout-icon"><Icon name="forecast" size={20} /></div><div><strong>Ожидается рост к пятнице</strong><p>Прогноз показывает до 824 обращений в пятницу (+8,2% к среднему). Проверьте доступность операторов в Алматы и Астане.</p></div></div></div>
}

function ReportsPage({ onToast }: { onToast: (message: string) => void }) {
  const reports = [{ title: 'Еженедельная сводка по обращениям', meta: '16–22 сентября 2026 · 8 страниц', type: 'PDF', color: 'rose' }, { title: 'Региональные показатели SLA', meta: 'Сентябрь 2026 · 20 регионов', type: 'XLSX', color: 'mint' }, { title: 'Контрольная выборка обучения', meta: 'Цикл LC-2026-09-21 · 1 842 решения', type: 'PDF', color: 'blue' }]
  return <div className="analytics-page"><section className="reports-toolbar"><div className="report-filter"><Icon name="search" size={16} /><input aria-label="Поиск отчётов" placeholder="Найти отчёт" /></div><button className="button button-primary" onClick={() => onToast('Новый отчёт добавлен в очередь формирования')}><span>+</span>Сформировать отчёт</button></section><section className="panel reports-panel"><PanelHeading title="Последние отчёты" action="Архив отчётов" /><div className="reports-list">{reports.map((report) => <div className="report-row" key={report.title}><div className={`report-file report-file-${report.color}`}><Icon name="file" size={19} /><small>{report.type}</small></div><div className="report-copy"><strong>{report.title}</strong><span>{report.meta}</span></div><span className="report-ready"><i />Готов</span><button className="button button-quiet"><Icon name="download" size={15} />Скачать</button><button className="icon-button" aria-label={`Другие действия: ${report.title}`}><Icon name="more" size={17} /></button></div>)}</div></section></div>
}

function LearningPage({ learning }: { learning: LearningCycle }) {
  const steps = [{ key: 'COLLECT', label: 'Сбор обратной связи', detail: `${learning.feedbackCount.toLocaleString('ru-RU')} решений`, done: true }, { key: 'TRAIN', label: 'Обучение кандидата', detail: 'cls-2026-09-21.1', done: true }, { key: 'EVALUATE', label: 'Оценка в shadow', detail: 'Macro F1 · 0.901', done: true }, { key: 'REVIEW', label: 'Решение человека', detail: 'Ожидает ревью', done: false }]
  return <div className="analytics-page"><div className="learning-layout"><section className="panel learning-progress"><div className="panel-heading"><div><h2>Цикл {learning.id}</h2><p className="panel-subtitle">Последнее обновление · {learning.updatedAt}</p></div><span className="stage-chip">{learning.stage}</span></div><div className="learning-stepper">{steps.map((step, index) => <div className={`learning-step ${step.done ? 'step-done' : 'step-current'}`} key={step.key}><div className="step-marker">{step.done ? <Icon name="check" size={15} /> : index + 1}</div><div><strong>{step.label}</strong><span>{step.detail}</span></div>{index < steps.length - 1 && <div className="step-line" />}</div>)}</div><div className="learning-rule"><Icon name="help" size={17} /><p>Кандидат не попадёт в production автоматически. После ревью результат можно <strong>продвинуть</strong> или <strong>отклонить</strong>.</p></div></section><section className="panel dataset-panel"><PanelHeading title="Датасет цикла" action="Открыть манифест" /><div className="dataset-stat"><span>Версия</span><strong>{learning.dataset}</strong></div><div className="dataset-stat"><span>Решений оператора</span><strong>{learning.feedbackCount.toLocaleString('ru-RU')}</strong></div><div className="dataset-stat"><span>Кандидат</span><strong>{learning.candidate}</strong></div><div className="dataset-stat"><span>Языки</span><strong>RU · KZ</strong></div></section></div><section className="panel learning-log"><PanelHeading title="История цикла" action="Все события" /><div className="event-list"><div><span className="event-dot" /><span><strong>Кандидат готов к ревью</strong><small>Сегодня, 08:30 · ML Reviewer</small></span></div><div><span className="event-dot" /><span><strong>Shadow-оценка завершена</strong><small>Сегодня, 07:55 · Macro F1 0.901</small></span></div><div><span className="event-dot" /><span><strong>Выборка зафиксирована</strong><small>Вчера, 18:12 · 1 842 решения</small></span></div></div></section></div>
}

function ModelsPage({ models }: { models: ModelStatus[] }) {
  return <div className="analytics-page"><div className="model-summary"><div className="model-summary-copy"><span className="eyebrow"><span className="eyebrow-line" />Контур моделей</span><h2>Модели работают штатно</h2><p>Все production-версии отвечают за последние 24 часа. Один кандидат ожидает решения ревьюера.</p></div><div className="model-health"><span className="health-ring"><Icon name="check" size={22} /></span><strong>99,98%</strong><span>доступность inference</span></div></div><section className="panel models-panel"><PanelHeading title="Реестр моделей" action="Открыть реестр" /><div className="models-table"><div className="models-head"><span>Модель</span><span>Версия</span><span>Статус</span><span>Метрика</span><span>Обновлена</span><span /></div>{models.map((model) => <div className="models-row" key={model.version}><strong>{model.name}</strong><code>{model.version}</code><span className={`model-status model-${model.status}`}><i />{model.status === 'production' ? 'Production' : model.status === 'candidate' ? 'Кандидат' : 'Shadow'}</span><span><strong>{model.metricValue}</strong><small>{model.metric}</small></span><span>{model.updatedAt}</span><button className="icon-button" aria-label={`Действия для ${model.name}`}><Icon name="more" size={17} /></button></div>)}</div></section></div>
}

export default App

