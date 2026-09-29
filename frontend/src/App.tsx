import { translateUi } from './uiSettings'
import { useCallback, useEffect, useState, type ReactElement } from 'react'
import { loadAnalyticsDrilldown, loadDashboard, subscribeToAlertChanges } from './api/client'
import type { AnalyticsDrilldownTicket, DashboardFilters, DrilldownDimension } from './api/client'
import type { ApiSource, DashboardData, DatasetProvenance } from './types'
import { PageHeader, routeTitles, Sidebar, Topbar, type Route } from './components/AppChrome'
import { AuditLogPage } from './components/AuditLogPage'
import { Icon } from './components/Icon'
import { OperatorPage } from './pages/OperatorPage'
import { CleanAlertsPage, CleanForecastPage, CleanRegionsPage, CleanReportsPage, CleanTimeSeriesPage, CleanTopicsPage } from './pages/SituationPages'
import { CleanLearningPage, CleanModelsPage } from './pages/LearningPages'
import { OverviewPage } from './pages/OverviewPage'
import { canOpenRoute, readDemoRole, roleHome, saveDemoRole, type DemoRole } from './session'
import { useUiSettings } from './uiSettings'

type DrilldownHandler = (dimension: DrilldownDimension, value: string | undefined, label: string) => void
type DrilldownState = { label: string; items: AnalyticsDrilldownTicket[]; total: number } | null
function datasetProvenanceNotice(provenance: DatasetProvenance | undefined, locale: 'ru' | 'kk'): string | undefined {
  if (!provenance) return undefined

  const parts: string[] = []
  if (provenance.synthetic_ticket_count > 0 && provenance.real_ticket_count > 0) {
    parts.push(locale === 'kk' ? `Қосылған деректерде синтетикалық жазбалар (${provenance.synthetic_ticket_count}) және нақты жазбалар (${provenance.real_ticket_count}) бар.` : `В подключённых данных есть синтетические записи (${provenance.synthetic_ticket_count}) и реальные записи (${provenance.real_ticket_count}).`)
  } else if (provenance.synthetic_ticket_count > 0) {
    parts.push(locale === 'kk' ? `Синтетикалық жиын: ${provenance.synthetic_ticket_count} өтініш. Бұл тапсырыс берушінің операциялық статистикасы емес, демонстрациялық деректер.` : `Синтетический набор: ${provenance.synthetic_ticket_count} обращений. Это демонстрационные данные, не операционная статистика заказчика.`)
  } else if (provenance.real_ticket_count > 0) {
    parts.push(locale === 'kk' ? `Нақты жазбалар: ${provenance.real_ticket_count}.` : `Реальные записи: ${provenance.real_ticket_count}.`)
  }
  if (provenance.unassigned_ticket_count > 0) {
    parts.push(locale === 'kk' ? `Деректер жиынымен байланысы жоқ: ${provenance.unassigned_ticket_count}.` : `Без связи с набором данных: ${provenance.unassigned_ticket_count}.`)
  }
  if (provenance.quarantined_row_count > 0) {
    parts.push(locale === 'kk' ? `Карантинде: ${provenance.quarantined_row_count}.` : `В карантине: ${provenance.quarantined_row_count}.`)
  }

  return parts.length > 0 ? parts.join(' ') : undefined
}

function routeFromHash(): Route {
  const path = window.location.hash.replace(/^#/, '') || '/operator'
  return (Object.keys(routeTitles).includes(path) ? path : '/operator') as Route
}

function navigate(route: Route) {
  window.location.hash = route
}

function App() {
  const { locale, t } = useUiSettings()
  const [role, setRole] = useState<DemoRole | null>(readDemoRole)
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
    if (!role) return
    setLoading(true)
    try {
      const result = await loadDashboard(filters, role)
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
  }, [filters, role])

  useEffect(() => {
    if (role && !canOpenRoute(role, route)) navigate(roleHome[role] as Route)
  }, [role, route])

  useEffect(() => {
    const onHashChange = () => {
      setRoute(routeFromHash())
      window.scrollTo(0, 0)
    }
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  useEffect(() => {
    if (!role || !canOpenRoute(role, route) || route === '/situation/audit') return
    let active = true
    setLoading(true)
    loadDashboard(filters, role).then((result) => {
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
  }, [filters, route, role])

  useEffect(() => {
    if (role !== 'MANAGER' || !route.startsWith('/situation') || route === '/situation/audit' || source !== 'api') return
    let active = true
    const unsubscribe = subscribeToAlertChanges(() => {
      loadDashboard(filters, role).then((result) => {
        if (!active) return
        setData(result.data)
        setApiError(result.error)
      }).catch((error: unknown) => {
        if (active) setApiError(error instanceof Error ? error.message : 'Не удалось обновить оповещения')
      })
    })
    return () => { active = false; unsubscribe() }
  }, [route, role, source, filters])

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
    ? datasetProvenanceNotice(data?.datasetProvenance, locale)
    : undefined

  const chooseRole = (nextRole: DemoRole) => {
    saveDemoRole(nextRole)
    setRole(nextRole)
    setData(null)
    setLoading(true)
    navigate(roleHome[nextRole] as Route)
  }

  const logout = () => {
    saveDemoRole(null)
    setRole(null)
    setData(null)
    setMobileNavOpen(false)
    navigate('/operator')
  }

  if (!role) return <DemoAccountScreen onChoose={chooseRole} />
  if (!canOpenRoute(role, route)) return <LoadingState />

  return (
    <div className="app-shell app">
      <Sidebar
        route={route}
        role={role}
        mobileOpen={mobileNavOpen}
        activeAlertCount={data?.alerts.filter((alert) => alert.status !== 'Закрыт').length ?? 0}
        onNavigate={(next) => { navigate(next); setMobileNavOpen(false) }}
      />
      <main className="main-shell">
        <Topbar
          route={route}
          role={role}
          systemStatus={loading ? 'Подключение…' : source === 'api' ? (apiError ? 'API недоступен' : 'API подключён') : 'Демо-данные'}
          onOpenNav={() => setMobileNavOpen(true)}
          onLogout={logout}
        />
        {!loading && source === 'demo' && route !== '/situation/audit' && <div className="demo-banner"><span className="status-dot" /><span className="demo-banner-message">{t('Демо-данные · не являются операционной статистикой заказчика')}{apiError && <small className="demo-banner-detail">{t('Core API недоступен')}</small>}</span></div>}
        {!loading && provenanceNotice && (
          <div className="demo-banner">
            <span className="status-dot" />
            <span className="demo-banner-message">{provenanceNotice}</span>
          </div>
        )}
        {source === 'api' && apiError && route !== '/situation/audit' && <div className="error-banner"><span className="status-dot" /> {t('API недоступен')} · {apiError}</div>}
        <div className="page-wrap content">
          <PageHeader {...title} route={route} filters={filters} onFiltersChange={setFilters} filterOptions={data?.filterOptions} />
          {route === '/situation/audit' ? <AuditLogPage /> : loading ? <LoadingState /> : data ? <RouteContent route={route} data={data} onDataChange={setData} onRefresh={refreshDashboard} onToast={showToast} filters={filters} onFiltersChange={setFilters} onDrilldown={openDrilldown} drilldown={drilldown} drilldownLoading={drilldownLoading} /> : <ErrorState onRetry={() => window.location.reload()} />}
        </div>
      </main>
      {toast && <div role="status" aria-live="polite" className="toast"><span className="toast-check"><Icon name="check" size={15} /></span>{toast}<button aria-label={t('Закрыть уведомление')} className="icon-button toast-close" onClick={() => setToast(null)}><Icon name="close" size={15} /></button></div>}
    </div>
  )
}

function RouteContent({ route, data, onDataChange, onRefresh, onToast, filters, onFiltersChange, onDrilldown, drilldown, drilldownLoading }: { route: Exclude<Route, '/situation/audit'>; data: DashboardData; onDataChange: (data: DashboardData) => void; onRefresh: () => Promise<void>; onToast: (message: string) => void; filters: DashboardFilters; onFiltersChange: (filters: DashboardFilters) => void; onDrilldown: DrilldownHandler; drilldown: DrilldownState; drilldownLoading: boolean }) {
  let content: ReactElement
  switch (route) {
    case '/operator': content = <OperatorPage tickets={data.tickets} overview={data.overview} taxonomy={data.filterOptions} onDataChange={(tickets) => onDataChange({ ...data, tickets })} onToast={onToast} />; break
    case '/situation/overview': content = <OverviewPage data={data} filters={filters} onNavigate={navigate} onDrilldown={onDrilldown} />; break
    case '/situation/regions': content = <CleanRegionsPage regions={data.regions} onDrilldown={onDrilldown} />; break
    case '/situation/topics': content = <CleanTopicsPage topics={data.topics} onDrilldown={onDrilldown} />; break
    case '/situation/time-series': content = <CleanTimeSeriesPage timeSeries={data.timeSeries} onDrilldown={onDrilldown} />; break
    case '/situation/alerts': content = <CleanAlertsPage alerts={data.alerts} onRefresh={onRefresh} onToast={onToast} onDrilldown={onDrilldown} />; break
    case '/situation/forecast': content = <CleanForecastPage forecast={data.forecast} history={data.forecastHistory ?? []} previousForecast={data.forecastPreviousPoints ?? []} reforecast={data.forecastReforecast} managerSignals={data.forecastManagerSignals ?? []} capacityAssessment={data.forecastCapacityAssessment} runId={data.forecastRunId} issuedAt={data.forecastIssuedAt} status={data.forecastStatus} modelVersion={data.forecastModelVersion} model={data.forecastModel} source={data.forecastSource} insufficientHistory={data.forecastInsufficientHistory ?? false} forecastStart={data.forecastStart} expectedPeaks={data.forecastExpectedPeaks ?? []} backtest={data.forecastBacktest} horizon={filters.forecastHorizon ?? 30} filters={filters} filterOptions={data.filterOptions} onHorizonChange={(horizon) => onFiltersChange({ ...filters, forecastHorizon: horizon })} />; break
    case '/situation/reports': content = <CleanReportsPage filters={filters} filterOptions={data.filterOptions} />; break
    case '/situation/learning': content = <CleanLearningPage learning={data.learning} onRefresh={onRefresh} onToast={onToast} />; break
    case '/situation/models': content = <CleanModelsPage models={data.models} driftTriggers={data.driftTriggers ?? []} canReview={data.canReviewDriftTriggers ?? false} onRefresh={onRefresh} onToast={onToast} />; break
  }
  return <>{content}{route !== '/operator' && <AnalyticsDrilldownPanel state={drilldown} loading={drilldownLoading} />}</>
}

function AnalyticsDrilldownPanel({ state, loading }: { state: DrilldownState; loading: boolean }) {
  if (loading) return <section className="panel drilldown-panel drilldown" aria-live="polite"><div className="panel-heading panel-head"><h2>{translateUi("Исходные обращения")}</h2></div><p className="panel-note">{translateUi("Загружаем обращения…")}</p></section>
  if (!state) return null
  return <section className="panel drilldown-panel drilldown" aria-live="polite"><div className="panel-heading panel-head"><h2>{translateUi("Обращения в выбранном срезе")}</h2><span className="drilldown-label">{state.label} · {state.total}</span></div><p className="panel-note">{translateUi("Исходный текст и контактные данные скрыты в аналитике.")}</p>{state.items.length ? <div className="drilldown-list">{state.items.map((ticket) => <article className="drilldown-ticket drill-item" key={ticket.id}><div><strong>{ticket.id}</strong><span>{ticket.region_name} · {ticket.topic_label}</span></div><small>{ticket.created_at} · {ticket.status} · {translateUi(ticket.priority)}</small></article>)}</div> : <p className="panel-note">{translateUi("За выбранный период обращений не найдено.")}</p>}</section>
}

function LoadingState() {
  return <div className="loading-grid" aria-label={translateUi("Загрузка данных")}><div className="skeleton skeleton-wide" /><div className="skeleton-row"><div className="skeleton" /><div className="skeleton" /><div className="skeleton" /></div><div className="skeleton skeleton-large" /></div>
}

function ErrorState({ onRetry }: { onRetry: () => void }) {
  return <div className="state-card"><div className="state-icon error-icon">!</div><h2>{translateUi("Не удалось загрузить рабочие данные")}</h2><p>{translateUi("Проверьте подключение к API и повторите попытку.")}</p><button className="button button-primary" onClick={onRetry}>{translateUi("Повторить")}</button></div>
}

function DemoAccountScreen({ onChoose }: { onChoose: (role: DemoRole) => void }) {
  const { t, locale, setLocale, theme, setTheme } = useUiSettings()
  const accounts: Array<{ role: DemoRole; title: string; description: string; icon: 'inbox' | 'grid' | 'model' }> = [
    { role: 'OPERATOR', title: 'Оператор', description: 'Работа с обращениями, рекомендациями и решениями', icon: 'inbox' },
    { role: 'MANAGER', title: 'Руководитель', description: 'Аналитика, сигналы, прогнозы и отчёты', icon: 'grid' },
    { role: 'ML_REVIEWER', title: 'Проверяющий ML', description: 'Циклы обучения, оценка и выпуск моделей', icon: 'model' },
  ]
  return <main className="account-screen">
    <div className="account-toolbar"><div className="account-brand"><span className="brand-mark"><span>109</span></span><strong>PULSE 109</strong></div><div className="account-preferences"><button type="button" className="account-preference" onClick={() => setLocale(locale === 'ru' ? 'kk' : 'ru')} aria-label={t('Язык')}>{locale === 'ru' ? 'ҚАЗ' : 'RU'}</button><button type="button" className="account-preference" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label={t(theme === 'dark' ? 'Светлая тема' : 'Тёмная тема')}><Icon name={theme === 'dark' ? 'sun' : 'moon'} size={18} /></button></div></div>
    <section className="account-card" aria-labelledby="account-title"><span className="account-kicker">PULSE 109 · DEMO</span><h1 id="account-title">{t('Выберите тестовый аккаунт')}</h1><p>{t('Для входа пароль не нужен. Выберите роль, чтобы увидеть соответствующее рабочее место.')}</p><div className="account-list">{accounts.map((account) => <button className="account-option" type="button" key={account.role} onClick={() => onChoose(account.role)}><span className="account-option-icon"><Icon name={account.icon} size={20} /></span><span className="account-option-copy"><strong>{t(account.title)}</strong><small>{t(account.description)}</small></span><Icon name="arrow" size={19} /></button>)}</div><small className="account-disclaimer">{t('Тестовый режим · действия записываются от выбранной роли')}</small></section>
  </main>
}

export default App
