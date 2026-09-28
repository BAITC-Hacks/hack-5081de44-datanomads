import type { DashboardFilters } from '../api/client'
import { useEffect, useRef, useState } from 'react'
import type { DashboardData } from '../types'
import { Icon, type IconName } from './Icon'
import type { DemoRole } from '../session'
import { useUiSettings } from '../uiSettings'

export type Route =
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

export const routeTitles: Record<Route, { eyebrow: string; title: string; description: string }> = {
  '/operator': { eyebrow: 'Рабочее место', title: 'Входящие обращения', description: 'Решения оператора, которым можно доверять' },
  '/situation/overview': { eyebrow: 'Центр ситуации', title: 'Обзор потока', description: 'Что происходит с обращениями прямо сейчас' },
  '/situation/regions': { eyebrow: 'Центр ситуации', title: 'Регионы', description: 'Где меняется нагрузка и появляется риск' },
  '/situation/topics': { eyebrow: 'Центр ситуации', title: 'Темы обращений', description: 'Распределение спроса по таксономии Pulse' },
  '/situation/time-series': { eyebrow: 'Центр ситуации', title: 'Временная динамика', description: 'Ритм обращений за последние 30 дней' },
  '/situation/alerts': { eyebrow: 'Центр ситуации', title: 'Оповещения', description: 'Сигналы, которые требуют внимания команды' },
  '/situation/forecast': { eyebrow: 'Центр ситуации', title: 'Прогноз', description: 'Ожидаемая нагрузка на ближайшие дни' },
  '/situation/reports': { eyebrow: 'Центр ситуации', title: 'Отчёты', description: 'Срезы для руководителей и рабочих встреч' },
  '/situation/learning': { eyebrow: 'Контроль моделей', title: 'Цикл обучения', description: 'Как обратная связь становится улучшением модели' },
  '/situation/models': { eyebrow: 'Контроль моделей', title: 'Статус моделей', description: 'Версии, метрики и решение о продвижении' },
  '/situation/audit': { eyebrow: 'Администрирование', title: 'Журнал аудита', description: 'Кто, когда и с каким объектом выполнял действие' },
}

type NavItemDefinition = { label: string; route: Route; icon: IconName; badge?: string }

const operatorNav: NavItemDefinition[] = [
  { label: 'Входящие', route: '/operator', icon: 'inbox' },
]

const situationNav: NavItemDefinition[] = [
  { label: 'Обзор', route: '/situation/overview', icon: 'grid' },
  { label: 'Регионы', route: '/situation/regions', icon: 'map' },
  { label: 'Темы', route: '/situation/topics', icon: 'tag' },
  { label: 'Временная динамика', route: '/situation/time-series', icon: 'trend' },
  { label: 'Оповещения', route: '/situation/alerts', icon: 'bell' },
  { label: 'Прогноз', route: '/situation/forecast', icon: 'forecast' },
  { label: 'Отчёты', route: '/situation/reports', icon: 'file' },
  { label: 'Цикл обучения', route: '/situation/learning', icon: 'cycle' },
  { label: 'Статус моделей', route: '/situation/models', icon: 'model' },
  { label: 'Журнал аудита', route: '/situation/audit', icon: 'clock' },
]

export function Sidebar({ route, role, mobileOpen, activeAlertCount, onNavigate }: { route: Route; role: DemoRole; mobileOpen: boolean; activeAlertCount: number; onNavigate: (route: Route) => void }) {
  const { t } = useUiSettings()
  const navWithAlertCount = situationNav.map((item) => ({
    ...item,
    badge: item.route === '/situation/alerts' && activeAlertCount > 0 ? String(activeAlertCount) : undefined,
  }))

  return <>
    {mobileOpen && <button className="sidebar-scrim mobile-scrim" aria-label={t('Закрыть меню')} onClick={() => onNavigate(route)} />}
    <aside className={`sidebar ${mobileOpen ? 'open' : ''}`} aria-label={t('Основная навигация')}>
      <div className="brand brand-lockup">
        <div className="brand-mark"><span>109</span></div>
        <div><strong>PULSE 109</strong></div>
      </div>
      <nav className="nav-list nav-scroll">
        {role === 'OPERATOR' && <><div className="sidebar-section-label nav-label">{t('Оператор')}</div>{operatorNav.map((item) => <NavItem key={item.route} item={item} active={route === item.route} onNavigate={onNavigate} />)}</>}
        {role === 'MANAGER' && <><div className="sidebar-section-label nav-label">{t('Центр ситуации')}</div>{navWithAlertCount.slice(0, 7).map((item) => <NavItem key={item.route} item={item} active={route === item.route} onNavigate={onNavigate} />)}<div className="sidebar-section-label nav-label">{t('Администрирование')}</div><NavItem item={navWithAlertCount[9]} active={route === '/situation/audit'} onNavigate={onNavigate} /></>}
        {role === 'ML_REVIEWER' && <><div className="sidebar-section-label nav-label">{t('Контроль моделей')}</div>{navWithAlertCount.slice(7, 9).map((item) => <NavItem key={item.route} item={item} active={route === item.route} onNavigate={onNavigate} />)}</>}
      </nav>
    </aside>
  </>
}

function NavItem({ item, active, onNavigate }: { item: NavItemDefinition; active: boolean; onNavigate: (route: Route) => void }) {
  const { t } = useUiSettings()
  return <button className={`nav-item ${active ? 'nav-item-active active' : ''}`} onClick={() => onNavigate(item.route)} aria-current={active ? 'page' : undefined}><span className="nav-icon"><Icon name={item.icon} size={18} /></span><span className="nav-label-text">{t(item.label)}</span>{item.badge && <span className="nav-badge">{item.badge}</span>}</button>
}

export function Topbar({ route, role, systemStatus, onOpenNav, onLogout }: { route: Route; role: DemoRole; systemStatus: string; onOpenNav: () => void; onLogout: () => void }) {
  const { t, locale, setLocale, theme, setTheme } = useUiSettings()
  const profileRef = useRef<HTMLDetailsElement>(null)
  useEffect(() => {
    const closeOnOutside = (event: PointerEvent) => {
      if (profileRef.current && !profileRef.current.contains(event.target as Node)) profileRef.current.open = false
    }
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && profileRef.current) profileRef.current.open = false
    }
    document.addEventListener('pointerdown', closeOnOutside)
    document.addEventListener('keydown', closeOnEscape)
    return () => { document.removeEventListener('pointerdown', closeOnOutside); document.removeEventListener('keydown', closeOnEscape) }
  }, [])
  const roleLabel = role === 'MANAGER' ? t('Руководитель') : role === 'ML_REVIEWER' ? t('Проверяющий ML') : t('Оператор')
  return <header className="topbar">
    <div className="crumb">
      <button className="icon-button icon-btn mobile-menu-button mobile-menu" aria-label={t('Открыть меню')} onClick={onOpenNav}><span className="menu-lines" /></button>
      <strong>PULSE</strong><span>› {t(routeTitles[route].title)}</span>
    </div>
    <div className="top-actions">
      <span className="status-pill"><i className={`env-dot ${systemStatus === 'API недоступен' ? 'status-dot-error' : systemStatus === 'Демо-данные' || systemStatus === 'Подключение…' ? 'status-dot-demo' : ''}`} />{t(systemStatus)}</span>
      <details className="profile-menu" ref={profileRef}>
        <summary aria-label={t('Профиль')}><span className="profile-avatar"><Icon name="user" size={18} /></span><span className="profile-role">{roleLabel}</span><Icon name="chevron" size={14} /></summary>
        <div className="profile-popover">
          <div className="profile-heading"><strong>{roleLabel}</strong><span>{t('Тестовый режим · действия записываются от выбранной роли')}</span></div>
          <div className="profile-setting"><span>{t('Язык')}</span><div className="profile-segment" role="group" aria-label={t('Язык')}><button type="button" aria-pressed={locale === 'ru'} onClick={() => setLocale('ru')}>RU</button><button type="button" aria-pressed={locale === 'kk'} onClick={() => setLocale('kk')}>ҚАЗ</button></div></div>
          <div className="profile-setting"><span>{t('Оформление')}</span><button type="button" className="profile-theme" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}><Icon name={theme === 'dark' ? 'sun' : 'moon'} size={17} />{t(theme === 'dark' ? 'Светлая тема' : 'Тёмная тема')}</button></div>
          <button type="button" className="profile-logout" onClick={onLogout}><Icon name="logout" size={17} />{t('Выйти')}</button>
        </div>
      </details>
    </div>
  </header>
}

export function PageHeader({ eyebrow, title, description, route, filters, onFiltersChange, filterOptions }: { eyebrow: string; title: string; description: string; route: Route; filters: DashboardFilters; onFiltersChange: (filters: DashboardFilters) => void; filterOptions?: DashboardData['filterOptions'] }) {
  const { t } = useUiSettings()
  const [mobileFiltersOpen, setMobileFiltersOpen] = useState(false)
  const showFilters = route.startsWith('/situation/') && !['/situation/audit', '/situation/learning', '/situation/models'].includes(route) && filterOptions
  const resetFilters = () => onFiltersChange({ range: filters.range, forecastHorizon: filters.forecastHorizon })
  const activeFilterCount = [filters.regionId, filters.topicId, filters.serviceId, filters.status, filters.district, filters.channel].filter(Boolean).length

  return <>
    <div className="page-header page-head">
      <div className="page-copy"><h1>{t(title)}</h1><span className="sr-only">{t(eyebrow)}. {t(description)}</span></div>
    </div>
    {showFilters && <button type="button" className="mobile-filters-toggle" aria-expanded={mobileFiltersOpen} aria-controls="analytics-filters" onClick={() => setMobileFiltersOpen((open) => !open)}><Icon name="filter" size={17} />{t(mobileFiltersOpen ? 'Скрыть фильтры' : 'Показать фильтры')}{activeFilterCount > 0 && <span>{activeFilterCount}</span>}</button>}
    {showFilters && <div id="analytics-filters" className={`period-control toolbar analytics-filter-selects ${mobileFiltersOpen ? '' : 'mobile-collapsed'}`} aria-label={t('Фильтры аналитики')}>
      {route !== '/situation/forecast' && <div className="period-buttons seg" role="group" aria-label={t('Период')}>
        {([['7d', '7 дней'], ['30d', '30 дней'], ['90d', '90 дней']] as const).map(([value, label]) => <button type="button" key={value} className={`period-button ${filters.range === value ? 'active' : ''}`} aria-pressed={filters.range === value} onClick={() => onFiltersChange({ ...filters, range: value })}>{t(label)}</button>)}
      </div>}
      <label className="field"><span>{t('Регион')}</span><select value={filters.regionId ?? ''} onChange={(event) => onFiltersChange({ ...filters, regionId: event.target.value || undefined })}><option value="">{t('Все регионы')}</option>{filterOptions.regions.map((option) => <option value={option.id} key={option.id}>{t(option.label)}</option>)}</select></label>
      <label className="field"><span>{t('Тема')}</span><select value={filters.topicId ?? ''} onChange={(event) => onFiltersChange({ ...filters, topicId: event.target.value || undefined })}><option value="">{t('Все темы')}</option>{filterOptions.topics.map((option) => <option value={option.id} key={option.id}>{t(option.label)}</option>)}</select></label>
      {filterOptions.services.length > 0 && <label className="field"><span>{t('Служба')}</span><select value={filters.serviceId ?? ''} onChange={(event) => onFiltersChange({ ...filters, serviceId: event.target.value || undefined })}><option value="">{t('Все службы')}</option>{filterOptions.services.map((option) => <option value={option.id} key={option.id}>{t(option.label)}</option>)}</select></label>}
      {filterOptions.statuses.length > 0 && <label className="field"><span>{t('Статус')}</span><select value={filters.status ?? ''} onChange={(event) => onFiltersChange({ ...filters, status: event.target.value || undefined })}><option value="">{t('Все статусы')}</option>{filterOptions.statuses.map((option) => <option value={option.id} key={option.id}>{t(option.label)}</option>)}</select></label>}
      {filterOptions.districts.length > 0 && <label className="field"><span>{t('Район')}</span><select value={filters.district ?? ''} onChange={(event) => onFiltersChange({ ...filters, district: event.target.value || undefined })}><option value="">{t('Все районы')}</option>{filterOptions.districts.map((option) => <option value={option.id} key={option.id}>{t(option.label)}</option>)}</select></label>}
      {filterOptions.channels.length > 0 && <label className="field"><span>{t('Канал')}</span><select value={filters.channel ?? ''} onChange={(event) => onFiltersChange({ ...filters, channel: event.target.value || undefined })}><option value="">{t('Все каналы')}</option>{filterOptions.channels.map((option) => <option value={option.id} key={option.id}>{t(option.label)}</option>)}</select></label>}
      <button type="button" className="button button-quiet quiet filter-reset" onClick={resetFilters}>{t('Сбросить')}</button>
    </div>}
  </>
}
