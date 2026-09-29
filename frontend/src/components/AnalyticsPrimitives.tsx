import { localeTag, translateUi } from '../uiSettings'
import type { DrilldownDimension } from '../api/client'
import type { Alert, RegionMetric } from '../types'
import { Icon, type IconName } from './Icon'

export type DrilldownHandler = (dimension: DrilldownDimension, value: string | undefined, label: string) => void

export function MetricCard({ label, value, change, detail, tone, icon, onClick }: { label: string; value: string; change: string; detail: string; tone: string; icon: IconName; onClick?: () => void }) {
  return <article className={`metric-card metric-${tone}${onClick ? ' drilldown-row' : ''}`} role={onClick ? 'button' : undefined} tabIndex={onClick ? 0 : undefined} onClick={onClick} onKeyDown={onClick ? (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onClick() } } : undefined}><div className="metric-top"><span>{label}</span><span className="metric-icon"><Icon name={icon} size={17} /></span></div><div className="metric-value">{value}</div><div className="metric-bottom"><span className="metric-change">{change}</span><span>{detail}</span></div></article>
}

export function NoData({ message }: { message: string }) {
  return <div className="state-card"><div className="state-icon">—</div><p>{message}</p></div>
}

export function PanelHeading({ title, action, onClick }: { title: string; action?: string; onClick?: () => void }) {
  return <div className="panel-heading panel-head"><h2>{title}</h2>{action && <button className="text-button text-btn" onClick={onClick}>{action}<Icon name="arrow" size={14} /></button>}</div>
}

export function AlertListItem({ alert, onDrilldown }: { alert: Alert; onDrilldown: DrilldownHandler }) {
  return <button className="alert-list-item drilldown-row" onClick={() => onDrilldown('alert', alert.id, `Оповещение: ${alert.title}`)}><span className={`alert-dot alert-${alert.severity}`} /><span><strong>{translateUi(alert.title)}</strong><span>{translateUi(alert.region)} · {alert.affectedTickets} {translateUi("обращений")}</span></span><Icon name="arrow" size={14} /></button>
}

export function RegionRow({ region, onDrilldown }: { region: RegionMetric; onDrilldown?: DrilldownHandler }) {
  const row = <><strong>{translateUi(region.name)}</strong><span>{region.tickets.toLocaleString(localeTag())}</span><span>{region.previousTickets?.toLocaleString(localeTag()) ?? '—'}</span><span>{formatAnalyticsChange(region.changeAbs, region.change)}</span></>
  if (!onDrilldown) return <div className="region-table-row">{row}</div>
  return <button className="region-table-row drilldown-row" onClick={() => onDrilldown('region', region.id, `Регион: ${region.name}`)}>{row}</button>
}

export function formatAnalyticsChange(changeAbs: number | undefined, changePct: number | undefined): string {
  if (changeAbs == null) return '—'
  const absolute = `${changeAbs > 0 ? '+' : ''}${changeAbs.toLocaleString(localeTag())}`
  if (changePct == null) return absolute
  const percent = `${changePct > 0 ? '+' : ''}${changePct.toLocaleString(localeTag(), { maximumFractionDigits: 1 })}%`
  return `${absolute} (${percent})`
}
