import type { Priority, RuleProvenance, RuleSource } from './types'

const RULE_SOURCES = new Set<RuleSource>(['OFFICIAL', 'LABEL_HISTORY', 'MANUAL'])

export function mapPriority(value: string): Priority {
  const normalized = value.trim().toLocaleLowerCase()
  if (normalized === 'critical' || normalized === 'критический' || normalized === 'критично') return 'Критический'
  if (normalized === 'high' || normalized === 'высокий' || normalized === 'высокая') return 'Высокий'
  if (normalized === 'low' || normalized === 'низкий' || normalized === 'низкая') return 'Низкий'
  if (normalized === 'unknown' || normalized === 'unavailable' || !normalized) return 'Не определён'
  return 'Средний'
}

export function manualRuleProvenance(reason: string): RuleProvenance {
  return { source: 'MANUAL', version: null, reason }
}

export function mapRuleProvenance(
  value: { source?: string; version?: number | null; reason?: string } | undefined,
  fallbackReason: string,
): RuleProvenance {
  const source = value?.source && RULE_SOURCES.has(value.source as RuleSource)
    ? value.source as RuleSource
    : 'MANUAL'
  const version = typeof value?.version === 'number' && Number.isInteger(value.version) && value.version > 0
    ? value.version
    : null
  const reason = value?.reason?.trim() || fallbackReason
  return { source, version, reason }
}
