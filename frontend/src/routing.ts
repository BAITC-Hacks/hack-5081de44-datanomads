import type { ExplainabilityFact, Priority, RuleProvenance, RuleSource } from './types'

const RULE_SOURCES = new Set<RuleSource>(['OFFICIAL', 'LABEL_HISTORY', 'MANUAL'])
const RULE_FACT_FIELDS = new Set<ExplainabilityFact['field']>(['topic_id', 'region_id'])
const RULE_FACT_LABELS: Record<ExplainabilityFact['field'], string> = {
  topic_id: 'ID темы',
  region_id: 'ID региона',
}

export function formatExplainabilityFact(fact: ExplainabilityFact): string {
  return RULE_FACT_LABELS[fact.field] + ': ' + fact.value
}

export function mapPriority(value: string): Priority {
  const normalized = value.trim().toLocaleLowerCase()
  if (normalized === 'critical' || normalized === 'критический' || normalized === 'критично') return 'Критический'
  if (normalized === 'high' || normalized === 'высокий' || normalized === 'высокая') return 'Высокий'
  if (normalized === 'medium' || normalized === 'normal' || normalized === 'средний' || normalized === 'обычный') return 'Средний'
  if (normalized === 'low' || normalized === 'низкий' || normalized === 'низкая') return 'Низкий'
  if (normalized === 'unknown' || normalized === 'unavailable' || !normalized) return 'Не определён'
  return 'Не определён'
}

export function manualRuleProvenance(reason: string): RuleProvenance {
  return { source: 'MANUAL', version: null, reason }
}

export function mapRuleProvenance(
  value: {
    source?: string
    version?: number | null
    reason?: string
    facts_used?: Array<{ field?: string; value?: string }>
  } | undefined,
  fallbackReason: string,
): RuleProvenance {
  const source = value?.source && RULE_SOURCES.has(value.source as RuleSource)
    ? value.source as RuleSource
    : 'MANUAL'
  const version = typeof value?.version === 'number' && Number.isInteger(value.version) && value.version > 0
    ? value.version
    : null
  const reason = value?.reason?.trim() || fallbackReason
  const facts = Array.isArray(value?.facts_used) ? value.facts_used : []
  const factsUsed = facts.flatMap((fact) => {
    const field = fact.field as ExplainabilityFact['field'] | undefined
    const factValue = fact.value?.trim()
    return field && RULE_FACT_FIELDS.has(field) && factValue
      ? [{ field, value: factValue }]
      : []
  })
  return { source, version, reason, ...(factsUsed.length ? { factsUsed } : {}) }
}
