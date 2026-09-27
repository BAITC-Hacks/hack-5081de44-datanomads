import type { RelatedTicketDetail, Ticket } from './types'

export interface RelatedCandidateInput {
  ticket_id: string
  score: number
  relation: string
  topic_id?: string
  region_id?: string
  created_at?: string
  matched_factors?: string[]
  suggestion?: {
    score: number
    threshold: number
    rule_version: string
    model_version: string
    distance_metric: string
  }
}

export interface RelatedCandidate extends RelatedCandidateInput {
  candidateTypes: Array<'similar' | 'duplicate' | 'repeat'>
}

type CandidateType = RelatedCandidate['candidateTypes'][number]

const DEFAULT_RELATED_TICKET_LIMIT = 3
const RELATED_CANDIDATE_THRESHOLD = 0.78
const MAX_CONTEXT_PREVIEW_TEXT_LENGTH = 10_000
const RELATION_PRIORITY: Record<'similar' | 'repeat' | 'duplicate', number> = {
  similar: 0,
  repeat: 1,
  duplicate: 2,
}

export function buildContextPreviewText(originalText: string, answer: string): string {
  const normalizedAnswer = answer.trim()
  if (!normalizedAnswer) throw new Error('Укажите ответ заявителя')

  const text = `${originalText.trim()}\n\nОтвет заявителя: ${normalizedAnswer}`
  if ([...text].length > MAX_CONTEXT_PREVIEW_TEXT_LENGTH) {
    throw new Error('Текст с уточнением превышает лимит 10000 символов')
  }
  return text
}

export type TicketContextComparisonState = 'same' | 'different' | 'unavailable'

export interface TicketContextComparison {
  key: string
  label: string
  currentValue?: string
  relatedValue?: string
  state: TicketContextComparisonState
  currentSource?: string
  relatedSource?: string
}

function normalizedComparisonValue(value?: string | null): string | undefined {
  const normalized = value?.trim().toLocaleLowerCase()
  if (!normalized || [
    '—',
    'не указано',
    'не указан',
    'не определено',
    'не определён',
    'не определена',
    'не определены',
    'не предоставлено',
    'регион не указан',
    'тема не указана',
    'статус не указан',
    'время не указано',
    'unknown',
    'unavailable',
  ].includes(normalized)) {
    return undefined
  }
  return normalized
}

function compareTextFact(
  key: string,
  label: string,
  currentValue?: string | null,
  relatedValue?: string | null,
  currentComparisonValue?: string | null,
  relatedComparisonValue?: string | null,
): TicketContextComparison {
  const currentKey = normalizedComparisonValue(currentComparisonValue) ?? normalizedComparisonValue(currentValue)
  const relatedKey = normalizedComparisonValue(relatedComparisonValue) ?? normalizedComparisonValue(relatedValue)
  return {
    key,
    label,
    currentValue: currentValue?.trim() || undefined,
    relatedValue: relatedValue?.trim() || undefined,
    state: currentKey && relatedKey ? currentKey === relatedKey ? 'same' : 'different' : 'unavailable',
  }
}

function compareTimeFact(
  key: string,
  label: string,
  currentValue?: string | null,
  relatedValue?: string | null,
): TicketContextComparison {
  const currentTime = currentValue ? Date.parse(currentValue) : Number.NaN
  const relatedTime = relatedValue ? Date.parse(relatedValue) : Number.NaN
  return {
    key,
    label,
    currentValue: currentValue?.trim() || undefined,
    relatedValue: relatedValue?.trim() || undefined,
    state: Number.isFinite(currentTime) && Number.isFinite(relatedTime)
      ? currentTime === relatedTime ? 'same' : 'different'
      : 'unavailable',
  }
}

export function compareRelatedTicketContext(
  current: Ticket & { objectType?: string },
  related: RelatedTicketDetail & { objectType?: string },
): TicketContextComparison[] {
  const currentDecision = current.latestDecision
  const relatedDecision = related.latestDecision
  const confirmedTopic = (decision: Ticket['latestDecision']) =>
    decision?.confirmedTopicLabel ?? decision?.confirmedTopicId
  const topicComparison = compareTextFact(
    'topic',
    'Тема',
    current.topic,
    related.topic,
    currentDecision?.confirmedTopicId ?? current.predictedTopicId ?? current.topicId
      ?? (current.latestDecision ? confirmedTopic(currentDecision) : current.predictedTopic ?? current.topic),
    relatedDecision?.confirmedTopicId ?? related.topicId
      ?? (relatedDecision ? confirmedTopic(relatedDecision) : related.topic),
  )

  return [
    {
      ...topicComparison,
      currentSource: currentDecision ? 'решение оператора' : 'рекомендация Pulse',
      relatedSource: relatedDecision ? 'решение оператора' : 'значение записи источника',
    },
    compareTextFact(
      'object_type',
      'Тип объекта',
      current.objectType,
      related.objectType,
    ),
    compareTextFact('region', 'Регион', current.region, related.region, current.regionId, related.regionId),
    compareTimeFact('created_at', 'Создано', current.createdAt, related.createdAt),
    compareTimeFact('closed_at', 'Закрыто', current.closedAt, related.closedAt),
    compareTextFact('source_status', 'Статус источника', current.sourceStatus, related.status),
    compareTextFact('confirmed_topic', 'Подтверждённая тема',
      currentDecision ? confirmedTopic(currentDecision) : undefined,
      relatedDecision ? confirmedTopic(relatedDecision) : undefined),
    compareTextFact('confirmed_service', 'Подтверждённая служба', currentDecision?.service, relatedDecision?.service),
    compareTextFact('confirmed_action', 'Действие оператора', currentDecision?.action, relatedDecision?.action),
    compareTextFact('confirmed_priority', 'Подтверждённый приоритет', currentDecision?.priority, relatedDecision?.priority),
  ]
}

function relationType(value: string): 'similar' | 'repeat' | 'duplicate' {
  const normalized = value.trim().toLowerCase()
  if (normalized === 'duplicate' || normalized === 'дубликат' || normalized === 'возможный дубликат') return 'duplicate'
  if (normalized === 'repeat' || normalized === 'повтор' || normalized === 'возможное повторное обращение') return 'repeat'
  return 'similar'
}

export function mapTicketChannel(source?: string | null) {
  switch (source?.trim().toLowerCase()) {
    case 'egov':
    case 'e-gov':
    case 'e_gov':
      return 'eGov' as const
    case 'call-center':
    case 'call_center':
    case 'call center':
      return 'Call-центр' as const
    case 'mobile':
    case 'mobile-app':
    case 'mobile_app':
      return 'Мобильное приложение' as const
    case 'whatsapp':
      return 'WhatsApp' as const
    default:
      return 'Не указан' as const
  }
}

export function combineRelatedCandidates(
  similar: readonly RelatedCandidateInput[] = [],
  duplicates: readonly RelatedCandidateInput[] = [],
  repeats: readonly RelatedCandidateInput[] = [],
): RelatedCandidate[] {
  const merged = new Map<string, RelatedCandidate>()
  const groups: Array<[CandidateType, readonly RelatedCandidateInput[]]> = [
    ['similar', similar],
    ['duplicate', duplicates],
    ['repeat', repeats],
  ]

  for (const [candidateType, candidates] of groups) {
    for (const candidate of candidates) {
      const existing = merged.get(candidate.ticket_id)
      const candidateRelation = relationType(candidate.relation)
      const candidateTypes = existing?.candidateTypes ?? []
      const nextTypes = candidateTypes.includes(candidateType)
        ? candidateTypes
        : [...candidateTypes, candidateType]
      const relation = existing && RELATION_PRIORITY[relationType(existing.relation)] > RELATION_PRIORITY[candidateRelation]
        ? existing.relation
        : candidate.relation
      const topicId = existing?.topic_id ?? candidate.topic_id
      const regionId = existing?.region_id ?? candidate.region_id
      const createdAt = existing?.created_at ?? candidate.created_at
      const relationIsCandidate = !existing || RELATION_PRIORITY[candidateRelation] >= RELATION_PRIORITY[relationType(existing.relation)]
      const matchedFactors = existing?.matched_factors ?? candidate.matched_factors
      const suggestion = relationIsCandidate ? candidate.suggestion ?? existing?.suggestion : existing?.suggestion ?? candidate.suggestion

      merged.set(candidate.ticket_id, {
        ticket_id: candidate.ticket_id,
        score: Math.max(existing?.score ?? 0, candidate.score),
        relation,
        ...(topicId === undefined ? {} : { topic_id: topicId }),
        ...(regionId === undefined ? {} : { region_id: regionId }),
        ...(createdAt === undefined ? {} : { created_at: createdAt }),
        ...(matchedFactors === undefined ? {} : { matched_factors: matchedFactors }),
        ...(suggestion === undefined ? {} : { suggestion }),
        candidateTypes: nextTypes,
      })
    }
  }

  return [...merged.values()].sort((left, right) => right.score - left.score || left.ticket_id.localeCompare(right.ticket_id))
}

export function topRelatedCandidates(
  candidates: readonly RelatedCandidate[],
  limit = DEFAULT_RELATED_TICKET_LIMIT,
): RelatedCandidate[] {
  return candidates
    .filter((candidate) => Number.isFinite(candidate.score)
      && candidate.score >= (candidate.suggestion?.threshold ?? RELATED_CANDIDATE_THRESHOLD))
    .slice(0, Math.max(0, limit))
}

const RELATED_FACTOR_LABELS: Record<string, string> = {
  topic_match: 'Совпадает тема',
  region_match: 'Совпадает регион',
  within_30_days: 'В пределах 30 дней',
}

export function mapRelatedFactors(factors?: readonly string[]) {
  if (!factors) return []
  return [...new Set(factors.flatMap((factor) => {
    const label = RELATED_FACTOR_LABELS[factor]
    return label ? [label] : []
  }))]
}

export function formatRuntimeRate(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value) || value < 0 || value > 1) return '—'
  return `${Math.round(value * 100)}%`
}

export function formatDecisionTime(value: number | null | undefined, emptyLabel = '—') {
  if (value == null || !Number.isFinite(value) || value < 0) return emptyLabel
  const totalMinutes = Math.round(value)
  if (totalMinutes < 60) return `${totalMinutes} мин`
  const hours = Math.floor(totalMinutes / 60)
  const minutes = totalMinutes % 60
  return minutes ? `${hours} ч ${minutes} мин` : `${hours} ч`
}

function isKnownIdentifier(value?: string) {
  const normalized = value?.trim().toLowerCase()
  return Boolean(normalized && normalized !== 'unknown' && normalized !== 'unavailable')
}

function identifiersMatch(left?: string, right?: string) {
  return isKnownIdentifier(left) && isKnownIdentifier(right) && left?.trim().toLowerCase() === right?.trim().toLowerCase()
}

export function matchingTicketFactors(
  currentTopicId: string,
  currentRegionId: string,
  candidateTopicId?: string,
  candidateRegionId?: string,
) {
  const factors: string[] = []
  if (identifiersMatch(currentTopicId, candidateTopicId)) factors.push('Совпадает тема')
  if (identifiersMatch(currentRegionId, candidateRegionId)) factors.push('Совпадает регион')
  return factors
}
