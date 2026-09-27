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
const RELATION_PRIORITY: Record<'similar' | 'repeat' | 'duplicate', number> = {
  similar: 0,
  repeat: 1,
  duplicate: 2,
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
