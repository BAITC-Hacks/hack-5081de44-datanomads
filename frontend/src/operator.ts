export interface RelatedCandidateInput {
  ticket_id: string
  score: number
  relation: string
  topic_id?: string
  region_id?: string
}

export interface RelatedCandidate extends RelatedCandidateInput {
  candidateTypes: Array<'similar' | 'duplicate' | 'repeat'>
}

type CandidateType = RelatedCandidate['candidateTypes'][number]

const DEFAULT_RELATED_TICKET_LIMIT = 3
const RELATION_PRIORITY: Record<'similar' | 'repeat' | 'duplicate', number> = {
  similar: 0,
  repeat: 1,
  duplicate: 2,
}

function relationType(value: string): 'similar' | 'repeat' | 'duplicate' {
  const normalized = value.trim().toLowerCase()
  if (normalized === 'duplicate' || normalized === 'дубликат') return 'duplicate'
  if (normalized === 'repeat' || normalized === 'повтор') return 'repeat'
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

      merged.set(candidate.ticket_id, {
        ticket_id: candidate.ticket_id,
        score: Math.max(existing?.score ?? 0, candidate.score),
        relation,
        ...(topicId === undefined ? {} : { topic_id: topicId }),
        ...(regionId === undefined ? {} : { region_id: regionId }),
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
  return candidates.slice(0, Math.max(0, limit))
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
