import { demoData } from '../data/demo'
import type { Alert, AssistPreviewState, DashboardData, DatasetProvenance, ForecastPoint, LearningCycle, ModelStatus, OperatorRuntimeMetrics, Priority, RegionMetric, RelatedTicketDetail, RelationSuggestionSnapshot, RuleProvenance, SimilarTicket, Ticket, TopicMetric } from '../types'
import { mapLanguage } from '../language'
import { classificationAlternatives, normalizeConfidenceState } from '../classification'
import { mapPriority, mapRuleProvenance } from '../routing'
import { combineRelatedCandidates, mapRelatedFactors, mapTicketChannel, matchingTicketFactors, topRelatedCandidates } from '../operator'

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '/api/v1').replace(/\/$/, '')
const API_ROLE = import.meta.env.VITE_PULSE_ROLE ?? 'ADMIN'
const DEMO_ENABLED = import.meta.env.VITE_PULSE_DEMO === 'true'

export interface ApiResult<T> {
  data: T
  source: 'api' | 'demo'
  error?: string
}

export interface DashboardFilters {
  range: string
  regionId?: string
  topicId?: string
  serviceId?: string
  status?: string
  district?: string
  channel?: string
}

export type DrilldownDimension = 'overview' | 'region' | 'topic' | 'date' | 'alert'

function queryString(filters: DashboardFilters) {
  const params = new URLSearchParams({ range: filters.range })
  if (filters.regionId) params.set('region_id', filters.regionId)
  if (filters.topicId) params.set('topic_id', filters.topicId)
  if (filters.serviceId) params.set('service_id', filters.serviceId)
  if (filters.status) params.set('status', filters.status)
  if (filters.district) params.set('district', filters.district)
  if (filters.channel) params.set('channel', filters.channel)
  return params.toString()
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { Accept: 'application/json', 'X-Pulse-Role': API_ROLE, 'X-User-Id': 'pulse-web', ...options?.headers },
  })
  if (!response.ok) {
    throw new Error(`API ${response.status}`)
  }
  return response.json() as Promise<T>
}

export interface BackendTicket {
  id: string
  external_ref: string
  text: string
  language: string
  region_id: string
  region_name: string
  topic_id: string
  topic_label: string
  priority: string
  status: string
  source: string
  created_at: string
  closed_at?: string | null
  updated_at: string
}

interface BackendAlternative {
  topic_id: string
  topic_label: string
  confidence: number
}

interface BackendPrediction {
  ticket_id: string
  model_version: string
  topic_id: string
  topic_label: string
  confidence: number
  confidence_state: string
  recommended_service: string
  predicted_priority: string
  routing_reason?: string
  service_provenance?: RuleProvenance
  priority_provenance?: RuleProvenance
  alternatives: BackendAlternative[]
}

interface BackendSimilar {
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

interface BackendAssistPreview {
  prediction?: BackendPrediction
  similar_tickets?: BackendSimilar[]
  duplicate_candidates?: BackendSimilar[]
  repeat_candidates?: BackendSimilar[]
  response_template?: { body: string; approved?: boolean; source?: string }
  orchestration?: AssistPreviewState
}

interface BackendTicketDetail {
  ticket: BackendTicket
  prediction: BackendPrediction
  latest_decision?: {
    action: string
    predicted_topic_id?: string
    predicted_service?: string
    predicted_priority?: string
    model_version?: string
    confirmed_topic_id: string
    confirmed_topic_label?: string
    service: string
    priority: string
    service_provenance?: RuleProvenance
    priority_provenance?: RuleProvenance
    created_at?: string
  }
}

interface BackendDecisionResponse {
  ticket: BackendTicket
  prediction: BackendPrediction
  decision: NonNullable<BackendTicketDetail['latest_decision']>
}

interface BackendAnalytics {
  source?: string
  overview: { total_tickets: number; open_tickets: number; resolved_tickets: number; high_priority_tickets: number; operator_decisions?: number; confirmed_decisions?: number; corrected_decisions?: number; avg_decision_minutes?: number | null; change_abs?: number; change_pct?: number | null }
  runtime_metrics?: {
    operator_decision_time_minutes: number | null
    operator_decision_time_samples: number
    classification_correction_rate: number | null
    classification_corrections: number
    classification_decisions: number
    routing_correction_rate: number | null
    routing_corrections: number
    routing_decisions: number
    priority_correction_rate: number | null
    priority_corrections: number
    priority_decisions: number
    similarity_usefulness: number | null
    similarity_feedback_count: number
    duplicate_precision: number | null
    duplicate_feedback_count: number
  }
  by_region: Array<{ id: string; label: string; tickets: number; high_priority: number; avg_confidence: number; change_abs?: number; change_pct?: number }>
  by_topic: Array<{ id: string; label: string; tickets: number; high_priority: number; avg_confidence: number; change_abs?: number; change_pct?: number }>
  time_series: Array<{ date: string; tickets: number; resolved: number }>
}

interface BackendForecast { source?: string; status?: string; insufficient_history?: boolean; history?: Array<{ date: string; tickets: number; resolved: number }>; points: Array<{ date: string; tickets: number; resolved: number }>; model_version: string; model?: string; expected_peaks?: string[]; backtest?: Record<string, unknown> }
interface BackendAlert { id: string; severity: string; status: string; title: string; description: string; region_id: string; topic_id: string; ticket_count: number; detected_at: string }
interface BackendAlerts { source?: string; items: BackendAlert[] }
interface BackendLearningCycle { id: string; state: string; dataset_version: string; candidate_model_version: string; feedback_count: number; updated_at: string; decision_note?: string; metrics: { macro_f1?: number | null; accuracy?: number | null; evaluated_samples?: number } }
interface BackendLearning { source?: string; items?: BackendLearningCycle[]; active_cycle?: BackendLearningCycle; production_model?: { id: string; status: string }; controlled_loop?: Record<string, unknown> }
interface BackendModels { source?: string; items: Array<{ id: string; model_family: string; status: string; metrics: { macro_f1: number | null; accuracy: number | null }; created_at: string }> }
export interface CandidateEvaluation {
  cycle_id: string
  state: string
  offline_metrics: Record<string, unknown>
  shadow_metrics: Record<string, unknown>
  critical_regressions: unknown[]
  sample_size: number
  promotion_policy_version: string
  decision: string
}
export interface TaxonomyOption { id: string; label: string }
interface BackendTaxonomy {
  regions: TaxonomyOption[]
  topics: TaxonomyOption[]
  services: TaxonomyOption[]
  statuses?: TaxonomyOption[]
  districts?: TaxonomyOption[]
  channels?: TaxonomyOption[]
}

function mapLearningStage(value: string): LearningCycle['stage'] {
  switch (value.toUpperCase()) {
    case 'TRAINING':
    case 'TRAIN':
      return 'TRAIN'
    case 'EVALUATE':
      return 'EVALUATE'
    case 'DECISION':
      return 'DECISION'
    case 'PROMOTED':
      return 'PROMOTED'
    case 'REJECTED':
      return 'REJECTED'
    case 'INSUFFICIENT_FEEDBACK':
      return 'INSUFFICIENT_FEEDBACK'
    default:
      return 'COLLECT'
  }
}

function mapRelation(value: string): SimilarTicket['relation'] {
  const normalized = value.trim().toLowerCase()
  if (normalized === 'duplicate' || normalized === 'дубликат') return 'Возможный дубликат'
  if (normalized === 'repeat' || normalized === 'повтор') return 'Возможное повторное обращение'
  return 'Похожий'
}

function topicIdForLabel(label?: string, topics: TaxonomyOption[] = []) {
  if (!label) return undefined
  const option = topics.find((item) => item.id === label || item.label === label)
  if (option) return option.id
  if (label.trim().toLocaleLowerCase() === 'другая тема') return 'unknown'
  return label
}

function serviceIdForLabel(label?: string, services: TaxonomyOption[] = []) {
  if (!label) return undefined
  if (label.trim().toLocaleLowerCase() === 'другая служба') return 'service_other'
  return services.find((item) => item.id === label || item.label === label)?.id ?? label
}

function mapBackendTicket(item: BackendTicket, detail?: BackendTicketDetail, knownTickets: BackendTicket[] = [], preview?: BackendAssistPreview): Ticket {
  const prediction = detail?.prediction ?? preview?.prediction
  const latest = detail?.latest_decision
  const related = topRelatedCandidates(combineRelatedCandidates(preview?.similar_tickets, preview?.duplicate_candidates, preview?.repeat_candidates))
  const relationTopicId = latest?.confirmed_topic_id ?? item.topic_id
  const topic = latest
    ? latest.confirmed_topic_label ?? latest.confirmed_topic_id
    : prediction?.topic_label ?? 'Не определено'
  const alternatives = prediction?.alternatives ?? []
  const confidenceAvailable = Boolean(prediction && prediction.model_version !== 'unavailable')
  const confidenceState = normalizeConfidenceState(prediction?.confidence_state, prediction?.confidence ?? 0, confidenceAvailable)
  const text = item.text?.trim() || 'Текст обращения не предоставлен'
  const similar = related.map((candidate) => {
    const matchedFactors = mapRelatedFactors(candidate.matched_factors)
    return {
      id: candidate.ticket_id,
      title: knownTickets.find((ticket) => ticket.id === candidate.ticket_id)?.text ?? 'Связанное обращение',
      similarity: candidate.score,
      createdAt: candidate.created_at?.trim() || 'Дата не загружена',
      relation: mapRelation(candidate.relation),
      candidateTypes: candidate.candidateTypes,
      matchedFactors: matchedFactors.length
        ? matchedFactors
        : matchingTicketFactors(relationTopicId, item.region_id, candidate.topic_id, candidate.region_id),
      suggestion: candidate.suggestion ? {
        score: candidate.suggestion.score,
        threshold: candidate.suggestion.threshold,
        ruleVersion: candidate.suggestion.rule_version,
        modelVersion: candidate.suggestion.model_version,
        distanceMetric: candidate.suggestion.distance_metric,
      } : undefined,
    }
  })
  const status = latest?.action === 'correct' ? 'corrected' : latest?.action === 'confirm' || item.status === 'triaged' ? 'confirmed' : 'new'
  return {
    id: item.id,
    originalText: text,
    externalRef: item.external_ref?.trim() || undefined,
    modelVersion: prediction?.model_version === 'unavailable' ? undefined : prediction?.model_version,
    language: mapLanguage(preview?.orchestration?.language ?? item.language),
    topic,
    predictedTopic: prediction?.topic_label ?? 'Не определено',
    confidence: prediction?.confidence ?? 0,
    confidenceState,
    confidenceAvailable: confidenceAvailable && confidenceState !== 'UNAVAILABLE',
    alternatives: classificationAlternatives(confidenceState, prediction?.topic_id ?? '', alternatives).map((alternative) => ({ topic: alternative.topic_label, confidence: alternative.confidence })),
    service: latest?.service ?? (prediction?.recommended_service && !['UNKNOWN', 'unavailable'].includes(prediction.recommended_service) ? prediction.recommended_service : 'Не определена'),
    priority: mapPriority(latest?.priority ?? prediction?.predicted_priority ?? item.priority),
    routingReason: prediction?.routing_reason,
    serviceProvenance: mapRuleProvenance(
      latest?.service_provenance ?? prediction?.service_provenance,
      latest
        ? 'Источник подтверждённой службы не сохранён; проверьте решение вручную'
        : 'Источник рекомендованной службы не сохранён; проверьте вручную',
    ),
    priorityProvenance: mapRuleProvenance(
      latest?.priority_provenance ?? prediction?.priority_provenance,
      latest
        ? 'Источник подтверждённого приоритета не сохранён; проверьте решение вручную'
        : 'Источник рекомендованного приоритета не сохранён; проверьте вручную',
    ),
    region: item.region_name?.trim() || 'Регион не указан',
    createdAt: item.created_at?.trim() || 'Время не указано',
    status,
    similar,
    responseTemplate: preview?.response_template?.body ?? 'Шаблон ответа сейчас недоступен. Составьте ответ вручную.',
    responseTemplateApproved: preview?.response_template?.approved,
    responseTemplateSource: preview?.response_template?.source,
    assistPreview: preview?.orchestration,
    channel: mapTicketChannel(item.source),
  }
}

export async function loadRelatedTicketDetail(ticketId: string): Promise<RelatedTicketDetail> {
  const detail = await request<BackendTicketDetail>(`/tickets/${encodeURIComponent(ticketId)}`)
  const ticket = detail.ticket
  const decision = detail.latest_decision
  const decisionPriority = decision?.priority?.trim()
  const decisionService = decision?.service?.trim()

  return {
    id: ticket.id,
    externalRef: ticket.external_ref?.trim() || undefined,
    originalText: ticket.text?.trim() || 'Текст обращения не предоставлен',
    topic: ticket.topic_label?.trim() || 'Тема не указана',
    region: ticket.region_name?.trim() || 'Регион не указан',
    createdAt: ticket.created_at?.trim() || 'Время не указано',
    closedAt: ticket.closed_at?.trim() || undefined,
    status: ticket.status?.trim() || 'Статус не указан',
    channel: mapTicketChannel(ticket.source),
    latestDecision: decision ? {
      action: decision.action,
      confirmedTopicId: decision.confirmed_topic_id,
      service: decisionService && !['unknown', 'unavailable'].includes(decisionService.toLowerCase()) ? decisionService : undefined,
      priority: decisionPriority && !['unknown', 'unavailable'].includes(decisionPriority.toLowerCase()) ? mapPriority(decisionPriority) : undefined,
      createdAt: decision.created_at?.trim() || undefined,
    } : undefined,
  }
}

async function loadApiDashboard(filters: DashboardFilters): Promise<DashboardData> {
  const analyticsQuery = queryString(filters)
  const forecastQuery = new URLSearchParams({ horizon: '30', ...(filters.regionId ? { region_id: filters.regionId } : {}), ...(filters.topicId ? { topic_id: filters.topicId } : {}), ...(filters.serviceId ? { service_id: filters.serviceId } : {}), ...(filters.status ? { status: filters.status } : {}), ...(filters.district ? { district: filters.district } : {}), ...(filters.channel ? { channel: filters.channel } : {}) }).toString()
  const alertsQuery = filters.regionId ? `?region_id=${encodeURIComponent(filters.regionId)}` : ''
  const [ticketResponse, analytics, forecast, alertsResponse, learning, models, taxonomy, datasetProvenance] = await Promise.all([
    request<{ items: BackendTicket[] }>('/tickets?limit=50'),
    request<BackendAnalytics>(`/analytics?${analyticsQuery}`),
    request<BackendForecast>(`/forecast?${forecastQuery}`),
    request<BackendAlerts>(`/alerts${alertsQuery}`),
    request<BackendLearning>('/learning'),
    request<BackendModels>('/models'),
    request<BackendTaxonomy>('/taxonomy'),
    request<DatasetProvenance>('/datasets/provenance'),
  ])
  const detailResults = await Promise.all(ticketResponse.items.map((ticket) => request<BackendTicketDetail>(`/tickets/${encodeURIComponent(ticket.id)}`)))
  const previewResults = await Promise.all(ticketResponse.items.map((ticket) => request<BackendAssistPreview>('/assist/preview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ticket_id: ticket.id }) })))
  const tickets = ticketResponse.items.map((ticket, index) => mapBackendTicket(ticket, detailResults[index], ticketResponse.items, previewResults[index]))
  const regions: RegionMetric[] = analytics.by_region.filter((region) => region.tickets > 0).map((region) => ({ id: region.id, name: region.label, tickets: region.tickets, change: region.change_pct }))
  const topics: TopicMetric[] = analytics.by_topic.filter((topic) => topic.tickets > 0).map((topic, index) => ({ id: topic.id, name: topic.label, value: Math.round((topic.tickets / Math.max(1, analytics.overview.total_tickets)) * 100), change: topic.change_pct, color: ['#8cf0c8', '#a7d9ff', '#f8d488', '#d2b5ff', '#ff9d9d'][index % 5] }))
  const alerts: Alert[] = alertsResponse.items.map((alert) => ({ id: alert.id, title: alert.title, description: alert.description, severity: alert.severity.toLowerCase() === 'critical' ? 'critical' : alert.severity.toLowerCase() === 'high' ? 'watch' : 'info', region: alert.region_id, topic: alert.topic_id, detectedAt: alert.detected_at, affectedTickets: alert.ticket_count, status: alert.status.toLowerCase() === 'acknowledged' ? 'В работе' : alert.status.toLowerCase() === 'closed' ? 'Закрыт' : 'Новый' }))
  const forecastPoints: ForecastPoint[] = forecast.points.map((point) => ({ label: point.date, forecast: point.tickets }))
  const cycle = learning.active_cycle ?? learning.items?.[0]
  const learningData: LearningCycle = cycle ? { id: cycle.id, stage: mapLearningStage(cycle.state), dataset: cycle.dataset_version, feedbackCount: cycle.feedback_count, candidate: cycle.candidate_model_version, updatedAt: cycle.updated_at, decisionNote: cycle.decision_note } : { id: 'нет данных', stage: 'COLLECT', dataset: 'нет данных', feedbackCount: 0, candidate: 'нет данных', updatedAt: 'нет данных' }
  const modelData: ModelStatus[] = models.items.map((model) => {
    const status = model.status.toLowerCase()
    return {
      name: model.model_family,
      version: model.id,
      status: status === 'production' || status === 'candidate' || status === 'shadow' || status === 'rejected' || status === 'archived' ? status : 'shadow',
      metric: 'доступно в реестре',
      metricValue: model.metrics.macro_f1 == null ? '—' : model.metrics.macro_f1.toFixed(3),
      updatedAt: model.created_at,
    }
  })
  const runtime = analytics.runtime_metrics
  const operatorMetrics: OperatorRuntimeMetrics = {
    operatorDecisionTimeMinutes: runtime?.operator_decision_time_minutes
      ?? analytics.overview.avg_decision_minutes
      ?? null,
    operatorDecisionTimeSamples: runtime?.operator_decision_time_samples ?? 0,
    classificationCorrectionRate: runtime?.classification_correction_rate ?? null,
    classificationCorrections: runtime?.classification_corrections ?? 0,
    classificationDecisions: runtime?.classification_decisions ?? 0,
    routingCorrectionRate: runtime?.routing_correction_rate ?? null,
    routingCorrections: runtime?.routing_corrections ?? 0,
    routingDecisions: runtime?.routing_decisions ?? 0,
    priorityCorrectionRate: runtime?.priority_correction_rate ?? null,
    priorityCorrections: runtime?.priority_corrections ?? 0,
    priorityDecisions: runtime?.priority_decisions ?? 0,
    similarityUsefulness: runtime?.similarity_usefulness ?? null,
    similarityFeedbackCount: runtime?.similarity_feedback_count ?? 0,
    duplicatePrecision: runtime?.duplicate_precision ?? null,
    duplicateFeedbackCount: runtime?.duplicate_feedback_count ?? 0,
  }
  return {
    tickets,
    overview: {
      totalTickets: analytics.overview.total_tickets,
      openTickets: analytics.overview.open_tickets,
      resolvedTickets: analytics.overview.resolved_tickets,
      highPriorityTickets: analytics.overview.high_priority_tickets,
      operatorDecisions: analytics.overview.operator_decisions ?? 0,
      confirmedDecisions: analytics.overview.confirmed_decisions ?? 0,
      correctedDecisions: analytics.overview.corrected_decisions ?? 0,
      changeAbs: analytics.overview.change_abs ?? 0,
      avgDecisionMinutes: analytics.overview.avg_decision_minutes ?? undefined,
      changePct: analytics.overview.change_pct ?? undefined,
    },
    operatorMetrics,
    regions,
    topics,
    alerts,
    forecast: forecastPoints,
    models: modelData,
    learning: learningData,
    timeSeries: analytics.time_series,
    reportSource: 'postgres',
    forecastStatus: forecast.status,
    forecastModelVersion: forecast.model_version,
    datasetProvenance,
    filterOptions: {
      regions: taxonomy.regions ?? analytics.by_region.map((region) => ({ id: region.id, label: region.label })),
      topics: taxonomy.topics ?? [],
      services: taxonomy.services ?? [],
      statuses: taxonomy.statuses ?? [],
      districts: taxonomy.districts ?? [],
      channels: taxonomy.channels ?? [],
    },
  }
}

export async function loadDashboard(filters: DashboardFilters = { range: '7d' }): Promise<ApiResult<DashboardData>> {
  try {
    return { data: await loadApiDashboard(filters), source: 'api' }
  } catch (error) {
    if (!DEMO_ENABLED) throw error
    return {
      data: demoData,
      source: 'demo',
      error: error instanceof Error ? error.message : 'API недоступен',
    }
  }
}

export function subscribeToAlertChanges(onChange: () => void): () => void {
  const events = new EventSource(`${API_BASE}/events`, { withCredentials: true })
  events.addEventListener('alerts.changed', onChange)
  events.addEventListener('alerts.resync', onChange)
  return () => events.close()
}

export async function loadTickets(): Promise<ApiResult<Ticket[]>> {
  try {
    const response = await request<{ items: BackendTicket[] }>('/tickets?limit=50')
    const tickets = await Promise.all(
      response.items.map(async (item) => {
        const [detail, preview] = await Promise.all([
          request<BackendTicketDetail>(`/tickets/${encodeURIComponent(item.id)}`),
          request<BackendAssistPreview>('/assist/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticket_id: item.id }),
          }),
        ])
        return mapBackendTicket(item, detail, response.items, preview)
      }),
    )
    return { data: tickets, source: 'api' }
  } catch (error) {
    if (!DEMO_ENABLED) throw error
    return {
      data: demoData.tickets,
      source: 'demo',
      error: error instanceof Error ? error.message : 'API недоступен',
    }
  }
}

export async function submitDecision(ticketId: string, decision: { status: 'confirmed' | 'corrected'; topic?: string; service?: string; priority?: string }, taxonomy?: Pick<DashboardData['filterOptions'], 'topics' | 'services'>) {
  try {
    const action = decision.status === 'confirmed' ? 'confirm' : 'correct'
    const saved = await request<BackendDecisionResponse>(`/assist/${encodeURIComponent(ticketId)}/${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topic_id: topicIdForLabel(decision.topic, taxonomy?.topics),
        service: serviceIdForLabel(decision.service, taxonomy?.services),
        priority: decision.priority === 'Критический' ? 'critical' : decision.priority === 'Высокий' ? 'high' : decision.priority === 'Низкий' ? 'low' : decision.priority === 'Средний' ? 'medium' : undefined,
      }),
    })
    let preview: BackendAssistPreview | undefined
    try {
      preview = await request<BackendAssistPreview>('/assist/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticket_id: ticketId }),
      })
    } catch {
      return {
        source: 'api' as const,
        ticket: mapBackendTicket(saved.ticket, { ticket: saved.ticket, prediction: saved.prediction, latest_decision: saved.decision }, [saved.ticket]),
        warning: 'Решение сохранено, но шаблон ответа пока недоступен.',
      }
    }
    return {
      source: 'api' as const,
      ticket: mapBackendTicket(saved.ticket, { ticket: saved.ticket, prediction: saved.prediction, latest_decision: saved.decision }, [saved.ticket], preview),
    }
  } catch (error) {
    if (!DEMO_ENABLED) throw error
    return { source: 'demo' as const, error: error instanceof Error ? error.message : 'API недоступен' }
  }
}

export async function submitRelationFeedback(ticketId: string, relatedTicketId: string, relation: 'DUPLICATE' | 'REPEAT' | 'SIMILAR' | 'UNRELATED', decision: 'CONFIRMED' | 'REJECTED', suggestion?: RelationSuggestionSnapshot) {
  return request(`/tickets/${encodeURIComponent(ticketId)}/relation-feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      related_ticket_id: relatedTicketId,
      relation,
      decision,
      suggestion: suggestion ? {
        score: suggestion.score,
        threshold: suggestion.threshold,
        rule_version: suggestion.ruleVersion,
        model_version: suggestion.modelVersion,
        distance_metric: suggestion.distanceMetric,
      } : undefined,
    }),
  })
}

export async function acknowledgeAlert(alertId: string): Promise<Pick<BackendAlert, 'id' | 'status'>> {
  return request<Pick<BackendAlert, 'id' | 'status'>>(`/alerts/${encodeURIComponent(alertId)}/ack`, { method: 'POST' })
}

export async function closeAlert(alertId: string): Promise<Pick<BackendAlert, 'id' | 'status'>> {
  return request<Pick<BackendAlert, 'id' | 'status'>>(`/alerts/${encodeURIComponent(alertId)}/close`, { method: 'POST' })
}

export async function closeLearningCycle(cycleId: string) {
  return request<{ job_id?: string | null; state: string; cycle: BackendLearningCycle; production_model_unchanged: boolean }>('/learning/cycle/close', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cycle_id: cycleId }),
  })
}

export async function loadCandidateEvaluation() {
  return request<CandidateEvaluation>('/learning/candidate/evaluation')
}

export async function promoteCandidate(note?: string) {
  return request<BackendLearningCycle>('/learning/candidate/promote', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ note: note?.trim() || undefined }),
  })
}

export async function rejectCandidate(note?: string) {
  return request<BackendLearningCycle>('/learning/candidate/reject', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ note: note?.trim() || undefined }),
  })
}

export function reportUrl(format: 'pdf' | 'xlsx', filters: DashboardFilters = { range: '30d' }) {
  return `${API_BASE}/analytics/export.${format}?${queryString(filters)}`
}

export async function loadAnalyticsDrilldown(dimension: DrilldownDimension, value: string | undefined, filters: DashboardFilters = { range: '30d' }) {
  const params = new URLSearchParams(queryString(filters))
  params.set('dimension', dimension)
  if (value) params.set('value', value)
  return request<{ items: BackendTicket[]; total: number; limit: number; offset: number }>(`/analytics/drilldown?${params.toString()}`)
}

export interface QueryIntentResult {
  intent: string
  number?: number | string
  rows?: QueryIntentRow[]
  result?: { points?: QueryIntentRow[] }
  source: string
}

interface QueryIntentRow {
  period?: string
  date?: string
  label?: string
  count?: number
  tickets?: number
}

export async function runQueryIntent(text: string, filters: DashboardFilters = { range: '30d' }) {
  return request<QueryIntentResult>('/analytics/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: text.trim(),
      range: filters.range,
      region_id: filters.regionId,
      topic_id: filters.topicId,
      filters: {
        range: filters.range,
        region_id: filters.regionId,
        topic_id: filters.topicId,
        service_id: filters.serviceId,
        status: filters.status,
        district: filters.district,
        channel: filters.channel,
      },
      limit: 10,
    }),
  })
}
