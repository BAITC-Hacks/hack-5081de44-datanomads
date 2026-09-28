import { demoData } from '../data/demo'
import type { ActionableContext, Alert, AssistPreviewState, ConfirmedDecisionSummary, ContextHandoffPackage, DashboardData, DatasetProvenance, DriftTrigger, ForecastBacktest, ForecastCapacityAssessment, ForecastCapacityInput, ForecastManagerSignal, ForecastPoint, ForecastReforecast, LearningCycle, ModelStatus, OperatorRuntimeMetrics, OutcomeVerificationRecord, OutcomeVerificationSnapshot, OutcomeVerificationState, Priority, RegionMetric, RelatedTicketDetail, RelationSuggestionSnapshot, RoutingFeedbackRecord, SimilarTicket, Ticket, TopicMetric } from '../types'
import { mapLanguage } from '../language'
import { classificationAlternatives, normalizeConfidenceState } from '../classification'
import { mapPriority, mapRuleProvenance } from '../routing'
import { buildContextPreviewText, combineRelatedCandidates, mapRelatedFactors, mapTicketChannel, topRelatedCandidates } from '../operator'

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
  forecastHorizon?: 30 | 60 | 90
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

export interface AnalyticsDrilldownTicket {
  id: string
  region_id: string
  region_name: string
  topic_id: string
  topic_label: string
  priority: string
  status: string
  created_at: string
}

interface BackendAlternative {
  topic_id: string
  topic_label: string
  confidence: number
}

interface BackendRuleProvenance {
  source?: string
  version?: number | null
  reason?: string
  facts_used?: Array<{ field?: string; value?: string }>
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
  service_provenance?: BackendRuleProvenance
  priority_provenance?: BackendRuleProvenance
  alternatives: BackendAlternative[]
}

interface BackendActionableContext {
  status: ActionableContext['status']
  rule_id: string
  reason: string
  missing_fact?: 'topic'
  question?: string
  decision_critical_fields: ActionableContext['decisionCriticalFields']
  options: Array<{
    topic_id: string
    topic_label: string
    service: string
    priority: string
  }>
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
  ticket: BackendTicket
  prediction?: BackendPrediction
  actionable_context: BackendActionableContext
  similar_tickets?: BackendSimilar[]
  duplicate_candidates?: BackendSimilar[]
  repeat_candidates?: BackendSimilar[]
  response_template?: {
    id?: string
    template_key?: string
    topic_id?: string
    service_id?: string
    version?: number
    body: string
    approved?: boolean
    source?: string
  }
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
    service_provenance?: BackendRuleProvenance
    priority_provenance?: BackendRuleProvenance
    created_at?: string
  }
}

interface BackendRoutingFeedbackRecord {
  id: string
  ticket_id: string
  operator_decision_id: string
  original_route_recommendation: string
  operator_confirmed_route: string
  service_feedback: 'ACCEPTED' | 'CORRECTED'
  corrected_target_service: string | null
  actor_user_id: string
  source_system: 'DEMO_SIMULATION'
  evaluation_status: 'PENDING_OFFLINE_REVIEW'
  created_at: string
}

interface BackendRoutingFeedbackListResponse {
  items: BackendRoutingFeedbackRecord[]
}

interface BackendOutcomeVerificationRecord {
  id: string
  ticket_id: string
  state: Exclude<OutcomeVerificationState, 'UNKNOWN'>
  source_system: string
  channel: string
  actor_user_id: string
  created_at: string
}

interface BackendOutcomeVerificationSnapshot {
  ticket_id: string
  official_ticket_status: string
  state: OutcomeVerificationState
  latest: BackendOutcomeVerificationRecord | null
  history: BackendOutcomeVerificationRecord[]
}

interface BackendContextHandoffEvidenceReference {
  source_type: 'ticket' | 'operator_decision'
  record_id: string
  field: string
  label: string
}

interface BackendContextHandoffPackage {
  package_version: 'context-handoff.v1'
  ticket_id: string
  what_happened: string
  where: {
    region_id: string
    region_name: string
    district: string | null
    address: string | null
    object: string | null
  }
  when_or_since: { received_at: string; reported_since: string | null }
  scale: string | null
  confirmed_facts: Array<{
    label: string
    value: string
    evidence: BackendContextHandoffEvidenceReference
  }>
  unknown_facts: string[]
  route: {
    recommended_service: string | null
    confirmed_service: string
    operator_decision_id: string
    explanation: string
    provenance_source: 'OFFICIAL' | 'LABEL_HISTORY' | 'MANUAL'
    provenance_version: number | null
  }
  linked_attachment_count: number
  evidence_references: BackendContextHandoffEvidenceReference[]
}

interface BackendDecisionResponse {
  ticket: BackendTicket
  prediction: BackendPrediction
  decision: NonNullable<BackendTicketDetail['latest_decision']>
  learning_feedback_status: string
  learning_feedback_cycle_id: string | null
}

interface BackendAnalytics {
  source?: string
  overview: { total_tickets: number; open_tickets: number; resolved_tickets: number; high_priority_tickets: number; operator_decisions?: number; confirmed_decisions?: number; corrected_decisions?: number; avg_decision_minutes?: number | null; previous_total_tickets?: number; change_abs?: number; change_pct?: number | null }
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

interface BackendForecast {
  source?: string
  status?: string
  insufficient_history?: boolean
  horizon_days?: number
  history?: Array<{ date: string; tickets: number; resolved: number }>
  forecast_start?: string | null
  points: Array<{ date: string; tickets: number; resolved: number }>
  model_version: string
  model?: string
  expected_peaks?: string[]
  backtest?: ForecastBacktest
  capacity_assessment: {
    status: ForecastCapacityAssessment['status']
    missing_inputs: ForecastCapacityInput[]
  }
  run_id?: string
  issued_at?: string
  reforecast?: {
    previous_run_id: string
    previous_model_version: string
    previous_issued_at: string
    actual_observations: Array<{ date: string; previous_forecast: number; actual: number; absolute_error: number }>
    future_comparisons: Array<{ date: string; previous_forecast: number; updated_forecast: number; delta: number }>
    peak_change?: {
      previous_peak_date: string
      updated_peak_date: string
      previous_peak: number
      updated_peak: number
      delta: number
      policy_version: string
      threshold?: number | null
      status: string
      manager_signal_id?: string | null
    } | null
  } | null
  manager_signals?: Array<{
    id: string
    run_id: string
    previous_run_id: string
    created_at: string
    previous_peak: number
    updated_peak: number
    delta: number
    threshold: number
    policy_version: string
  }>
}
interface BackendAlertDetail {
  historical_counts?: number[]
  trigger_reasons?: string[]
  configuration?: { period_days?: number; robust_z_threshold?: number; ratio_threshold?: number }
}
interface BackendAlertMonitoring {
  state: 'MONITORING' | 'STABILIZED' | 'PERSISTING' | 'WORSENING' | 'RECURRED' | 'INSUFFICIENT_HISTORY'
  monitoring_period_days: number
  observation_period_days: number
  started_at: string
  ends_at: string
  started_by: string
  completed_at?: string | null
  evidence?: {
    source?: string
    reason?: string
    interpretation_scope?: string
    periods?: Array<{
      period_start: string
      period_end: string
      current_count: number
      baseline?: number
      deviation?: number
      robust_z?: number | null
      ratio?: number
      severity?: string | null
      signal_detected?: boolean
      source_ticket_ids: string[]
    }>
  } | null
}
interface BackendAlert {
  id: string
  incident_key: string
  severity: string
  status: string
  title: string
  description: string
  region_id: string
  topic_id: string
  ticket_count: number
  period_start?: string | null
  period_end?: string | null
  current_count?: number
  baseline?: number | null
  deviation?: number | null
  robust_z?: number | null
  ratio?: number | null
  detector_version?: string | null
  linked_ticket_ids?: string[]
  created_at?: string
  detected_at: string
  detail?: BackendAlertDetail
  monitoring?: BackendAlertMonitoring | null
}
interface BackendAlerts { source?: string; items: BackendAlert[]; total?: number }
interface BackendLearningCycle { id: string; cycle_id: string; state: string; dataset_version: string; candidate_model_version: string; collect_started_at: string; collect_ends_at: string; evaluation_started_at: string | null; evaluation_ends_at: string | null; shadow_prediction_count: number; shadow_inference_failures: number; shadow_operator_decision_count: number; blind_ab_enabled: boolean; production_model_version: string | null; frozen_evaluation_dataset_version: string | null; candidate_dataset_checksum: string | null; min_feedback_count: number; promotion_policy_version: string; manual_close_enabled: boolean; feedback_count: number; updated_at: string; decision_note?: string; metrics: { macro_f1?: number | null; accuracy?: number | null; evaluated_samples?: number } }
interface BackendLearning { source?: string; items?: BackendLearningCycle[]; active_cycle?: BackendLearningCycle; production_model?: { id: string; status: string }; controlled_loop?: Record<string, unknown> }
interface BackendModels { source?: string; items: Array<{ id: string; model_family: string; status: string; metrics: { macro_f1: number | null; accuracy: number | null }; created_at: string }> }
interface BackendDriftTriggers { items: DriftTrigger[]; total: number; limit: number; offset: number; can_review: boolean }
export interface CandidateEvaluation {
  schema_version: 'candidate-evaluation.v1'
  status: 'PENDING' | 'COMPLETED' | 'FAILED'
  cycle_id: string
  candidate_model_version: string
  production_model_version: string
  candidate_dataset_version: string
  evaluation_version: string
  policy_version: string
  promotion_policy: { version: string; thresholds: Record<string, number> }
  offline_evaluation: CandidateModelEvaluation
  baseline_evaluation: CandidateModelEvaluation
  shadow_evaluation: {
    sample_count: number
    agreement_with_confirmed: number | null
    correction_rate_delta: number | null
    critical_regressions: string[]
    metrics: Record<string, unknown>
    blind_ab: 'DISABLED' | 'ENABLED'
  }
  gates: Array<{
    key: string
    status: 'PASSED' | 'FAILED' | 'INSUFFICIENT_EVIDENCE' | 'PENDING'
    observed?: number | null
    threshold?: number | null
    reason?: string
  }>
  decision: 'PENDING_HUMAN_DECISION' | 'PASS' | 'FAIL' | 'INSUFFICIENT_EVIDENCE'
  evaluated_at: string
  synthetic: boolean
  candidate_comparisons?: CandidateEvaluationComparison[]
  evaluation_set?: CandidateEvaluationSet
}
export interface CandidateEvaluationComparison {
  candidate_model_version: string
  status: 'REGISTERED' | 'EVALUATING' | 'EVALUATED' | 'EVALUATION_FAILED' | 'PROMOTED' | 'REJECTED'
  candidate_dataset_version: string | null
  job_state: 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | null
  evaluation: CandidateEvaluation | null
}
export interface CandidateEvaluationSet {
  cycle_id: string
  dataset_version: string | null
  sample_count: number
  window_started_at: string | null
  window_ended_at: string | null
}
export interface RegisterLearningCandidateResponse {
  cycle_id: string
  candidate_model_version: string
  candidate_dataset_version: string
  frozen_evaluation_dataset_version: string
  status: 'REGISTERED'
  production_model_unchanged: true
}
export interface CandidateModelEvaluation {
  evaluation_id: string
  model_version: string
  dataset_version: string
  status: 'COMPLETED' | 'INSUFFICIENT_DATA' | 'FAILED'
  sample_count: number
  metrics: {
    macro_f1?: number | null
    accuracy?: number | null
    per_class_f1?: Record<string, number>
    production_macro_f1?: number | null
    production_accuracy?: number | null
    production_per_class_f1?: Record<string, number>
    per_class_changes?: Record<string, number>
    per_class_support?: Record<string, number>
    reason?: string
    [key: string]: unknown
  }
  critical_regressions: string[]
  synthetic: boolean
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
    case 'TRAINING_FAILED':
      return 'TRAINING_FAILED'
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
    case 'DATASET_BUILD_FAILED':
      return 'DATASET_BUILD_FAILED'
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
  const topic = latest
    ? latest.confirmed_topic_label ?? latest.confirmed_topic_id
    : prediction?.topic_label ?? 'Не определено'
  const alternatives = prediction?.alternatives ?? []
  const confidenceAvailable = Boolean(prediction && prediction.model_version !== 'unavailable')
  const confidenceState = normalizeConfidenceState(prediction?.confidence_state, prediction?.confidence ?? 0, confidenceAvailable)
  const text = item.text?.trim() || 'Текст обращения не предоставлен'
  const recommendedService = prediction?.recommended_service?.trim()
  const recommendedServiceIsKnown = Boolean(
    recommendedService && !['UNKNOWN', 'unavailable'].includes(recommendedService.toLowerCase()),
  )
  const recommendedServiceProvenance = prediction
    ? mapRuleProvenance(
      prediction.service_provenance,
      'Источник рекомендованной службы не сохранён; проверьте вручную',
    )
    : undefined
  const recommendedPriorityProvenance = prediction
    ? mapRuleProvenance(
      prediction.priority_provenance,
      'Источник рекомендованного приоритета не сохранён; проверьте вручную',
    )
    : undefined
  const similar = related.map((candidate) => {
    return {
      id: candidate.ticket_id,
      title: knownTickets.find((ticket) => ticket.id === candidate.ticket_id)?.text ?? 'Связанное обращение',
      similarity: candidate.score,
      createdAt: candidate.created_at?.trim() || 'Дата не загружена',
      relation: mapRelation(candidate.relation),
      candidateTypes: candidate.candidateTypes,
      matchedFactors: mapRelatedFactors(candidate.matched_factors),
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
    regionId: item.region_id?.trim() || undefined,
    modelVersion: prediction?.model_version === 'unavailable' ? undefined : prediction?.model_version,
    language: mapLanguage(preview?.orchestration?.language ?? item.language),
    topic,
    topicId: latest?.confirmed_topic_id ?? prediction?.topic_id ?? item.topic_id,
    predictedTopic: prediction?.topic_label ?? 'Не определено',
    predictedTopicId: prediction?.topic_id,
    confidence: prediction?.confidence ?? 0,
    confidenceState,
    confidenceAvailable: confidenceAvailable && confidenceState !== 'UNAVAILABLE',
    alternatives: classificationAlternatives(confidenceState, prediction?.topic_id ?? '', alternatives).map((alternative) => ({ topic: alternative.topic_label, confidence: alternative.confidence })),
    service: latest?.service ?? (prediction?.recommended_service && !['UNKNOWN', 'unavailable'].includes(prediction.recommended_service) ? prediction.recommended_service : 'Не определена'),
    priority: mapPriority(latest?.priority ?? prediction?.predicted_priority ?? item.priority),
    recommendedService: recommendedServiceIsKnown ? recommendedService : undefined,
    recommendedPriority: prediction ? mapPriority(prediction.predicted_priority) : undefined,
    recommendedServiceProvenance,
    recommendedPriorityProvenance,
    confirmedDecisionAvailable: Boolean(latest),
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
    responseTemplate: preview?.response_template?.source === 'UNAVAILABLE'
      ? preview.response_template.body
      : preview?.response_template?.source === 'APPROVED_TEMPLATE' && preview.response_template.approved === true
        ? preview.response_template.body
        : '',
    responseTemplateApproved: preview?.response_template?.approved,
    responseTemplateSource: preview?.response_template?.source,
    responseTemplateId: preview?.response_template?.id,
    responseTemplateKey: preview?.response_template?.template_key,
    responseTemplateVersion: preview?.response_template?.version,
    actionableContext: preview?.actionable_context ? {
      status: preview.actionable_context.status,
      ruleId: preview.actionable_context.rule_id,
      reason: preview.actionable_context.reason,
      missingFact: preview.actionable_context.missing_fact,
      question: preview.actionable_context.question,
      decisionCriticalFields: preview.actionable_context.decision_critical_fields,
      options: preview.actionable_context.options.map((option) => ({
        topicId: option.topic_id,
        topicLabel: option.topic_label,
        service: option.service,
        priority: mapPriority(option.priority),
      })),
    } : undefined,
    sourceStatus: item.status?.trim() || undefined,
    closedAt: item.closed_at === null ? null : item.closed_at?.trim() || undefined,
    latestDecision: latest ? mapConfirmedDecision(latest) : undefined,
    assistPreview: preview?.orchestration,
    channel: mapTicketChannel(item.source),
  }
}

function mapConfirmedDecision(decision: NonNullable<BackendTicketDetail['latest_decision']>): ConfirmedDecisionSummary {
  const service = decision.service?.trim()
  const priority = decision.priority?.trim()
  return {
    action: decision.action,
    confirmedTopicId: decision.confirmed_topic_id,
    confirmedTopicLabel: decision.confirmed_topic_label?.trim() || undefined,
    service: service && !['unknown', 'unavailable'].includes(service.toLowerCase()) ? service : undefined,
    priority: priority && !['unknown', 'unavailable'].includes(priority.toLowerCase()) ? mapPriority(priority) : undefined,
    createdAt: decision.created_at?.trim() || undefined,
  }
}

export async function loadRelatedTicketDetail(ticketId: string): Promise<RelatedTicketDetail> {
  const [detail, outcomeResult] = await Promise.all([
    request<BackendTicketDetail>(`/tickets/${encodeURIComponent(ticketId)}`),
    loadOutcomeVerification(ticketId)
      .then((snapshot) => ({ snapshot }))
      .catch((error: unknown) => ({ error: error instanceof Error ? error.message : 'ошибка API' })),
  ])
  const ticket = detail.ticket
  const decision = detail.latest_decision
  const decisionPriority = decision?.priority?.trim()
  const decisionService = decision?.service?.trim()

  return {
    id: ticket.id,
    externalRef: ticket.external_ref?.trim() || undefined,
    originalText: ticket.text?.trim() || 'Текст обращения не предоставлен',
    topic: decision?.confirmed_topic_label?.trim() || ticket.topic_label?.trim() || 'Тема не указана',
    topicId: decision?.confirmed_topic_id ?? ticket.topic_id,
    regionId: ticket.region_id?.trim() || undefined,
    region: ticket.region_name?.trim() || 'Регион не указан',
    createdAt: ticket.created_at?.trim() || 'Время не указано',
    closedAt: ticket.closed_at?.trim() || undefined,
    status: ticket.status?.trim() || 'Статус не указан',
    channel: mapTicketChannel(ticket.source),
    latestDecision: decision ? {
      action: decision.action,
      confirmedTopicId: decision.confirmed_topic_id,
      confirmedTopicLabel: decision.confirmed_topic_label?.trim() || undefined,
      service: decisionService && !['unknown', 'unavailable'].includes(decisionService.toLowerCase()) ? decisionService : undefined,
      priority: decisionPriority && !['unknown', 'unavailable'].includes(decisionPriority.toLowerCase()) ? mapPriority(decisionPriority) : undefined,
      createdAt: decision.created_at?.trim() || undefined,
    } : undefined,
    outcomeVerification: 'snapshot' in outcomeResult ? outcomeResult.snapshot : undefined,
    outcomeVerificationError: 'error' in outcomeResult ? outcomeResult.error : undefined,
  }
}

async function loadApiDashboard(filters: DashboardFilters): Promise<DashboardData> {
  const analyticsQuery = queryString(filters)
  const forecastRequest = {
    horizon: filters.forecastHorizon ?? 30,
    region_id: filters.regionId,
    topic_id: filters.topicId,
    service_id: filters.serviceId,
    status: filters.status,
    district: filters.district,
    channel: filters.channel,
  }
  const alertsQuery = filters.regionId ? `?region_id=${encodeURIComponent(filters.regionId)}` : ''
  const [ticketResponse, analytics, forecast, alertsResponse, learning, models, driftTriggers, taxonomy, datasetProvenance] = await Promise.all([
    request<{ items: BackendTicket[] }>('/tickets?limit=50'),
    request<BackendAnalytics>(`/analytics?${analyticsQuery}`),
    request<BackendForecast>('/forecast/reforecast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(forecastRequest),
    }),
    request<BackendAlerts>(`/alerts${alertsQuery}`),
    request<BackendLearning>('/learning'),
    request<BackendModels>('/models'),
    request<BackendDriftTriggers>('/learning/drift-triggers?limit=50'),
    request<BackendTaxonomy>('/taxonomy'),
    request<DatasetProvenance>('/datasets/provenance'),
  ])
  const detailResults = await Promise.all(ticketResponse.items.map((ticket) => request<BackendTicketDetail>(`/tickets/${encodeURIComponent(ticket.id)}`)))
  const previewResults = await Promise.all(ticketResponse.items.map((ticket) => request<BackendAssistPreview>('/assist/preview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ticket_id: ticket.id }) })))
  const tickets = ticketResponse.items.map((ticket, index) => mapBackendTicket(ticket, detailResults[index], ticketResponse.items, previewResults[index]))
  const regions: RegionMetric[] = analytics.by_region.map((region) => ({ id: region.id, name: region.label, tickets: region.tickets, previousTickets: region.change_abs == null ? undefined : Math.max(0, region.tickets - region.change_abs), changeAbs: region.change_abs, change: region.change_pct }))
  const topics: TopicMetric[] = analytics.by_topic.filter((topic) => topic.tickets > 0 || (topic.change_abs != null && topic.change_abs < 0)).map((topic, index) => ({ id: topic.id, name: topic.label, value: Math.round((topic.tickets / Math.max(1, analytics.overview.total_tickets)) * 100), tickets: topic.tickets, previousTickets: topic.change_abs == null ? undefined : Math.max(0, topic.tickets - topic.change_abs), changeAbs: topic.change_abs, change: topic.change_pct, color: ['#8cf0c8', '#a7d9ff', '#f8d488', '#d2b5ff', '#ff9d9d'][index % 5] }))
  const alerts: Alert[] = alertsResponse.items.map((alert) => {
    const region = taxonomy.regions.find((item) => item.id === alert.region_id)?.label ?? alert.region_id
    const topic = taxonomy.topics.find((item) => item.id === alert.topic_id)?.label ?? alert.topic_id
    const status = alert.status.toUpperCase()
    const configuration = alert.detail?.configuration
    const currentCount = alert.current_count ?? alert.ticket_count
    return {
      id: alert.id,
      incidentKey: alert.incident_key,
      title: alert.title,
      description: alert.description,
      severity: alert.severity.toLowerCase() === 'critical' ? 'critical' : alert.severity.toLowerCase() === 'high' ? 'watch' : 'info',
      region,
      topic,
      detectedAt: alert.detected_at,
      createdAt: alert.created_at ?? alert.detected_at,
      periodStart: alert.period_start ?? undefined,
      periodEnd: alert.period_end ?? undefined,
      affectedTickets: currentCount,
      currentCount,
      baseline: alert.baseline ?? undefined,
      deviation: alert.deviation ?? undefined,
      robustZ: alert.robust_z ?? undefined,
      ratio: alert.ratio ?? undefined,
      detectorVersion: alert.detector_version ?? undefined,
      linkedTicketIds: alert.linked_ticket_ids ?? [],
      historyCounts: alert.detail?.historical_counts,
      triggerReasons: alert.detail?.trigger_reasons,
      robustZThreshold: configuration?.robust_z_threshold,
      ratioThreshold: configuration?.ratio_threshold,
      periodDays: configuration?.period_days,
      monitoring: alert.monitoring ?? undefined,
      status: status === 'ACKNOWLEDGED' ? 'В работе' : status === 'CLOSED' ? 'Закрыт' : 'Новый',
    }
  })
  const forecastPoints: ForecastPoint[] = forecast.points.map((point) => ({ label: point.date, forecast: point.tickets }))
  const forecastHistory: ForecastPoint[] = (forecast.history ?? []).map((point) => ({ label: point.date, actual: point.tickets }))
  const forecastReforecast: ForecastReforecast | undefined = forecast.reforecast
    ? {
      previousRunId: forecast.reforecast.previous_run_id,
      previousModelVersion: forecast.reforecast.previous_model_version,
      previousIssuedAt: forecast.reforecast.previous_issued_at,
      actualObservations: forecast.reforecast.actual_observations.map((observation) => ({
        date: observation.date,
        previousForecast: observation.previous_forecast,
        actual: observation.actual,
        absoluteError: observation.absolute_error,
      })),
      futureComparisons: forecast.reforecast.future_comparisons.map((comparison) => ({
        date: comparison.date,
        previousForecast: comparison.previous_forecast,
        updatedForecast: comparison.updated_forecast,
        delta: comparison.delta,
      })),
      peakChange: forecast.reforecast.peak_change ? {
        previousPeakDate: forecast.reforecast.peak_change.previous_peak_date,
        updatedPeakDate: forecast.reforecast.peak_change.updated_peak_date,
        previousPeak: forecast.reforecast.peak_change.previous_peak,
        updatedPeak: forecast.reforecast.peak_change.updated_peak,
        delta: forecast.reforecast.peak_change.delta,
        policyVersion: forecast.reforecast.peak_change.policy_version,
        threshold: forecast.reforecast.peak_change.threshold,
        status: forecast.reforecast.peak_change.status,
        managerSignalId: forecast.reforecast.peak_change.manager_signal_id,
      } : undefined,
    }
    : undefined
  const forecastPreviousPoints: ForecastPoint[] = [
    ...(forecastReforecast?.actualObservations ?? []).map((observation) => ({ label: observation.date, forecast: observation.previousForecast })),
    ...(forecastReforecast?.futureComparisons ?? []).map((comparison) => ({ label: comparison.date, forecast: comparison.previousForecast })),
  ]
  const forecastManagerSignals: ForecastManagerSignal[] = (forecast.manager_signals ?? []).map((signal) => ({
    id: signal.id,
    runId: signal.run_id,
    previousRunId: signal.previous_run_id,
    createdAt: signal.created_at,
    previousPeak: signal.previous_peak,
    updatedPeak: signal.updated_peak,
    delta: signal.delta,
    threshold: signal.threshold,
    policyVersion: signal.policy_version,
  }))
  const cycle = learning.active_cycle ?? learning.items?.[0]
  const learningData: LearningCycle = cycle ? { id: cycle.id, stage: mapLearningStage(cycle.state), dataset: cycle.dataset_version, feedbackCount: cycle.feedback_count, candidate: cycle.candidate_model_version, collectStartedAt: cycle.collect_started_at, collectEndsAt: cycle.collect_ends_at, evaluationStartedAt: cycle.evaluation_started_at ?? undefined, evaluationEndsAt: cycle.evaluation_ends_at ?? undefined, shadowPredictionCount: cycle.shadow_prediction_count, shadowInferenceFailures: cycle.shadow_inference_failures, shadowOperatorDecisionCount: cycle.shadow_operator_decision_count, blindAbEnabled: cycle.blind_ab_enabled, productionModelVersion: cycle.production_model_version ?? undefined, frozenEvaluationDatasetVersion: cycle.frozen_evaluation_dataset_version ?? undefined, candidateDatasetChecksum: cycle.candidate_dataset_checksum ?? undefined, minFeedbackCount: cycle.min_feedback_count, promotionPolicyVersion: cycle.promotion_policy_version, manualCloseEnabled: cycle.manual_close_enabled, updatedAt: cycle.updated_at, decisionNote: cycle.decision_note } : { id: 'нет данных', stage: 'COLLECT', dataset: 'нет данных', feedbackCount: 0, candidate: 'нет данных', collectStartedAt: '', collectEndsAt: '', shadowPredictionCount: 0, shadowInferenceFailures: 0, shadowOperatorDecisionCount: 0, blindAbEnabled: false, minFeedbackCount: 0, promotionPolicyVersion: 'policy-v1', manualCloseEnabled: false, updatedAt: 'нет данных' }
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
      previousTotalTickets: analytics.overview.previous_total_tickets,
      avgDecisionMinutes: analytics.overview.avg_decision_minutes ?? undefined,
      changePct: analytics.overview.change_pct ?? undefined,
    },
    operatorMetrics,
    regions,
    topics,
    alerts,
    forecast: forecastPoints,
    models: modelData,
    driftTriggers: driftTriggers.items,
    canReviewDriftTriggers: driftTriggers.can_review,
    learning: learningData,
    timeSeries: analytics.time_series,
    reportSource: analytics.source ?? 'postgres',
    forecastStatus: forecast.status,
    forecastModelVersion: forecast.model_version,
    forecastModel: forecast.model,
    forecastSource: forecast.source,
    forecastInsufficientHistory: forecast.insufficient_history ?? forecast.status === 'INSUFFICIENT_HISTORY',
    forecastHorizonDays: forecast.horizon_days,
    forecastHistory,
    forecastStart: forecast.forecast_start ?? undefined,
    forecastExpectedPeaks: forecast.expected_peaks ?? [],
    forecastBacktest: forecast.backtest,
    forecastCapacityAssessment: {
      status: forecast.capacity_assessment.status,
      missingInputs: forecast.capacity_assessment.missing_inputs,
    },
    forecastRunId: forecast.run_id,
    forecastIssuedAt: forecast.issued_at,
    forecastReforecast,
    forecastPreviousPoints,
    forecastManagerSignals,
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

export async function previewTicketWithContext(ticket: Ticket, answer: string): Promise<Ticket> {
  const text = buildContextPreviewText(ticket.originalText, answer)
  const preview = await request<BackendAssistPreview>('/assist/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text,
      language: ticket.language,
      region_id: ticket.regionId,
    }),
  })
  return mapBackendTicket(preview.ticket, undefined, [preview.ticket], preview)
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
        learningFeedbackStatus: saved.learning_feedback_status,
        learningFeedbackCycleId: saved.learning_feedback_cycle_id ?? undefined,
        warning: 'Решение сохранено, но шаблон ответа пока недоступен.',
      }
    }
    return {
      source: 'api' as const,
      ticket: mapBackendTicket(saved.ticket, { ticket: saved.ticket, prediction: saved.prediction, latest_decision: saved.decision }, [saved.ticket], preview),
      learningFeedbackStatus: saved.learning_feedback_status,
      learningFeedbackCycleId: saved.learning_feedback_cycle_id ?? undefined,
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

function mapRoutingFeedback(item: BackendRoutingFeedbackRecord): RoutingFeedbackRecord {
  return {
    id: item.id,
    ticketId: item.ticket_id,
    operatorDecisionId: item.operator_decision_id,
    originalRouteRecommendation: item.original_route_recommendation,
    operatorConfirmedRoute: item.operator_confirmed_route,
    serviceFeedback: item.service_feedback,
    correctedTargetService: item.corrected_target_service?.trim() || undefined,
    actorUserId: item.actor_user_id,
    sourceSystem: item.source_system,
    evaluationStatus: item.evaluation_status,
    createdAt: item.created_at,
  }
}

function mapOutcomeVerificationRecord(item: BackendOutcomeVerificationRecord): OutcomeVerificationRecord {
  return {
    id: item.id,
    ticketId: item.ticket_id,
    state: item.state,
    sourceSystem: item.source_system,
    channel: item.channel,
    actorUserId: item.actor_user_id,
    createdAt: item.created_at,
  }
}

function mapOutcomeVerification(item: BackendOutcomeVerificationSnapshot): OutcomeVerificationSnapshot {
  return {
    ticketId: item.ticket_id,
    officialTicketStatus: item.official_ticket_status,
    state: item.state,
    latest: item.latest ? mapOutcomeVerificationRecord(item.latest) : undefined,
    history: item.history.map(mapOutcomeVerificationRecord),
  }
}

export async function loadRoutingFeedback(ticketId: string): Promise<RoutingFeedbackRecord[]> {
  const response = await request<BackendRoutingFeedbackListResponse>(
    `/tickets/${encodeURIComponent(ticketId)}/routing-feedback`,
  )
  return response.items.map(mapRoutingFeedback)
}

export async function submitRoutingFeedback(
  ticketId: string,
  serviceFeedback: 'ACCEPTED' | 'CORRECTED',
  correctedTargetService?: string,
): Promise<RoutingFeedbackRecord> {
  const response = await request<BackendRoutingFeedbackRecord>(
    `/tickets/${encodeURIComponent(ticketId)}/routing-feedback`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        service_feedback: serviceFeedback,
        ...(serviceFeedback === 'CORRECTED' ? { corrected_target_service: correctedTargetService } : {}),
      }),
    },
  )
  return mapRoutingFeedback(response)
}

export async function loadOutcomeVerification(ticketId: string): Promise<OutcomeVerificationSnapshot> {
  const response = await request<BackendOutcomeVerificationSnapshot>(
    `/tickets/${encodeURIComponent(ticketId)}/outcome-verification`,
  )
  return mapOutcomeVerification(response)
}

export async function submitOutcomeVerification(
  ticketId: string,
  state: Exclude<OutcomeVerificationState, 'UNKNOWN'>,
): Promise<OutcomeVerificationSnapshot> {
  const response = await request<BackendOutcomeVerificationSnapshot>(
    `/tickets/${encodeURIComponent(ticketId)}/outcome-verification`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ state }),
    },
  )
  return mapOutcomeVerification(response)
}

export async function loadContextHandoffPackage(ticketId: string): Promise<ContextHandoffPackage> {
  const item = await request<BackendContextHandoffPackage>(
    `/tickets/${encodeURIComponent(ticketId)}/handoff-package`,
  )
  const optionalValue = (value: string | null | undefined) => value?.trim() || undefined
  return {
    packageVersion: item.package_version,
    ticketId: item.ticket_id,
    whatHappened: item.what_happened,
    where: {
      regionId: item.where.region_id,
      regionName: item.where.region_name,
      district: optionalValue(item.where.district),
      address: optionalValue(item.where.address),
      object: optionalValue(item.where.object),
    },
    whenOrSince: {
      receivedAt: item.when_or_since.received_at,
      reportedSince: optionalValue(item.when_or_since.reported_since),
    },
    scale: optionalValue(item.scale),
    confirmedFacts: item.confirmed_facts.map((fact) => ({
      label: fact.label,
      value: fact.value,
      evidence: {
        sourceType: fact.evidence.source_type,
        recordId: fact.evidence.record_id,
        field: fact.evidence.field,
        label: fact.evidence.label,
      },
    })),
    unknownFacts: item.unknown_facts,
    route: {
      recommendedService: optionalValue(item.route.recommended_service),
      confirmedService: item.route.confirmed_service,
      operatorDecisionId: item.route.operator_decision_id,
      explanation: item.route.explanation,
      provenanceSource: item.route.provenance_source,
      provenanceVersion: item.route.provenance_version ?? undefined,
    },
    linkedAttachmentCount: item.linked_attachment_count,
    evidenceReferences: item.evidence_references.map((reference) => ({
      sourceType: reference.source_type,
      recordId: reference.record_id,
      field: reference.field,
      label: reference.label,
    })),
  }
}

export async function acknowledgeAlert(alertId: string): Promise<Pick<BackendAlert, 'id' | 'status'>> {
  return request<Pick<BackendAlert, 'id' | 'status'>>(`/alerts/${encodeURIComponent(alertId)}/ack`, { method: 'POST' })
}

export async function closeAlert(alertId: string): Promise<Pick<BackendAlert, 'id' | 'status'>> {
  return request<Pick<BackendAlert, 'id' | 'status'>>(`/alerts/${encodeURIComponent(alertId)}/close`, { method: 'POST' })
}

export async function startAlertMonitoring(alertId: string, monitoringPeriodDays: number): Promise<void> {
  await request<BackendAlert>(`/alerts/${encodeURIComponent(alertId)}/monitor`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ monitoring_period_days: monitoringPeriodDays }),
  })
}

export async function closeLearningCycle(cycleId: string) {
  return request<{ job_id?: string | null; state: string; cycle: BackendLearningCycle; production_model_unchanged: boolean }>('/learning/cycle/close', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cycle_id: cycleId }),
  })
}

export async function createLearningCycle(): Promise<BackendLearningCycle> {
  return request<BackendLearningCycle>('/learning', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  })
}

export async function loadCandidateEvaluation() {
  return request<CandidateEvaluation>('/learning/candidate/evaluation')
}

export async function registerLearningCandidate(cycleId: string, candidateModelVersion: string) {
  return request<RegisterLearningCandidateResponse>(`/learning/${encodeURIComponent(cycleId)}/candidates`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ candidate_model_version: candidateModelVersion.trim() }),
  })
}

export async function promoteCandidate(note?: string, candidateModelVersion?: string) {
  return request<BackendLearningCycle>('/learning/candidate/promote', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      note: note?.trim() || undefined,
      candidate_model_version: candidateModelVersion?.trim() || undefined,
    }),
  })
}

export async function rejectCandidate(note?: string) {
  return request<BackendLearningCycle>('/learning/candidate/reject', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ note: note?.trim() || undefined }),
  })
}

export async function reviewDriftTrigger(evidenceId: string, decision: 'OPEN_CANDIDATE_CYCLE' | 'DISMISS') {
  return request<DriftTrigger>(`/learning/drift-triggers/${encodeURIComponent(evidenceId)}/review`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ decision }),
  })
}

export function reportUrl(format: 'pdf' | 'xlsx', filters: DashboardFilters = { range: '30d' }) {
  return `${API_BASE}/analytics/export.${format}?${queryString(filters)}`
}

export async function loadAnalyticsDrilldown(dimension: DrilldownDimension, value: string | undefined, filters: DashboardFilters = { range: '30d' }) {
  const params = new URLSearchParams(queryString(filters))
  params.set('dimension', dimension)
  if (value) params.set('value', value)
  return request<{ items: AnalyticsDrilldownTicket[]; total: number; limit: number; offset: number }>(`/analytics/drilldown?${params.toString()}`)
}

export interface AuditLogEvent {
  id: number
  actor_id: string | null
  action: string
  entity_type: string
  entity_id: string | null
  request_id: string | null
  created_at: string
}

export interface AuditLogPage {
  items: AuditLogEvent[]
  total: number
  limit: number
  offset: number
}

export async function loadAuditLog(limit = 50, offset = 0) {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) })
  return request<AuditLogPage>(`/audit?${params.toString()}`)
}

export interface QueryIntentResult {
  intent: 'count' | 'trend' | 'compare_regions' | 'top_topics' | 'spikes' | 'forecast'
  summary: { label: string; value: number | null; text: string }
  number: number | null
  rows: QueryIntentRow[]
  table: QueryIntentRow[]
  table_columns: Array<{ key: string; label: string }>
  series: QueryIntentSeriesPoint[]
  chart: { type: 'line' | 'bar'; x: string; y: string; title: string; forecast_start: string | null }
  filters: QueryIntentFilterValues
  interpreted_filters: QueryIntentFilterValues & { group_by: string; limit: number | null; horizon_days: number | null }
  period: { range: string; days: number; start: string; end: string }
  grouping: string
  comparison: {
    type: string
    current_total: number
    previous_total: number
    change_abs: number
    change_pct: number | null
    current_start: string
    current_end: string
    previous_start: string
    previous_end: string
  }
  comparison_definition: string
  source: string
  generated_at: string
  forecast_status: string | null
  forecast_insufficient_history: boolean | null
  forecast_start: string | null
  forecast_model_version: string | null
  forecast_model: string | null
  forecast_horizon_days: number | null
  forecast_history: QueryIntentRow[] | null
  forecast_points: QueryIntentRow[] | null
  expected_peaks: string[] | null
  backtest: Record<string, unknown> | null
}

export interface QueryIntentRow {
  key?: string
  label?: string
  period?: string
  date?: string
  count?: number
  tickets?: number
  resolved?: number
  change_abs?: number | null
  change_pct?: number | null
  baseline?: number
  deviation?: number
  is_spike?: boolean
  segment?: 'history' | 'forecast'
  [key: string]: string | number | boolean | null | undefined
}

export interface QueryIntentSeriesPoint extends QueryIntentRow {
  date?: string
  label?: string
  count?: number
  segment?: 'history' | 'forecast'
}

export interface QueryIntentFilterValues {
  region_id?: string | null
  topic_id?: string | null
  service_id?: string | null
  status?: string | null
  district?: string | null
  channel?: string | null
  range?: string
  group_by?: string
}

export async function runQueryIntent(text: string, filters: DashboardFilters = { range: '30d' }) {
  try {
    return await request<QueryIntentResult>('/analytics/query', {
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
        limit: 100,
        ...(/прогноз|forecast/i.test(text) ? { horizon_days: filters.forecastHorizon ?? 30 } : {}),
      }),
    })
  } catch (error: unknown) {
    if (error instanceof Error && (error.message === 'API 400' || error.message === 'API 422')) {
      throw new Error('Запрос не распознан или фильтр недоступен. Спросите о количестве, динамике, регионах, темах, всплесках или прогнозе.')
    }
    throw error
  }
}
