export type TicketStatus = 'new' | 'confirmed' | 'corrected'
export type Priority = 'Критический' | 'Высокий' | 'Средний' | 'Низкий' | 'Не определён'
export type RuleSource = 'OFFICIAL' | 'LABEL_HISTORY' | 'MANUAL'
export interface RuleProvenance {
  source: RuleSource
  version?: number | null
  reason: string
}
export type PreviewLanguage = 'RU' | 'KZ' | 'MIXED' | 'UNKNOWN'
export type ClassificationConfidenceState = 'CONFIDENT' | 'UNCERTAIN' | 'LOW_CONFIDENCE' | 'UNAVAILABLE'

export interface AssistPreviewStage {
  name: string
  status: 'completed' | 'partial' | 'unavailable' | 'skipped' | 'unknown'
  latency_ms: number
  model_version?: string
  error_code?: string
}

export interface AssistPreviewState {
  request_id: string
  trace_id: string
  status: 'complete' | 'partial'
  needs_review: boolean
  language: PreviewLanguage
  latency_ms: number
  model_versions: Record<string, string>
  stages: AssistPreviewStage[]
}

export interface TicketAlternative {
  topic: string
  confidence: number
}

export interface RelationSuggestionSnapshot {
  score: number
  threshold: number
  ruleVersion: string
  modelVersion: string
  distanceMetric: string
}

export interface SimilarTicket {
  id: string
  title: string
  similarity: number
  createdAt: string
  relation: 'Похожий' | 'Возможное повторное обращение' | 'Возможный дубликат'
  candidateTypes?: Array<'similar' | 'duplicate' | 'repeat'>
  matchedFactors?: string[]
  suggestion?: RelationSuggestionSnapshot
}

export interface RelatedTicketDetail {
  id: string
  externalRef?: string
  originalText: string
  topic: string
  region: string
  createdAt: string
  closedAt?: string
  status: string
  channel: Ticket['channel']
  latestDecision?: {
    action: string
    confirmedTopicId: string
    service?: string
    priority?: Priority
    createdAt?: string
  }
}

export interface Ticket {
  id: string
  originalText: string
  externalRef?: string
  modelVersion?: string
  language: PreviewLanguage
  topic: string
  predictedTopic?: string
  confidence: number
  confidenceState?: ClassificationConfidenceState
  confidenceAvailable?: boolean
  alternatives: TicketAlternative[]
  service: string
  priority: Priority
  routingReason?: string
  serviceProvenance?: RuleProvenance
  priorityProvenance?: RuleProvenance
  region: string
  createdAt: string
  status: TicketStatus
  similar: SimilarTicket[]
  responseTemplate: string
  responseTemplateApproved?: boolean
  responseTemplateSource?: string
  assistPreview?: AssistPreviewState
  channel: 'eGov' | 'Call-центр' | 'Мобильное приложение' | 'WhatsApp' | 'Не указан'
}

export interface RegionMetric {
  id?: string
  name: string
  tickets: number
  change?: number
  risk?: 'stable' | 'watch' | 'critical'
}

export interface TopicMetric {
  id?: string
  name: string
  value: number
  change?: number
  color: string
}

export interface Alert {
  id: string
  title: string
  description: string
  severity: 'critical' | 'watch' | 'info'
  region: string
  topic: string
  detectedAt: string
  affectedTickets: number
  status: 'Новый' | 'В работе' | 'Закрыт'
}

export interface ForecastPoint {
  label: string
  actual?: number
  forecast?: number
  low?: number
  high?: number
}

export interface ModelStatus {
  name: string
  version: string
  status: 'production' | 'candidate' | 'shadow' | 'rejected' | 'archived'
  metric: string
  metricValue: string
  updatedAt: string
}

export interface LearningCycle {
  id: string
  stage: 'COLLECT' | 'TRAIN' | 'EVALUATE' | 'DECISION' | 'PROMOTED' | 'REJECTED' | 'INSUFFICIENT_FEEDBACK'
  dataset: string
  feedbackCount: number
  candidate: string
  updatedAt: string
  decisionNote?: string
}

export interface DatasetProvenance {
  synthetic_ticket_count: number
  real_ticket_count: number
  unassigned_ticket_count: number
  quarantined_row_count: number
  dataset_version_count: number
}

export interface OperatorRuntimeMetrics {
  operatorDecisionTimeMinutes: number | null
  operatorDecisionTimeSamples: number
  classificationCorrectionRate: number | null
  classificationCorrections: number
  classificationDecisions: number
  routingCorrectionRate: number | null
  routingCorrections: number
  routingDecisions: number
  priorityCorrectionRate: number | null
  priorityCorrections: number
  priorityDecisions: number
  similarityUsefulness: number | null
  similarityFeedbackCount: number
  duplicatePrecision: number | null
  duplicateFeedbackCount: number
}

export interface DashboardData {
  tickets: Ticket[]
  overview: {
    totalTickets: number
    openTickets: number
    resolvedTickets: number
    highPriorityTickets: number
    operatorDecisions: number
    confirmedDecisions: number
    correctedDecisions: number
    changeAbs: number
    avgDecisionMinutes?: number
    changePct?: number
  }
  operatorMetrics: OperatorRuntimeMetrics
  regions: RegionMetric[]
  topics: TopicMetric[]
  alerts: Alert[]
  forecast: ForecastPoint[]
  models: ModelStatus[]
  learning: LearningCycle
  timeSeries: Array<{ date: string; tickets: number; resolved: number }>
  reportSource: string
  forecastStatus?: string
  forecastModelVersion?: string
  datasetProvenance?: DatasetProvenance
  filterOptions: {
    regions: Array<{ id: string; label: string }>
    topics: Array<{ id: string; label: string }>
    services: Array<{ id: string; label: string }>
    statuses: Array<{ id: string; label: string }>
    districts: Array<{ id: string; label: string }>
    channels: Array<{ id: string; label: string }>
  }
}

export type ApiSource = 'api' | 'demo'
