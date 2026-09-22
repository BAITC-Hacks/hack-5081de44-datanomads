export type TicketStatus = 'new' | 'confirmed' | 'corrected'
export type Priority = 'Высокий' | 'Средний' | 'Низкий'

export interface TicketAlternative {
  topic: string
  confidence: number
}

export interface SimilarTicket {
  id: string
  title: string
  similarity: number
  createdAt: string
  relation: 'Похожий' | 'Повтор' | 'Дубликат'
}

export interface Ticket {
  id: string
  originalText: string
  modelVersion?: string
  language: 'RU' | 'KZ'
  topic: string
  confidence: number
  alternatives: TicketAlternative[]
  service: string
  priority: Priority
  routingReason?: string
  region: string
  createdAt: string
  status: TicketStatus
  similar: SimilarTicket[]
  responseTemplate: string
  responseTemplateApproved?: boolean
  responseTemplateSource?: string
  channel: 'eGov' | 'Call-центр' | 'Мобильное приложение' | 'WhatsApp'
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
