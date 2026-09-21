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
  language: 'RU' | 'KZ'
  topic: string
  confidence: number
  alternatives: TicketAlternative[]
  service: string
  priority: Priority
  region: string
  createdAt: string
  status: TicketStatus
  similar: SimilarTicket[]
  responseTemplate: string
  channel: 'eGov' | 'Call-центр' | 'Мобильное приложение' | 'WhatsApp'
}

export interface RegionMetric {
  name: string
  tickets: number
  change: number
  risk: 'stable' | 'watch' | 'critical'
}

export interface TopicMetric {
  name: string
  value: number
  change: number
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
  status: 'production' | 'candidate' | 'shadow'
  metric: string
  metricValue: string
  updatedAt: string
}

export interface LearningCycle {
  id: string
  stage: 'COLLECT' | 'TRAIN' | 'EVALUATE' | 'REVIEW'
  dataset: string
  feedbackCount: number
  candidate: string
  updatedAt: string
}

export interface DashboardData {
  tickets: Ticket[]
  regions: RegionMetric[]
  topics: TopicMetric[]
  alerts: Alert[]
  forecast: ForecastPoint[]
  models: ModelStatus[]
  learning: LearningCycle
}

export type ApiSource = 'api' | 'demo'

