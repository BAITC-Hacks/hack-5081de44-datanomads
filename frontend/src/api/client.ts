import { demoData } from '../data/demo'
import type { Alert, DashboardData, ForecastPoint, LearningCycle, ModelStatus, Priority, RegionMetric, SimilarTicket, Ticket, TopicMetric } from '../types'

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '/api/v1').replace(/\/$/, '')
const API_ROLE = import.meta.env.VITE_PULSE_ROLE ?? 'ADMIN'

export interface ApiResult<T> {
  data: T
  source: 'api' | 'demo'
  error?: string
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

interface BackendTicket {
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
  alternatives: BackendAlternative[]
}

interface BackendSimilar {
  ticket_id: string
  score: number
  relation: string
}

interface BackendAssistPreview {
  similar_tickets: BackendSimilar[]
  response_template?: { body: string }
}

interface BackendTicketDetail {
  ticket: BackendTicket
  prediction: BackendPrediction
  latest_decision?: { action: string; confirmed_topic_id: string; service: string; priority: string }
}

interface BackendAnalytics {
  source?: string
  overview: { total_tickets: number; open_tickets: number; resolved_tickets: number; high_priority_tickets: number }
  by_region: Array<{ id: string; label: string; tickets: number; high_priority: number; avg_confidence: number }>
  by_topic: Array<{ id: string; label: string; tickets: number; high_priority: number; avg_confidence: number }>
  time_series: Array<{ date: string; tickets: number; resolved: number }>
}

interface BackendForecast { source?: string; points: Array<{ date: string; tickets: number; resolved: number }>; model: string }
interface BackendAlert { id: string; severity: string; status: string; title: string; description: string; region_id: string; topic_id: string; ticket_count: number; detected_at: string }
interface BackendAlerts { source?: string; items: BackendAlert[] }
interface BackendLearning { source?: string; active_cycle?: { id: string; state: string; dataset_version: string; candidate_model_version: string; feedback_count: number; updated_at: string; metrics: { macro_f1: number } } }
interface BackendModels { source?: string; items: Array<{ id: string; model_family: string; status: string; metrics: { macro_f1: number | null; accuracy: number | null }; created_at: string }> }

function mapPriority(value: string): Priority {
  if (value === 'high' || value === 'Высокий') return 'Высокий'
  if (value === 'low' || value === 'Низкий') return 'Низкий'
  return 'Средний'
}

function mapLanguage(value: string): 'RU' | 'KZ' {
  return value.toLowerCase() === 'kk' || value.toLowerCase() === 'kz' ? 'KZ' : 'RU'
}

function mapRelation(value: string): SimilarTicket['relation'] {
  if (value === 'duplicate' || value === 'Дубликат') return 'Дубликат'
  if (value === 'repeat' || value === 'Повтор') return 'Повтор'
  return 'Похожий'
}

const topicIds: Record<string, string> = {
  'Водоснабжение': 'TOPIC-WATER',
  'Сумен жабдықтау': 'TOPIC-WATER',
  'Дороги и благоустройство': 'TOPIC-ROADS',
  'Здравоохранение': 'TOPIC-HEALTH',
  'Социальная поддержка': 'TOPIC-SOCIAL',
  'Социальная помощь': 'TOPIC-SOCIAL',
  'Образование': 'TOPIC-EDUCATION',
  'Коммунальные услуги': 'TOPIC-UTILITIES',
  'ЖКХ и инфраструктура': 'TOPIC-UTILITIES',
  'Безопасность': 'TOPIC-SAFETY',
  'Безопасность и освещение': 'TOPIC-SAFETY',
  'Общественный транспорт': 'TOPIC-TRANSPORT',
  'Экология': 'TOPIC-ENVIRONMENT',
  'Санитарлық жағдай': 'TOPIC-ENVIRONMENT',
  'Государственные сервисы': 'TOPIC-DIGITAL',
  'Цифровые услуги': 'TOPIC-DIGITAL',
  'Жильё': 'TOPIC-HOUSING',
}

function topicIdForLabel(label?: string) {
  if (!label) return undefined
  return label.startsWith('TOPIC-') ? label : topicIds[label] ?? 'TOPIC-OTHER'
}

function mapBackendTicket(item: BackendTicket, detail?: BackendTicketDetail, knownTickets: BackendTicket[] = [], preview?: BackendAssistPreview): Ticket {
  const prediction = detail?.prediction
  const latest = detail?.latest_decision
  const related = preview?.similar_tickets ?? []
  const topic = prediction?.topic_label ?? item.topic_label
  const alternatives = prediction?.alternatives ?? []
  const text = item.text
  const similar = related.map((candidate) => ({
    id: candidate.ticket_id,
    title: knownTickets.find((ticket) => ticket.id === candidate.ticket_id)?.text ?? 'Связанное обращение',
    similarity: candidate.score,
    createdAt: 'недавно',
    relation: mapRelation(candidate.relation),
  }))
  const status = latest?.action === 'correct' ? 'corrected' : latest?.action === 'confirm' || item.status === 'triaged' ? 'confirmed' : 'new'
  return {
    id: item.id,
    originalText: text,
    language: mapLanguage(item.language),
    topic,
    confidence: prediction?.confidence ?? .75,
    alternatives: alternatives.map((alternative) => ({ topic: alternative.topic_label, confidence: alternative.confidence })),
    service: latest?.service ?? prediction?.recommended_service ?? 'Служба обработки обращений',
    priority: mapPriority(latest?.priority ?? prediction?.predicted_priority ?? item.priority),
    region: item.region_name,
    createdAt: item.created_at,
    status,
    similar,
    responseTemplate: preview?.response_template?.body ?? 'Обращение зарегистрировано и передано ответственному подразделению. О статусе сообщим дополнительно.',
    channel: item.source === 'mobile' ? 'Мобильное приложение' : item.source === 'call-center' ? 'Call-центр' : item.source === 'whatsapp' ? 'WhatsApp' : 'eGov',
  }
}

async function loadApiDashboard(): Promise<DashboardData> {
  const [ticketResponse, analytics, forecast, alertsResponse, learning, models] = await Promise.all([
    request<{ items: BackendTicket[] }>('/tickets?limit=50'),
    request<BackendAnalytics>('/analytics?range=7d'),
    request<BackendForecast>('/forecast'),
    request<BackendAlerts>('/alerts'),
    request<BackendLearning>('/learning'),
    request<BackendModels>('/models'),
  ])
  const detailResults = await Promise.all(ticketResponse.items.map((ticket) => request<BackendTicketDetail>(`/tickets/${encodeURIComponent(ticket.id)}`).catch(() => undefined)))
  const previewResults = await Promise.all(ticketResponse.items.map((ticket) => request<BackendAssistPreview>('/assist/preview', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ticket_id: ticket.id }) }).catch(() => undefined)))
  const tickets = ticketResponse.items.map((ticket, index) => mapBackendTicket(ticket, detailResults[index], ticketResponse.items, previewResults[index]))
  const regions: RegionMetric[] = analytics.source === 'postgres' ? analytics.by_region.filter((region) => region.tickets > 0).map((region) => ({ name: region.label, tickets: region.tickets })) : []
  const topics: TopicMetric[] = analytics.source === 'postgres' ? analytics.by_topic.filter((topic) => topic.tickets > 0).map((topic, index) => ({ name: topic.label, value: Math.round((topic.tickets / Math.max(1, analytics.overview.total_tickets)) * 100), color: ['#8cf0c8', '#a7d9ff', '#f8d488', '#d2b5ff', '#ff9d9d'][index % 5] })) : []
  const alerts: Alert[] = alertsResponse.source === 'postgres' ? alertsResponse.items.map((alert) => ({ id: alert.id, title: alert.title, description: alert.description, severity: alert.severity === 'critical' ? 'critical' : alert.severity === 'warning' ? 'watch' : 'info', region: alert.region_id, topic: alert.topic_id, detectedAt: alert.detected_at, affectedTickets: alert.ticket_count, status: alert.status === 'acknowledged' ? 'В работе' : alert.status === 'closed' ? 'Закрыт' : 'Новый' })) : []
  const forecastPoints: ForecastPoint[] = forecast.source === 'postgres' ? forecast.points.map((point) => ({ label: point.date.slice(5), actual: point.tickets })) : []
  const cycle = learning.active_cycle
  const learningData: LearningCycle = learning && learning.source === 'postgres' && cycle ? { id: cycle.id, stage: cycle.state === 'REVIEW' ? 'REVIEW' : cycle.state === 'EVALUATE' ? 'EVALUATE' : cycle.state === 'TRAIN' ? 'TRAIN' : 'COLLECT', dataset: cycle.dataset_version, feedbackCount: cycle.feedback_count, candidate: cycle.candidate_model_version, updatedAt: cycle.updated_at } : demoData.learning
  const modelData: ModelStatus[] = models.source === 'postgres' ? models.items.map((model) => ({ name: model.model_family, version: model.id, status: model.status === 'production' ? 'production' : model.status === 'candidate' ? 'candidate' : 'shadow', metric: 'доступно в реестре', metricValue: model.metrics.macro_f1 == null ? '—' : model.metrics.macro_f1.toFixed(3), updatedAt: model.created_at })) : []
  return { tickets, regions, topics, alerts, forecast: forecastPoints, models: modelData, learning: learningData }
}

export async function loadDashboard(): Promise<ApiResult<DashboardData>> {
  try {
    return { data: await loadApiDashboard(), source: 'api' }
  } catch (error) {
    return {
      data: demoData,
      source: 'demo',
      error: error instanceof Error ? error.message : 'API недоступен',
    }
  }
}

export async function loadTickets(): Promise<ApiResult<Ticket[]>> {
  try {
    const response = await request<{ items: BackendTicket[] }>('/tickets?limit=50')
    const tickets = await Promise.all(
      response.items.map(async (item) => {
        const [detail, preview] = await Promise.all([
          request<BackendTicketDetail>(`/tickets/${encodeURIComponent(item.id)}`).catch(() => undefined),
          request<BackendAssistPreview>('/assist/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticket_id: item.id }),
          }).catch(() => undefined),
        ])
        return mapBackendTicket(item, detail, response.items, preview)
      }),
    )
    return { data: tickets, source: 'api' }
  } catch (error) {
    return {
      data: demoData.tickets,
      source: 'demo',
      error: error instanceof Error ? error.message : 'API недоступен',
    }
  }
}

export async function submitDecision(ticketId: string, decision: { status: 'confirmed' | 'corrected'; topic?: string; service?: string; priority?: string }) {
  try {
    const action = decision.status === 'confirmed' ? 'confirm' : 'correct'
    await request(`/assist/${encodeURIComponent(ticketId)}/${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topic_id: topicIdForLabel(decision.topic),
        service: decision.service,
        priority: decision.priority === 'Высокий' ? 'high' : decision.priority === 'Низкий' ? 'low' : decision.priority === 'Средний' ? 'medium' : undefined,
      }),
    })
    return { source: 'api' as const }
  } catch (error) {
    return { source: 'demo' as const, error: error instanceof Error ? error.message : 'API недоступен' }
  }
}
