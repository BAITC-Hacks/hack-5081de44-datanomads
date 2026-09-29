import { formatUiDateTime, localeTag, translateUi } from '../uiSettings'
import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { loadContextHandoffPackage, loadOutcomeVerification, loadRelatedTicketDetail, loadRoutingFeedback, previewTicketWithContext, submitDecision, submitOutcomeVerification, submitRelationFeedback, submitRoutingFeedback } from '../api/client'
import type { DashboardData, ContextHandoffPackage, OutcomeVerificationSnapshot, OutcomeVerificationState, Priority, RelatedTicketDetail, RelationSuggestionSnapshot, RoutingFeedbackRecord, RuleProvenance, Ticket } from '../types'
import { Icon } from '../components/Icon'
import { languageLabel, languageReviewNotice } from '../language'
import { confidenceStateLabel, confidenceStateNotice, normalizeConfidenceState } from '../classification'
import { compareRelatedTicketContext, formatDecisionTime } from '../operator'
import { formatExplainabilityFact } from '../routing'

type RelatedTicketPanelState = {
  ticketId: string
  matchedFactors: string[]
  currentTicket: Ticket
  loading: boolean
  detail?: RelatedTicketDetail
  error?: string
}

const PREVIEW_STAGE_LABELS: Record<string, string> = {
  language: 'определение языка',
  classification: 'классификация',
  routing: 'маршрутизация',
  priority: 'определение приоритета',
  retrieval: 'поиск похожих обращений',
  duplicate_repeat: 'поиск повторов и дубликатов',
  response_template: 'шаблон ответа',
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`
}

function relationFeedbackType(relation: Ticket['similar'][number]['relation']): 'DUPLICATE' | 'REPEAT' | 'SIMILAR' {
  if (relation === 'Возможный дубликат') return 'DUPLICATE'
  if (relation === 'Возможное повторное обращение') return 'REPEAT'
  return 'SIMILAR'
}

export function OperatorPage({ tickets, overview, taxonomy, onDataChange, onToast }: { tickets: Ticket[]; overview: DashboardData['overview']; taxonomy: DashboardData['filterOptions']; onDataChange: (tickets: Ticket[]) => void; onToast: (message: string) => void }) {
  const [selectedId, setSelectedId] = useState(tickets[0]?.id ?? '')
  const [query, setQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<'all' | 'new' | 'confirmed' | 'corrected'>('all')
  const [priorityFilter, setPriorityFilter] = useState<'all' | Priority>('all')
  const [mobileDetailOpen, setMobileDetailOpen] = useState(false)
  const [relatedDetail, setRelatedDetail] = useState<RelatedTicketPanelState | null>(null)
  const relatedDetailRequestId = useRef(0)
  const selected = tickets.find((ticket) => ticket.id === selectedId) ?? tickets[0]

  const filtered = useMemo(() => tickets.filter((ticket) => {
    const needle = query.trim().toLowerCase()
    const matchesQuery = !needle || [ticket.id, ticket.originalText, ticket.topic, ticket.region].join(' ').toLowerCase().includes(needle)
    const matchesStatus = statusFilter === 'all' || ticket.status === statusFilter
    const matchesPriority = priorityFilter === 'all' || ticket.priority === priorityFilter
    return matchesQuery && matchesStatus && matchesPriority
  }), [tickets, query, statusFilter, priorityFilter])

  const updateTicket = async (ticketId: string, decision: { status: 'confirmed' | 'corrected'; topic?: string; service?: string; priority?: string }) => {
    const ticket = tickets.find((item) => item.id === ticketId)
    if (!ticket) return
    try {
      const result = await submitDecision(ticketId, decision, taxonomy)
      const updated = tickets.map((item) => {
        if (item.id !== ticketId) return item
        if (result.ticket) return result.ticket
        return {
          ...item,
          ...decision,
          priority: (decision.priority as Priority | undefined) ?? item.priority,
          ...(decision.status === 'corrected' ? {
            responseTemplate: '',
            responseTemplateApproved: false,
            responseTemplateSource: 'MANUAL_REQUIRED',
          } : {}),
        }
      })
      onDataChange(updated)
      const feedbackNotice = result.source === 'api' && result.learningFeedbackStatus === 'NO_ACTIVE_COLLECT_CYCLE'
        ? '; обратная связь не включена в цикл: сейчас нет активного COLLECT'
        : ''
      onToast(result.source === 'demo' ? `Решение по ${ticketId} изменено только на этом экране` : `${('warning' in result && result.warning) || `Решение по ${ticketId} сохранено`}${feedbackNotice}`)
    } catch (error) {
      onToast(`Не удалось сохранить решение: ${error instanceof Error ? error.message : 'ошибка API'}`)
    }
  }

  const updateRelation = async (ticketId: string, relatedTicketId: string, relation: 'DUPLICATE' | 'REPEAT' | 'SIMILAR' | 'UNRELATED', decision: 'CONFIRMED' | 'REJECTED', suggestion?: RelationSuggestionSnapshot) => {
    try {
      await submitRelationFeedback(ticketId, relatedTicketId, relation, decision, suggestion)
      onToast(`Обратная связь по ${relatedTicketId} сохранена: ${relation.toLowerCase()} · ${decision.toLowerCase()}`)
    } catch (error) {
      onToast(`Не удалось сохранить связь: ${error instanceof Error ? error.message : 'ошибка API'}`)
    }
  }

  const openRelatedTicket = async (ticketId: string, matchedFactors: string[], currentTicket: Ticket) => {
    const requestId = relatedDetailRequestId.current + 1
    relatedDetailRequestId.current = requestId
    setRelatedDetail({ ticketId, matchedFactors, currentTicket, loading: true })
    try {
      const detail = await loadRelatedTicketDetail(ticketId)
      if (relatedDetailRequestId.current !== requestId) return
      setRelatedDetail({ ticketId, matchedFactors, currentTicket, loading: false, detail })
    } catch (error) {
      if (relatedDetailRequestId.current !== requestId) return
      setRelatedDetail({
        ticketId,
        matchedFactors,
        currentTicket,
        loading: false,
        error: error instanceof Error ? error.message : 'ошибка API',
      })
    }
  }

  const closeRelatedTicket = () => {
    relatedDetailRequestId.current += 1
    setRelatedDetail(null)
  }

  const confirmationRate = overview.operatorDecisions ? Math.round((overview.confirmedDecisions / overview.operatorDecisions) * 100) : 0
  const decisionLatency = formatDecisionTime(overview.avgDecisionMinutes, '—')
  const confidenceTickets = filtered.filter((ticket) => ticket.confidenceAvailable !== false)
  const averageConfidence = confidenceTickets.length
    ? new Intl.NumberFormat(localeTag(), { style: 'percent', maximumFractionDigits: 1 }).format(confidenceTickets.reduce((sum, ticket) => sum + ticket.confidence, 0) / confidenceTickets.length)
    : '—'
  return <div className="operator-page">
    <div className="operator-summary summary-strip"><div className="summary-item"><span className="summary-label">{translateUi("Обращений в выборке")}</span><span className="summary-value">{filtered.length}</span><span className="summary-trend">{translateUi("по текущим фильтрам")}</span></div><div className="summary-item"><span className="summary-label">{translateUi("Подтверждено без правок")}</span><span className="summary-value">{confirmationRate}%</span><span className="summary-trend">{overview.confirmedDecisions} / {overview.operatorDecisions}</span></div><div className="summary-item"><span className="summary-label">{translateUi("Среднее до решения")}</span><span className="summary-value">{decisionLatency}</span><span className="summary-trend">{translateUi(overview.operatorDecisions ? 'по данным решений' : 'решений пока нет')}</span></div><div className="summary-item"><span className="summary-label">{translateUi("Средняя уверенность")}</span><span className="summary-value">{averageConfidence}</span><span className="summary-trend">{localeTag() === 'kk-KZ' ? `кезектегі ${confidenceTickets.length} бағалау бойынша` : `по ${confidenceTickets.length} оценкам в очереди`}</span></div></div>
    <div className="workbench-grid operator-shell">
      <section className="ticket-queue panel" aria-label={translateUi("Очередь обращений")}>
        <div className="section-toolbar panel-head"><div><h2>{translateUi("Очередь")} <span className="count-pill">{filtered.length}</span></h2><p>{translateUi("Обращений в загруженной выборке")}</p></div></div>
        <div className="filter-row queue-tools"><label className="inline-search searchbox"><Icon name="search" size={16} /><input aria-label={translateUi("Поиск по очереди")} value={query} onChange={(event) => setQuery(event.target.value)} placeholder={translateUi("ID, текст, тема, регион")} /></label><button type="button" className="button button-quiet quiet" onClick={() => { setQuery(''); setStatusFilter('all'); setPriorityFilter('all') }}>{translateUi("Сбросить")}</button></div>
        <div className="queue-filter-row" role="group" aria-label={translateUi("Фильтры очереди")}>
          {([['all', 'Все'], ['new', 'Новые'], ['confirmed', 'Подтверждённые'], ['corrected', 'Исправленные']] as const).map(([value, label]) => <button type="button" key={value} className={`chip ${statusFilter === value ? 'active' : ''}`} aria-pressed={statusFilter === value} onClick={() => setStatusFilter(value)}>{translateUi(label)}</button>)}
          {(['Критический', 'Высокий', 'Средний', 'Низкий'] as const).map((priority) => <button type="button" key={priority} className={`chip ${priorityFilter === priority ? 'active' : ''}`} aria-pressed={priorityFilter === priority} onClick={() => setPriorityFilter((current) => current === priority ? 'all' : priority)}>{translateUi(priority)}</button>)}
        </div>
        <div className="ticket-table-wrap table-wrap"><table className="ticket-table data-table"><thead><tr><th scope="col">{translateUi("Обращение")}</th><th scope="col">{translateUi("Тема")}</th><th scope="col">{translateUi("Уверенность")}</th><th scope="col">{translateUi("Регион")}</th><th scope="col">{translateUi("Приоритет")}</th></tr></thead><tbody>{filtered.map((ticket) => <tr key={ticket.id} className={selected?.id === ticket.id ? 'selected-row' : ''}><td><button type="button" className="ticket-row-trigger" aria-label={`${translateUi('Открыть обращение')} ${ticket.id}`} onClick={() => { setSelectedId(ticket.id); setMobileDetailOpen(true) }}><span className="ticket-id">{ticket.id}<span className={`channel-dot channel-${ticket.channel.replace(/[^a-zA-Z]/g, '').toLowerCase()}`} /></span><span className="ticket-preview">{ticket.originalText}</span><span className="ticket-time">{formatUiDateTime(ticket.createdAt)} · {translateUi(languageLabel(ticket.language))}</span></button></td><td><span className="topic-cell">{translateUi(ticket.topic)}</span><span className="status-text">{translateUi(ticket.status === 'new' ? 'Нужно решение' : ticket.status === 'confirmed' ? 'Подтверждено' : 'Исправлено')}</span></td><td><Confidence value={ticket.confidence} compact available={ticket.confidenceAvailable !== false} /></td><td><span className="region-cell">{translateUi(ticket.region)}</span></td><td><PriorityBadge priority={ticket.priority} /></td></tr>)}</tbody></table>{filtered.length === 0 && <div className="empty-state state"><div className="state-icon">⌕</div><h3>{translateUi("Ничего не найдено")}</h3><p>{translateUi("Измените запрос или сбросьте фильтры.")}</p><button className="button button-quiet quiet" onClick={() => { setQuery(''); setStatusFilter('all'); setPriorityFilter('all') }}>{translateUi("Сбросить фильтры")}</button></div>}</div>
      </section>
      {selected && <TicketDetail ticket={selected} taxonomy={taxonomy} open={mobileDetailOpen} onClose={() => setMobileDetailOpen(false)} onDecision={updateTicket} onRelationFeedback={updateRelation} onOpenRelated={openRelatedTicket} />}
    </div>
    {relatedDetail && <RelatedTicketDetailPanel state={relatedDetail} taxonomy={taxonomy} onClose={closeRelatedTicket} onRetry={() => { void openRelatedTicket(relatedDetail.ticketId, relatedDetail.matchedFactors, relatedDetail.currentTicket) }} />}
  </div>
}

function TicketDetail({ ticket, taxonomy, open, onClose, onDecision, onRelationFeedback, onOpenRelated }: { ticket: Ticket; taxonomy: DashboardData['filterOptions']; open: boolean; onClose: () => void; onDecision: (ticketId: string, decision: { status: 'confirmed' | 'corrected'; topic?: string; service?: string; priority?: string }) => Promise<void>; onRelationFeedback: (ticketId: string, relatedTicketId: string, relation: 'DUPLICATE' | 'REPEAT' | 'SIMILAR' | 'UNRELATED', decision: 'CONFIRMED' | 'REJECTED', suggestion?: RelationSuggestionSnapshot) => Promise<void>; onOpenRelated: (ticketId: string, matchedFactors: string[], currentTicket: Ticket) => void }) {
  const contextRequestId = useRef(0)
  const [correctionOpen, setCorrectionOpen] = useState(false)
  const [contextAnswer, setContextAnswer] = useState('')
  const [contextResult, setContextResult] = useState<Ticket | null>(null)
  const [contextLoading, setContextLoading] = useState(false)
  const [contextError, setContextError] = useState('')
  const [replyDraftMode, setReplyDraftMode] = useState<'closed' | 'template' | 'manual'>('closed')
  const [replyDraft, setReplyDraft] = useState('')
  const [templateIgnored, setTemplateIgnored] = useState(false)
  const [topic, setTopic] = useState(ticket.topic)
  const [service, setService] = useState(ticket.service)
  const [priority, setPriority] = useState<Priority>(ticket.priority)
  const confidenceState = normalizeConfidenceState(ticket.confidenceState, ticket.confidence, ticket.confidenceAvailable !== false)
  const confidenceAvailable = ticket.confidenceAvailable !== false && confidenceState !== 'UNAVAILABLE'
  const hasOperatorDecision = ticket.confirmedDecisionAvailable ?? (ticket.status !== 'new')
  const demoRecommendationFallback = ticket.confirmedDecisionAvailable === undefined && ticket.status === 'new'
  const recommendedService = ticket.recommendedService ?? (demoRecommendationFallback ? ticket.service : undefined)
  const recommendedPriority = ticket.recommendedPriority ?? (demoRecommendationFallback ? ticket.priority : undefined)
  const recommendedServiceProvenance = ticket.recommendedServiceProvenance
    ?? (demoRecommendationFallback ? ticket.serviceProvenance : undefined)
  const recommendedPriorityProvenance = ticket.recommendedPriorityProvenance
    ?? (demoRecommendationFallback ? ticket.priorityProvenance : undefined)
  const hasApprovedTemplate = ticket.responseTemplateApproved === true && ticket.responseTemplateSource === 'APPROVED_TEMPLATE'
  const preview = ticket.assistPreview
  const retrievalStage = preview?.stages.find((stage) => stage.name === 'retrieval')
  const emptyHistoryMessage = !preview
    ? 'История обращений не загружена.'
    : retrievalStage && retrievalStage.status !== 'completed'
      ? 'Поиск по истории недоступен. Проверьте связанные обращения вручную.'
      : 'Подходящие похожие обращения, дубликаты и повторы не найдены.'
  const languageNotice = languageReviewNotice(ticket.language)
  const incompleteStages = preview?.stages
    .filter((stage) => ['unavailable', 'skipped', 'unknown'].includes(stage.status))
    .map((stage) => PREVIEW_STAGE_LABELS[stage.name] ?? stage.name) ?? []
  const modelVersions = preview ? Object.entries(preview.model_versions).map(([name, version]) => `${name}: ${version}`).join(' · ') : ''
  const topicOptions = useMemo(() => {
    const options: Array<[string, string]> = [
      ...taxonomy.topics.map((item) => [item.id, item.label] as [string, string]),
      [ticket.topic, ticket.topic],
      ...ticket.alternatives.map((item) => [item.topic, item.topic] as [string, string]),
    ]
    return Array.from(new Map(options.map((option) => [option[1], option])).values())
  }, [taxonomy.topics, ticket.topic, ticket.alternatives])
  const serviceOptions = useMemo(() => {
    const options: Array<[string, string]> = [
      ...taxonomy.services.map((item) => [item.id, item.label] as [string, string]),
      [ticket.service, ticket.service],
    ]
    return Array.from(new Map(options.map((option) => [option[1], option])).values())
  }, [taxonomy.services, ticket.service])
  useEffect(() => {
    contextRequestId.current += 1
    setTopic(ticket.topic)
    setService(ticket.service)
    setPriority(ticket.priority)
    setCorrectionOpen(ticket.status === 'new' && (confidenceState === 'LOW_CONFIDENCE' || confidenceState === 'UNAVAILABLE'))
    setReplyDraftMode('closed')
    setReplyDraft('')
    setTemplateIgnored(false)
    setContextAnswer('')
    setContextResult(null)
    setContextLoading(false)
    setContextError('')
  }, [ticket.id, ticket.topic, ticket.service, ticket.priority, ticket.status, ticket.responseTemplateId, ticket.responseTemplateVersion, ticket.responseTemplateSource, ticket.responseTemplate, ticket.actionableContext?.question, confidenceState])

  const recalculateContext = async () => {
    const requestId = contextRequestId.current + 1
    contextRequestId.current = requestId
    setContextLoading(true)
    setContextError('')
    try {
      const result = await previewTicketWithContext(ticket, contextAnswer)
      if (contextRequestId.current === requestId) setContextResult(result)
    } catch (error) {
      if (contextRequestId.current === requestId) setContextError(error instanceof Error ? error.message : 'ошибка API')
    } finally {
      if (contextRequestId.current === requestId) setContextLoading(false)
    }
  }

  return <aside className={`ticket-detail detail panel ${open ? 'ticket-detail-open' : ''}`} aria-label={`Детали обращения ${ticket.id}`}>
    <div className="detail-header detail-head"><div><div className="detail-overline"><span className={`status-indicator ${ticket.status}`} />{translateUi(ticket.status === 'new' ? 'Требует решения' : ticket.status === 'confirmed' ? 'Подтверждено' : 'Исправлено')}</div><h2>{ticket.id}</h2></div><button className="icon-button icon-btn detail-close" aria-label={translateUi("Закрыть детали")} onClick={onClose}><Icon name="close" size={18} /></button></div>
    <div className="detail-scroll">
      <div className="original-text-block detail-section"><div className="field-label">{translateUi("Оригинальный текст")} <span className="language-chip">{translateUi(languageLabel(ticket.language))}</span></div><p>«{ticket.originalText}»</p><div className="source-line">{translateUi(ticket.channel)} · {formatUiDateTime(ticket.createdAt)} · {translateUi(ticket.region)}{ticket.externalRef && ` · № ${ticket.externalRef}`}</div></div>
      {languageNotice && <p className="panel-note" role="status">{translateUi(languageNotice)}</p>}
      {preview?.needs_review && (preview.status === 'partial' || incompleteStages.length > 0 || confidenceState === 'CONFIDENT') && <div className="preview-notice" role="status"><p>{translateUi(preview.status === 'partial' ? 'Предпросмотр неполный.' : incompleteStages.length > 0 ? 'Часть функций недоступна.' : 'Рекомендацию нужно проверить.')} {incompleteStages.length > 0 && `${translateUi('Недоступно:')} ${incompleteStages.map(translateUi).join(', ')}. `}{translateUi("Подтвердите тему, службу и приоритет после ручной проверки.")}</p><details><summary>{translateUi("Технические сведения")}</summary><small>{translateUi("Запрос")} {preview.request_id} · {preview.latency_ms.toFixed(0)} {translateUi("мс")}{modelVersions ? ` · ${modelVersions}` : ''}</small></details></div>}
      <div className="detail-section">
        <div className="field-label">{translateUi("Модель предложила")}</div>
        <div className="prediction-row">
          <div>
            <div className="prediction-topic">{translateUi(ticket.predictedTopic ?? ticket.topic)}</div>
            <div className={`classification-state classification-${confidenceState.toLowerCase()}`}>
              {translateUi(confidenceStateNotice(confidenceState))}
            </div>
          </div>
          <Confidence value={ticket.confidence} state={confidenceState} available={confidenceAvailable} />
        </div>
        {confidenceState === 'UNCERTAIN' && (
          <div className="alternatives">
            <span className="field-label">{translateUi("Возможные альтернативы")}</span>
            {ticket.alternatives.length ? ticket.alternatives.map((alternative) => (
              <div className="alternative-row" key={alternative.topic}>
                <span>{translateUi(alternative.topic)}</span>
                <span>{formatPercent(alternative.confidence)}</span>
              </div>
            )) : <p className="panel-note">{translateUi("Альтернативы не получены. Проверьте тему вручную.")}</p>}
          </div>
        )}
        <details className="model-meta">
          <summary>{translateUi("Технические сведения")}</summary>
          <span>{translateUi("Оценка модели:")} {confidenceAvailable ? formatPercent(ticket.confidence) : translateUi('нет данных')}</span>
          {ticket.modelVersion && <span>{translateUi("Модель:")} {ticket.modelVersion}</span>}
        </details>
      </div>
      {correctionOpen && (
        <div className="correction-panel">
          <div className="correction-heading">
            <strong>{translateUi("Выберите тему вручную")}</strong>
            <button className="icon-button" aria-label={translateUi("Закрыть форму исправления")} onClick={() => setCorrectionOpen(false)}>
              <Icon name="close" size={15} />
            </button>
          </div>
          <label>{translateUi("Тема")}
            <select value={topic} onChange={(event) => setTopic(event.target.value)}>
              {topicOptions.map(([id, label]) => <option value={label} key={id}>{translateUi(label)}</option>)}
              <option value="Другая тема">{translateUi("Другая тема")}</option>
            </select>
          </label>
          <label>{translateUi("Служба")}
            <select value={service} onChange={(event) => setService(event.target.value)}>
              {serviceOptions.map(([id, label]) => <option value={label} key={id}>{translateUi(label)}</option>)}
              <option value="Другая служба">{translateUi("Другая служба")}</option>
            </select>
          </label>
          <label>{translateUi("Приоритет")}
            <select value={priority} onChange={(event) => setPriority(event.target.value as Priority)}>
              <option value="Высокий">{translateUi("Высокий")}</option>
              <option value="Критический">{translateUi("Критический")}</option>
              <option value="Средний">{translateUi("Средний")}</option>
              <option value="Низкий">{translateUi("Низкий")}</option>
              <option value="Не определён">{translateUi("Не определён")}</option>
            </select>
          </label>
          <button className="button button-primary full-width" onClick={() => onDecision(ticket.id, { status: 'corrected', topic, service, priority })}>
            <Icon name="check" size={16} />{translateUi("Сохранить исправление")}
          </button>
        </div>
      )}
      <div className="detail-section">
        <div className="field-label">{translateUi("Рекомендация Pulse · маршрутизация и приоритет")}</div>
        <div className="routing-grid">
          <div className="routing-field">
            <span>{translateUi("Служба")}</span>
            <strong>{translateUi(recommendedService ?? 'Не предоставлена')}</strong>
            <RoutingProvenance
              provenance={recommendedServiceProvenance}
              fallbackReason="Источник рекомендованной службы не сохранён; проверьте вручную"
            />
          </div>
          <div className="routing-field">
            <span>{translateUi("Приоритет")}</span>
            <PriorityBadge priority={recommendedPriority ?? 'Не определён'} />
            <RoutingProvenance
              provenance={recommendedPriorityProvenance}
              fallbackReason="Источник рекомендованного приоритета не сохранён; проверьте вручную"
            />
          </div>
        </div>
        <p className="panel-note"><strong>{translateUi("Основание маршрутизации:")}</strong> {translateUi(ticket.routingReason?.trim() || 'Не предоставлено; проверьте службу и приоритет вручную.')}</p>
      </div>
      {ticket.actionableContext?.status === 'suggested' && ticket.actionableContext.question && (
        <section className="actionable-context" aria-label={translateUi("Уточнение для выбора темы")}>
          <div className="field-label">{translateUi("Рекомендация зависит от уточнения")}</div>
          <p className="actionable-context-question">{ticket.actionableContext.question}</p>
          <p className="panel-note">{translateUi("Правило")} {ticket.actionableContext.ruleId}{translateUi(": меняются")} {ticket.actionableContext.decisionCriticalFields.map((field) => field === 'service' ? 'служба' : 'приоритет').join(' и ')}.</p>
          <ul className="actionable-context-options">
            {ticket.actionableContext.options.map((option) => (
              <li key={option.topicId}><strong>{option.topicLabel}</strong><span>{option.service} · {option.priority}</span></li>
            ))}
          </ul>
          <label className="field-label" htmlFor={`context-answer-${ticket.id}`}>{translateUi("Ответ заявителя, полученный вне Pulse")}</label>
          <textarea
            id={`context-answer-${ticket.id}`}
            aria-label={translateUi("Ответ заявителя для пересчёта")}
            value={contextAnswer}
            onChange={(event) => setContextAnswer(event.target.value)}
            maxLength={2_000}
            rows={3}
            placeholder={translateUi("Внесите ответ, который оператор уже получил")}
          />
          <p className="panel-note">{translateUi("Ответ используется только для временного пересчёта. Исходное обращение не меняется; отправки в CRM нет.")}</p>
          <button className="button button-secondary" onClick={() => void recalculateContext()} disabled={contextLoading || !contextAnswer.trim()}>
            {contextLoading ? 'Пересчитываем…' : 'Пересчитать рекомендацию'}
          </button>
          {contextError && <p className="actionable-context-error" role="alert">{translateUi("Не удалось пересчитать рекомендацию:")} {contextError}</p>}
          {contextResult && (
            <div className="actionable-context-result" role="status" aria-live="polite">
              <strong>{translateUi("Временный результат после уточнения")}</strong>
              <span>{translateUi("Тема:")} {contextResult.predictedTopic ?? contextResult.topic}</span>
              <span>{translateUi("Служба:")} {contextResult.recommendedService ?? 'Не определена'}</span>
              <span>{translateUi("Приоритет:")} {contextResult.recommendedPriority ?? 'Не определён'}</span>
              {contextResult.actionableContext?.status === 'suggested' && contextResult.actionableContext.question && (
                <span>{translateUi("Осталась неопределённость:")} {contextResult.actionableContext.question}</span>
              )}
              {contextResult.actionableContext?.status === 'manual_review' && <span>{contextResult.actionableContext.reason}</span>}
            </div>
          )}
        </section>
      )}
      {hasOperatorDecision && (
        <div className="detail-section">
          <div className="field-label">{ticket.status === 'corrected' ? 'Исправление оператора' : 'Подтверждённое решение оператора'}</div>
          <div className="routing-grid">
            <div className="routing-field">
              <span>{translateUi("Служба")}</span>
              <strong>{translateUi(ticket.service)}</strong>
              <RoutingProvenance
                provenance={ticket.serviceProvenance}
                fallbackReason="Источник подтверждённой службы не сохранён; проверьте решение вручную"
              />
            </div>
            <div className="routing-field">
              <span>{translateUi("Приоритет")}</span>
              <PriorityBadge priority={ticket.priority} />
              <RoutingProvenance
                provenance={ticket.priorityProvenance}
                fallbackReason="Источник подтверждённого приоритета не сохранён; проверьте решение вручную"
              />
            </div>
          </div>
        </div>
      )}
      {!hasOperatorDecision && ticket.status !== 'new' && (
        <p className="panel-note" role="status">{translateUi("Статус записи отмечен как разобранный, но данные подтверждённого решения не предоставлены.")}</p>
      )}
      <ContextHandoffPanel
        ticket={ticket}
        hasConfirmedRoute={ticket.confirmedDecisionAvailable === true || Boolean(ticket.latestDecision?.service)}
      />
      <OutcomeVerificationPanel ticket={ticket} />
      <RoutingFeedbackPanel ticket={ticket} taxonomy={taxonomy} />
      <div className="detail-section">
        <div className="field-label">{translateUi("Ответ оператору")} <span className="language-chip">{hasApprovedTemplate ? (ticket.responseTemplateVersion ? `Утверждённый · v${ticket.responseTemplateVersion}` : 'Утверждённый шаблон') : 'Ручной ответ'}</span></div>
        {hasApprovedTemplate && !templateIgnored && (
          <div className="response-template">
            <p>{ticket.responseTemplate}</p>
            <div className="response-template-actions">
              <button className="button button-quiet" onClick={() => { setReplyDraft(ticket.responseTemplate); setReplyDraftMode('template') }}>{translateUi("Использовать шаблон")}</button>
              <button className="text-button" onClick={() => { setTemplateIgnored(true); setReplyDraftMode('closed'); setReplyDraft('') }}>{translateUi("Игнорировать")}</button>
            </div>
          </div>
        )}
        {!hasApprovedTemplate && replyDraftMode === 'closed' && (
          <p className="panel-note" role="status">
            {ticket.responseTemplateSource === 'UNAVAILABLE' && ticket.responseTemplate.trim()
              ? ticket.responseTemplate
              : ticket.status === 'new'
                ? 'Утверждённый шаблон появится после подтверждения темы и службы. Ответ можно написать вручную.'
                : 'Для этого решения нет утверждённого шаблона. Составьте ответ вручную.'}
          </p>
        )}
        {templateIgnored && replyDraftMode === 'closed' && (
          <button className="text-button" onClick={() => setTemplateIgnored(false)}>{translateUi("Показать утверждённый шаблон")}</button>
        )}
        {replyDraftMode === 'closed' && (
          <button className="button button-quiet" onClick={() => { setReplyDraft(''); setReplyDraftMode('manual') }}>{translateUi("Написать вручную")}</button>
        )}
        {replyDraftMode !== 'closed' && (
          <div className="response-draft">
            <label className="field-label" htmlFor={`reply-draft-${ticket.id}`}>{translateUi("Черновик ответа")}</label>
            <textarea id={`reply-draft-${ticket.id}`} aria-label={translateUi("Черновик ответа")} value={replyDraft} onChange={(event) => setReplyDraft(event.target.value)} rows={5} />
            <p className="panel-note">{translateUi("Черновик доступен только на этом экране. Отправки в CRM нет.")}</p>
            <button className="text-button" onClick={() => { setReplyDraftMode('closed'); setReplyDraft('') }}>{translateUi("Закрыть черновик")}</button>
          </div>
        )}
      </div>
      <div className="detail-section">
        <div className="section-inline-heading">
          <div className="field-label">{translateUi("История и связанные обращения")} <span className="count-pill">{ticket.similar.length}</span></div>
          <span className="field-label">{translateUi("проверьте связь")}</span>
        </div>
        <div className="similar-list">
          {ticket.similar.length ? ticket.similar.map((item) => {
            const relation = relationFeedbackType(item.relation)
            return <div className="similar-item" key={item.id}>
              <div className="similar-main-column">
                <button className="similar-main similar-open" aria-label={`Открыть оригинал ${item.id}`} onClick={() => onOpenRelated(item.id, item.matchedFactors ?? [], ticket)}>
                  <strong>{item.id}</strong><span>{item.title}</span><Icon name="arrow" size={14} />
                </button>
                <span className="similar-factors">{item.matchedFactors?.length ? item.matchedFactors.join(' · ') : 'Совпадающие признаки не предоставлены'}</span>
                {item.suggestion
                  ? <small>{translateUi("Отбор top-K: score")} {formatPercent(item.similarity)} {translateUi("· порог")} {formatPercent(item.suggestion.threshold)}</small>
                  : <small>{translateUi("Порог отбора top-K не сохранён")}</small>}
              </div>
              <div className="similar-meta">
                <span className={`relation-badge ${item.relation === 'Возможный дубликат' ? 'relation-duplicate' : item.relation === 'Возможное повторное обращение' ? 'relation-repeat' : ''}`}>{item.relation}</span>
                <span>{translateUi("Создано")} {item.createdAt}</span>
                <span>{formatPercent(item.similarity)}</span>
                <button className="text-button" onClick={() => onRelationFeedback(ticket.id, item.id, relation, 'CONFIRMED', item.suggestion)}>{translateUi("Подтвердить")}</button>
                <button className="text-button" onClick={() => onRelationFeedback(ticket.id, item.id, relation, 'REJECTED', item.suggestion)}>{translateUi("Отклонить")}</button>
              </div>
              {item.relation === 'Возможное повторное обращение' && <p className="recurrence-monitoring-note" role="note">{translateUi("Возможный повтор после недавнего закрытия. Проверьте предыдущую историю.")}</p>}
            </div>
          }) : <p className="panel-note">{emptyHistoryMessage}</p>}
        </div>
      </div>
    </div>
    {!correctionOpen && <div className="detail-actions"><button className="button button-primary" onClick={() => onDecision(ticket.id, { status: 'confirmed' })}><Icon name="check" size={16} />{translateUi("Подтвердить")}</button><button className="button button-secondary" onClick={() => setCorrectionOpen(true)}><Icon name="edit" size={16} />{translateUi("Исправить")}</button></div>}
  </aside>
}

function resolutionMemoryLabel(actionRecorded: boolean, outcomeState?: OutcomeVerificationState): string {
  const action = actionRecorded ? 'Action recorded' : 'Action not recorded'
  return outcomeState ? `${action}; outcome ${outcomeState}` : `${action}; outcome unavailable`
}

function OutcomeVerificationPanel({ ticket }: { ticket: Ticket }) {
  const [snapshot, setSnapshot] = useState<OutcomeVerificationSnapshot | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [selectedState, setSelectedState] = useState<Exclude<OutcomeVerificationState, 'UNKNOWN'> | ''>('')

  useEffect(() => {
    let active = true
    setSnapshot(null)
    setLoading(true)
    setError('')
    setNotice('')
    setSelectedState('')
    loadOutcomeVerification(ticket.id)
      .then((result) => { if (active) setSnapshot(result) })
      .catch((requestError: unknown) => {
        if (active) setError(requestError instanceof Error ? requestError.message : 'ошибка API')
      })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [ticket.id])

  const saveVerification = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!selectedState) return
    setSaving(true)
    setError('')
    setNotice('')
    try {
      const result = await submitOutcomeVerification(ticket.id, selectedState)
      setSnapshot(result)
      setNotice('Операторская симуляция сохранена. Официальный статус обращения не менялся.')
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'ошибка API')
    } finally {
      setSaving(false)
    }
  }

  const stateLabel = (state: OutcomeVerificationState) => ({
    UNKNOWN: 'Нет подтверждения (UNKNOWN)',
    VERIFIED: 'Результат подтверждён',
    PARTIAL: 'Результат частичный',
    DISPUTED: 'Результат оспаривается',
  })[state]

  return <section className="detail-section outcome-verification-panel" aria-label={translateUi("Проверка результата обращения")}>
    <div className="section-inline-heading">
      <div className="field-label">{translateUi("Проверка результата обращения")}</div>
      <span className="routing-simulation-badge">DEMO SIMULATION</span>
    </div>
    <p className="panel-note">{translateUi("Реальный канал обратной связи граждан не подключён. Эта запись — симуляция оператора; даже состояние VERIFIED не подтверждает результат гражданином и не меняет CRM.")}</p>
    {loading && <p className="panel-note" role="status">{translateUi("Загружаю состояние результата…")}</p>}
    {error && <p className="routing-feedback-error" role="alert">{translateUi("Не удалось загрузить или сохранить проверку результата:")} {error}</p>}
    {snapshot && <>
      <dl className="routing-feedback-routes outcome-verification-status">
        <div><dt>{translateUi("Официальный статус CRM")}</dt><dd>{snapshot.officialTicketStatus}</dd></div>
        <div><dt>{translateUi("Состояние результата в Pulse")}</dt><dd>{stateLabel(snapshot.state)}</dd></div>
      </dl>
      <p className="resolution-memory-state" role="status">{resolutionMemoryLabel(Boolean(ticket.latestDecision), snapshot.state)}</p>
      {!snapshot.latest && <p className="panel-note">{translateUi("Сведений о результате нет. Молчание и отсутствие обратной связи остаются UNKNOWN, даже если официальная запись закрыта.")}</p>}
      {snapshot.latest && <p className="panel-note">{translateUi("Последняя запись:")} {snapshot.latest.sourceSystem} · {snapshot.latest.channel} {translateUi("· оператор")} {snapshot.latest.actorUserId} · {snapshot.latest.createdAt}</p>}
      <form className="routing-feedback-form" onSubmit={(event) => void saveVerification(event)}>
        <label>{translateUi("Симулированный результат")}
          <select value={selectedState} onChange={(event) => setSelectedState(event.target.value as Exclude<OutcomeVerificationState, 'UNKNOWN'>)}>
            <option value="" disabled>{translateUi("Выберите результат")}</option>
            <option value="VERIFIED">{translateUi("Результат подтверждён")}</option>
            <option value="PARTIAL">{translateUi("Результат частичный")}</option>
            <option value="DISPUTED">{translateUi("Результат оспаривается")}</option>
          </select>
        </label>
        <button className="button button-secondary" type="submit" disabled={saving || loading || !selectedState}>
          {saving ? 'Сохраняем…' : 'Сохранить симуляцию'}
        </button>
      </form>
      {notice && <p className="routing-feedback-notice" role="status">{notice}</p>}
      {snapshot.history.length > 0 && <div className="routing-feedback-history" aria-label={translateUi("История проверки результата")}>
        {snapshot.history.map((item) => <article className="routing-feedback-record" key={item.id}>
          <div><strong>{stateLabel(item.state)}</strong><span>{item.sourceSystem} · {item.channel}</span></div>
          <small>{translateUi("Оператор")} {item.actorUserId} · {item.createdAt}</small>
        </article>)}
      </div>}
    </>}
  </section>
}

function RoutingFeedbackPanel({ ticket, taxonomy }: { ticket: Ticket; taxonomy: DashboardData['filterOptions'] }) {
  const [items, setItems] = useState<RoutingFeedbackRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [serviceFeedback, setServiceFeedback] = useState<'ACCEPTED' | 'CORRECTED'>('ACCEPTED')
  const [correctedTargetService, setCorrectedTargetService] = useState('')
  const hasOperatorDecision = ticket.confirmedDecisionAvailable === true || ticket.latestDecision !== undefined
  const serviceOptions = useMemo(
    () => Array.from(new Map(taxonomy.services.map((item) => [item.id, item])).values()),
    [taxonomy.services],
  )
  const correctedOptions = useMemo(
    () => serviceOptions.filter((item) => item.label.toLocaleLowerCase() !== ticket.service.toLocaleLowerCase()),
    [serviceOptions, ticket.service],
  )

  useEffect(() => {
    let active = true
    setLoading(true)
    setError('')
    setNotice('')
    loadRoutingFeedback(ticket.id)
      .then((records) => { if (active) setItems(records) })
      .catch((requestError: unknown) => {
        if (active) setError(requestError instanceof Error ? requestError.message : 'ошибка API')
      })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [ticket.id])

  useEffect(() => {
    if (!correctedTargetService || !correctedOptions.some((item) => item.id === correctedTargetService)) {
      setCorrectedTargetService(correctedOptions[0]?.id ?? '')
    }
  }, [correctedOptions, correctedTargetService])

  const saveFeedback = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!hasOperatorDecision) return
    setSaving(true)
    setError('')
    setNotice('')
    try {
      const record = await submitRoutingFeedback(
        ticket.id,
        serviceFeedback,
        serviceFeedback === 'CORRECTED' ? correctedTargetService : undefined,
      )
      setItems((current) => [record, ...current])
      setNotice('Симуляция сохранена для контролируемой офлайн-проверки. Правила маршрутизации не изменены.')
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'ошибка API')
    } finally {
      setSaving(false)
    }
  }

  return <section className="detail-section routing-feedback-panel" aria-label={translateUi("Обратная связь по маршрутизации")}>
    <div className="section-inline-heading">
      <div className="field-label">{translateUi("Ответ службы")}</div>
      <span className="routing-simulation-badge">DEMO SIMULATION</span>
    </div>
    <p className="panel-note">{translateUi("Внешний канал службы не подключён. Здесь сохраняется только демонстрационная симуляция, привязанная к решению оператора; она не меняет рабочие правила и не запускает обучение.")}</p>
    <dl className="routing-feedback-routes">
      <div><dt>{translateUi("Рекомендация Pulse")}</dt><dd>{ticket.recommendedService ?? 'Не предоставлена'}</dd></div>
      <div><dt>{translateUi("Подтверждённая служба оператора")}</dt><dd>{hasOperatorDecision ? ticket.service : 'Сначала зафиксируйте решение оператора'}</dd></div>
    </dl>
    {hasOperatorDecision && <form className="routing-feedback-form" onSubmit={(event) => void saveFeedback(event)}>
      <label>{translateUi("Симулированный ответ")}
        <select value={serviceFeedback} onChange={(event) => setServiceFeedback(event.target.value as 'ACCEPTED' | 'CORRECTED')}>
          <option value="ACCEPTED">{translateUi("Маршрут принят службой")}</option>
          <option value="CORRECTED">{translateUi("Служба направила в другое подразделение")}</option>
        </select>
      </label>
      {serviceFeedback === 'CORRECTED' && <label>{translateUi("Исправленный адресат")}
        <select aria-label={translateUi("Исправленный адресат службы")} value={correctedTargetService} onChange={(event) => setCorrectedTargetService(event.target.value)} required>
          {correctedOptions.map((item) => <option key={item.id} value={item.id}>{translateUi(item.label)}</option>)}
        </select>
      </label>}
      <button className="button button-secondary" type="submit" disabled={saving || (serviceFeedback === 'CORRECTED' && !correctedTargetService)}>
        {saving ? 'Сохраняем…' : 'Сохранить симуляцию'}
      </button>
    </form>}
    {!hasOperatorDecision && <p className="panel-note">{translateUi("Сначала подтвердите или исправьте службу в решении оператора. Служебная обратная связь без operator-confirmed route не принимается.")}</p>}
    {loading && <p className="panel-note" role="status">{translateUi("Загружаю историю симуляций…")}</p>}
    {error && <p className="routing-feedback-error" role="alert">{translateUi("Не удалось загрузить или сохранить обратную связь:")} {error}</p>}
    {notice && <p className="routing-feedback-notice" role="status">{notice}</p>}
    {!loading && !error && items.length === 0 && <p className="panel-note">{translateUi("Записей симуляции пока нет.")}</p>}
    {items.length > 0 && <div className="routing-feedback-history" aria-label={translateUi("История симуляций")}>
      {items.map((item) => <article className="routing-feedback-record" key={item.id}>
        <div><strong>{item.serviceFeedback === 'ACCEPTED' ? 'Маршрут принят' : 'Маршрут исправлен'}</strong><span>{item.sourceSystem} · {item.actorUserId} · {item.createdAt}</span></div>
        <p>{translateUi("Рекомендация:")} {item.originalRouteRecommendation} {translateUi("· подтверждено оператором:")} {item.operatorConfirmedRoute}</p>
        {item.correctedTargetService && <p>{translateUi("Исправленный адресат:")} {item.correctedTargetService}</p>}
        <small>{translateUi("Статус: ожидает контролируемой офлайн-проверки. Production-правила не изменены.")}</small>
      </article>)}
    </div>}
  </section>
}

function ContextHandoffPanel({ ticket, hasConfirmedRoute }: { ticket: Ticket; hasConfirmedRoute: boolean }) {
  const requestSequence = useRef(0)
  const [handoffPackage, setHandoffPackage] = useState<ContextHandoffPackage | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  useEffect(() => {
    requestSequence.current += 1
    setHandoffPackage(null)
    setLoading(false)
    setError('')
    setNotice('')
  }, [
    ticket.id,
    ticket.originalText,
    ticket.regionId,
    ticket.region,
    ticket.createdAt,
    ticket.confirmedDecisionAvailable,
    ticket.latestDecision?.createdAt,
    ticket.latestDecision?.service,
  ])

  const preparePackage = async () => {
    const requestId = requestSequence.current + 1
    requestSequence.current = requestId
    setLoading(true)
    setError('')
    setNotice('')
    try {
      const result = await loadContextHandoffPackage(ticket.id)
      if (requestSequence.current === requestId) setHandoffPackage(result)
    } catch (requestError) {
      if (requestSequence.current === requestId) {
        setError(requestError instanceof Error ? requestError.message : 'ошибка API')
      }
    } finally {
      if (requestSequence.current === requestId) setLoading(false)
    }
  }

  const copyPackage = async () => {
    if (!handoffPackage) return
    setError('')
    setNotice('')
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Буфер обмена недоступен в этом браузере')
      await navigator.clipboard.writeText(JSON.stringify(handoffPackage, null, 2))
      setNotice('Пакет скопирован локально. Отправка или изменение обращения не выполнялись.')
    } catch (copyError) {
      setError(copyError instanceof Error ? copyError.message : 'не удалось скопировать пакет')
    }
  }

  return <section className="detail-section context-handoff-panel" aria-label={translateUi("Пакет контекстной передачи")}>
    <div className="section-inline-heading">
      <div className="field-label">{translateUi("Пакет передачи с контекстом")}</div>
      <span className="handoff-version">context-handoff.v1</span>
    </div>
    <p className="panel-note">{translateUi("Пакет собирается только из полей обращения и подтверждённого решения. Исходный текст остаётся доступен отдельно; ничего не отправляется и карточка не меняется.")}</p>
    <button className="button button-secondary" type="button" onClick={() => void preparePackage()} disabled={!hasConfirmedRoute || loading}>
      {loading ? 'Собираем…' : handoffPackage ? 'Обновить пакет' : 'Подготовить пакет'}
    </button>
    {!hasConfirmedRoute && <p className="panel-note">{translateUi("Сначала сохраните operator-confirmed route. Рекомендации модели сами по себе не передаются.")}</p>}
    {error && <p className="routing-feedback-error" role="alert">{translateUi("Не удалось подготовить или скопировать пакет:")} {error}</p>}
    {notice && <p className="routing-feedback-notice" role="status">{notice}</p>}
    {handoffPackage && <div className="handoff-package" aria-label={translateUi("Сформированный пакет")}>
      <dl className="handoff-fields">
        <div><dt>{translateUi("Что произошло")}</dt><dd>{handoffPackage.whatHappened}</dd></div>
        <div><dt>{translateUi("Где")}</dt><dd>
          <span>{handoffPackage.where.regionName} ({handoffPackage.where.regionId})</span>
          {handoffPackage.where.district && <span>{translateUi("Район:")} {handoffPackage.where.district}</span>}
          {handoffPackage.where.address && <span>{translateUi("Адрес:")} {handoffPackage.where.address}</span>}
          {handoffPackage.where.object && <span>{translateUi("Объект:")} {handoffPackage.where.object}</span>}
        </dd></div>
        <div><dt>{translateUi("Когда / с какого момента")}</dt><dd>
          <span>{translateUi("Обращение зарегистрировано:")} {handoffPackage.whenOrSince.receivedAt}</span>
          <span>{translateUi("Начало события:")} {handoffPackage.whenOrSince.reportedSince ?? 'не указано в структурированных данных'}</span>
        </dd></div>
        <div><dt>{translateUi("Масштаб")}</dt><dd>{handoffPackage.scale ?? 'не указан в структурированных данных'}</dd></div>
      </dl>
      <div className="handoff-fact-group">
        <strong>{translateUi("Подтверждено решением оператора")}</strong>
        {handoffPackage.confirmedFacts.length > 0
          ? <ul>{handoffPackage.confirmedFacts.map((fact) => <li key={`${fact.evidence.field}:${fact.label}`}><span>{fact.label}: {fact.value}</span><small>{fact.evidence.sourceType}/{fact.evidence.recordId}.{fact.evidence.field}</small></li>)}</ul>
          : <p className="panel-note">{translateUi("Подтверждённые классификационные поля не сохранены.")}</p>}
      </div>
      <div className="handoff-fact-group">
        <strong>{translateUi("Неизвестно или требует отдельной проверки")}</strong>
        <ul>{handoffPackage.unknownFacts.map((fact) => <li key={fact}>{fact}</li>)}</ul>
      </div>
      <dl className="handoff-fields">
        <div><dt>{translateUi("Почему выбран маршрут")}</dt><dd>
          <span>{translateUi("Снимок рекомендации:")} {handoffPackage.route.recommendedService ?? 'не сохранён'}</span>
          <span>{translateUi("Подтверждено оператором:")} {handoffPackage.route.confirmedService}</span>
          <span>{handoffPackage.route.explanation}</span>
          <small>{handoffPackage.route.provenanceSource}{handoffPackage.route.provenanceVersion ? ` · v${handoffPackage.route.provenanceVersion}` : ''} {translateUi("· решение")} {handoffPackage.route.operatorDecisionId}</small>
        </dd></div>
        {handoffPackage.linkedAttachmentCount > 0 && <div><dt>{translateUi("Вложения")}</dt><dd>{handoffPackage.linkedAttachmentCount} {translateUi("доступны отдельно в исходной карточке; содержимое не копировалось.")}</dd></div>}
      </dl>
      <div className="handoff-fact-group">
        <strong>{translateUi("Ссылки на источники")}</strong>
        <ul>{handoffPackage.evidenceReferences.map((reference) => <li key={`${reference.sourceType}:${reference.recordId}:${reference.field}`}>
          <span>{reference.label}</span><small>{reference.sourceType}/{reference.recordId}.{reference.field}</small>
        </li>)}</ul>
      </div>
      <button className="button button-quiet" type="button" onClick={() => void copyPackage()}>{translateUi("Скопировать JSON пакета")}</button>
    </div>}
  </section>
}

function RelatedTicketDetailPanel({ state, taxonomy, onClose, onRetry }: { state: RelatedTicketPanelState; taxonomy: DashboardData['filterOptions']; onClose: () => void; onRetry: () => void }) {
  const detail = state.detail
  const decision = detail?.latestDecision
  const comparisons = detail ? compareRelatedTicketContext(state.currentTicket, detail) : []
  const decisionTopic = decision
    ? decision.confirmedTopicLabel ?? taxonomy.topics.find((item) => item.id === decision.confirmedTopicId)?.label ?? decision.confirmedTopicId
    : ''
  const decisionAction = decision?.action.toLowerCase() === 'confirm'
    ? 'Подтверждение предложенной темы'
    : decision?.action.toLowerCase() === 'correct'
      ? 'Исправление темы оператором'
      : decision?.action
  const memoryLabel = detail
    ? resolutionMemoryLabel(Boolean(decision), detail.outcomeVerification?.state)
    : ''

  return <div className="related-ticket-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <section className="related-ticket-panel" role="dialog" aria-modal="true" aria-labelledby="related-ticket-title">
      <header className="related-ticket-header">
        <div><span className="field-label">{translateUi("Контекст для проверки")}</span><h2 id="related-ticket-title">{translateUi("Связанное обращение")}</h2><span className="related-ticket-id">{state.ticketId}</span></div>
        <button className="icon-button" aria-label={translateUi("Закрыть связанное обращение")} onClick={onClose}><Icon name="close" size={18} /></button>
      </header>
      {state.loading ? <div className="related-ticket-state" role="status">{translateUi("Загружаю карточку обращения…")}</div> : state.error ? <div className="related-ticket-state" role="alert"><p>{translateUi("Не удалось загрузить карточку:")} {state.error}</p><button className="button button-secondary" onClick={onRetry}>{translateUi("Повторить")}</button></div> : detail && <div className="related-ticket-scroll">
        <section className="related-ticket-section">
          <div className="field-label">{translateUi("Оригинальный текст")}</div>
          <p className="related-ticket-original">«{detail.originalText}»</p>
          <p className="source-line">{detail.channel} · {translateUi(detail.region)}{detail.externalRef && ` · № ${detail.externalRef}`}</p>
        </section>
        <section className="related-ticket-section">
          <div className="field-label">{translateUi("Тема и статус в записи")}</div>
          <dl className="related-ticket-meta">
            <div><dt>{translateUi("Создано")}</dt><dd>{detail.createdAt}</dd></div>
            <div><dt>{translateUi("Закрыто")}</dt><dd>{detail.closedAt ?? 'Время закрытия не указано'}</dd></div>
            <div><dt>{translateUi("Тема")}</dt><dd>{translateUi(detail.topic)}</dd></div>
            <div><dt>{translateUi("Статус записи")}</dt><dd>{detail.status}</dd></div>
          </dl>
        </section>
        <section className="related-ticket-section">
          <div className="field-label">{translateUi("Последнее доступное действие оператора")}</div>
          {decision ? <dl className="related-ticket-meta">
            <div><dt>{translateUi("Действие")}</dt><dd>{decisionAction || 'Действие оператора зафиксировано'}</dd></div>
            <div><dt>{translateUi("Подтверждённая тема")}</dt><dd>{decisionTopic || 'Тема не указана'}</dd></div>
            {decision.service && <div><dt>{translateUi("Служба в решении")}</dt><dd>{decision.service}</dd></div>}
            {decision.priority && <div><dt>{translateUi("Приоритет в решении")}</dt><dd>{decision.priority}</dd></div>}
            {decision.createdAt && <div><dt>{translateUi("Время действия")}</dt><dd>{decision.createdAt}</dd></div>}
          </dl> : <p className="panel-note">{translateUi("Операторское действие не зафиксировано. Исход обращения неизвестен.")}</p>}
          <p className="resolution-memory-state" role="status">{memoryLabel}</p>
          {detail.outcomeVerificationError
            ? <p className="routing-feedback-error" role="alert">{translateUi("Не удалось загрузить outcome; закрытый статус нельзя трактовать как VERIFIED:")} {detail.outcomeVerificationError}</p>
            : detail.outcomeVerification?.latest && <p className="panel-note">{detail.outcomeVerification.latest.sourceSystem} · {detail.outcomeVerification.latest.channel} {translateUi("· оператор")} {detail.outcomeVerification.latest.actorUserId} · {detail.outcomeVerification.latest.createdAt}</p>}
        </section>
        <section className="related-ticket-section">
          <div className="field-label">{translateUi("Проверяемые признаки сходства")}</div>
          {state.matchedFactors.length ? <div className="related-factor-list">{state.matchedFactors.map((factor) => <span className="relation-badge" key={factor}>{factor}</span>)}</div> : <p className="panel-note">{translateUi("Совпадение по теме или региону не подтверждено доступными полями.")}</p>}
        </section>
        <p className="related-ticket-note">{translateUi("Это возможное сходство для проверки оператором. Оно не подтверждает, что объект или проблема те же, и не создаёт связь автоматически.")}</p>
        <section className="related-ticket-section key-comparison" aria-label={translateUi("Фактические сходства и различия")}>
          <div>
            <div className="field-label">{translateUi("Фактическое сравнение")}</div>
            <p className="panel-note">{translateUi("Текущая запись сопоставлена со связанной по структурированным полям. Это сравнение не оценивает правильность прошлого решения.")}</p>
          </div>
          <div className="key-comparison-group">
            <strong>{translateUi("ПОХОЖЕ")}</strong>
            {comparisons.filter((item) => ['topic', 'object_type', 'region'].includes(item.key)).map((item) => <ComparisonFact key={item.key} fact={item} />)}
          </div>
          <div className="key-comparison-group">
            <strong>{translateUi("ОТЛИЧАЕТСЯ")}</strong>
            {comparisons.filter((item) => ['created_at', 'closed_at', 'source_status', 'confirmed_topic', 'confirmed_service', 'confirmed_action', 'confirmed_priority'].includes(item.key)).map((item) => <ComparisonFact key={item.key} fact={item} />)}
            <div className="comparison-fact comparison-fact-unavailable"><span>{translateUi("Адрес")}</span><small>{translateUi("Не сравнивается: точный адрес недоступен в безопасном контуре.")}</small></div>
          </div>
          <p className="panel-note">{translateUi("Тип объекта не передан как безопасное структурированное поле. Неизвестные значения не считаются совпадением или различием.")}</p>
        </section>
      </div>}
    </section>
  </div>
}

function ComparisonFact({ fact }: { fact: ReturnType<typeof compareRelatedTicketContext>[number] }) {
  const stateLabel = fact.state === 'same' ? 'совпадает' : fact.state === 'different' ? 'различается' : 'недостаточно данных'
  const stateClass = `comparison-fact-${fact.state}`
  const currentLabel = fact.currentSource ? `Текущее (${fact.currentSource})` : 'Текущее'
  const relatedLabel = fact.relatedSource ? `связанное (${fact.relatedSource})` : 'связанное'
  return (
    <div className={`comparison-fact ${stateClass}`}>
      <div><span>{fact.label}</span><small>{stateLabel}</small></div>
      <p>{currentLabel}: {fact.currentValue ?? 'не указано'} · {relatedLabel}: {fact.relatedValue ?? 'не указано'}</p>
    </div>
  )
}

function Confidence({ value, state, compact = false, available = true }: { value: number; state?: Ticket['confidenceState']; compact?: boolean; available?: boolean }) {
  const confidenceState = normalizeConfidenceState(state, value, available)
  return <div className={`confidence ${compact ? 'confidence-compact' : ''}`}><div className="confidence-track"><span style={{ width: available ? `${value * 100}%` : '0%' }} /></div><strong>{available ? formatPercent(value) : '—'}</strong>{!compact && <small>{translateUi(available ? confidenceStateLabel(confidenceState) : 'нет данных')}</small>}</div>
}

function PriorityBadge({ priority }: { priority: Priority }) {
  const level = priority === 'Критический' ? 'critical' : priority === 'Высокий' ? 'high' : priority === 'Средний' ? 'medium' : priority === 'Низкий' ? 'low' : 'unknown'
  return <span className={`priority-badge priority-${level}`}><span />{translateUi(priority)}</span>
}

function RoutingProvenance({ provenance, fallbackReason }: { provenance?: RuleProvenance; fallbackReason: string }) {
  const value = provenance ?? { source: 'MANUAL' as const, version: null, reason: fallbackReason }
  const sourceLabel = value.source === 'OFFICIAL'
    ? 'Официальное правило'
    : value.source === 'LABEL_HISTORY'
      ? 'Историческая метка'
      : 'Ручной источник / решение'
  return (
    <div className={`routing-provenance routing-source-${value.source.toLowerCase()}`}>
      <span>{translateUi(value.reason)}</span>
      <details><summary>{translateUi(sourceLabel)}{value.version ? ` · ${translateUi('версия')} ${value.version}` : ''}</summary><small>{value.source} · {value.factsUsed?.length ? `${translateUi('Использованные факты:')} ${value.factsUsed.map(formatExplainabilityFact).join(' · ')}` : translateUi('Факты выбора не сохранены')}</small></details>
    </div>
  )
}
