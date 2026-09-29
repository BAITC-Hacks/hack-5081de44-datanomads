import { formatUiDateTime, localeTag, translateUi } from '../uiSettings'
import { useCallback, useEffect, useState } from 'react'
import { closeLearningCycle, completeModelRollout, createLearningCycle, loadCandidateEvaluation, loadModelRollouts, promoteCandidate, registerLearningCandidate, rejectCandidate, reviewDriftTrigger, rollbackModelRollout } from '../api/client'
import type { BackendModelRollout, CandidateEvaluation } from '../api/client'
import type { DashboardData, LearningCycle, ModelStatus } from '../types'
import { NoData, PanelHeading } from '../components/AnalyticsPrimitives'

function evaluationMetric(value: unknown, percentage = false): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return translateUi('недостаточно данных')
  return percentage ? `${(value * 100).toFixed(1)}%` : value.toFixed(3)
}

function evaluationGateLabel(key: string): string {
  const labels: Record<string, string> = {
    promotion_policy: 'Версия promotion policy поддерживается',
    offline_sample_count: 'Размер offline выборки',
    shadow_sample_count: 'Размер shadow выборки',
    macro_f1_non_inferiority: 'Macro-F1 candidate относительно production',
    critical_class_regressions: 'Регрессии по классам',
    shadow_correction_rate_delta: 'Изменение correction rate в shadow',
    candidate_shadow_inference_failures: 'Ошибки candidate inference',
    synthetic_evidence: 'Происхождение evidence',
    candidate_evaluation_job: 'Фоновая оценка candidate',
  }
  return labels[key] ?? key
}

function evaluationGateStatus(status: string): string {
  switch (status) {
    case 'PASSED': return 'Пройден'
    case 'FAILED': return 'Не пройден'
    case 'PENDING': return 'Выполняется'
    default: return 'Недостаточно данных'
  }
}

function promotionThresholdLabel(key: string): string {
  const labels: Record<string, string> = {
    minimum_offline_samples: 'Минимум offline записей',
    minimum_shadow_samples: 'Минимум shadow решений',
    maximum_macro_f1_regression: 'Допустимое снижение macro-F1',
    maximum_class_f1_regression: 'Допустимое снижение F1 класса',
    maximum_shadow_correction_rate_delta: 'Допустимый рост correction rate',
    maximum_shadow_inference_failures: 'Допустимые ошибки inference',
  }
  return labels[key] ?? key
}

function candidateEvaluationReady(evaluation: CandidateEvaluation | null | undefined): boolean {
  return evaluation?.status === 'COMPLETED'
    && evaluation.decision === 'PENDING_HUMAN_DECISION'
    && !evaluation.synthetic
    && evaluation.gates.length > 0
    && evaluation.gates.every((gate) => gate.status === 'PASSED')
}

export function CleanLearningPage({ learning, onRefresh, onToast }: { learning: LearningCycle; onRefresh: () => Promise<void>; onToast: (message: string) => void }) {
  const [evaluation, setEvaluation] = useState<Awaited<ReturnType<typeof loadCandidateEvaluation>> | null>(null)
  const [evaluationError, setEvaluationError] = useState<string | null>(null)
  const [busy, setBusy] = useState<'create' | 'close' | 'evaluation' | 'candidate-add' | 'promote' | 'reject' | null>(null)
  const [note, setNote] = useState('')
  const [candidateVersionInput, setCandidateVersionInput] = useState('')
  const [selectedCandidateVersion, setSelectedCandidateVersion] = useState('')

  useEffect(() => {
    let active = true
    let timer: number | undefined
    setEvaluation(null)
    setEvaluationError(null)
    if (!['EVALUATE', 'DECISION'].includes(learning.stage) || learning.id === 'нет данных') return () => { active = false }
    const load = async () => {
      try {
        const result = await loadCandidateEvaluation()
        if (!active) return
        setEvaluation(result)
        setSelectedCandidateVersion((current) => result.candidate_comparisons?.some((candidate) => candidate.candidate_model_version === current) ? current : '')
        const candidateJobsPending = result.candidate_comparisons?.some((candidate) =>
          candidate.status === 'EVALUATING' || candidate.job_state === 'QUEUED' || candidate.job_state === 'RUNNING',
        )
        if (learning.stage === 'DECISION' && (result.status === 'PENDING' || candidateJobsPending)) {
          timer = window.setTimeout(() => void load(), 4000)
        }
      } catch (error: unknown) {
        if (active) setEvaluationError(error instanceof Error ? error.message : 'Оценка пока недоступна')
      }
    }
    void load()
    return () => {
      active = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [learning.id, learning.stage, learning.updatedAt])

  const runAction = async (action: 'create' | 'close' | 'evaluation' | 'candidate-add' | 'promote' | 'reject') => {
    setBusy(action)
    try {
      if (action === 'create') {
        await createLearningCycle()
        onToast('Открыт новый цикл COLLECT с настроенным окном сбора')
      } else if (action === 'close') {
        const result = await closeLearningCycle(learning.id)
        onToast(result.state === 'INSUFFICIENT_FEEDBACK'
          ? `Сбор закрыт: ${result.cycle.feedback_count}/${result.cycle.min_feedback_count} валидных записей, кандидат не создан`
          : result.state === 'DECISION'
            ? 'Окно shadow evaluation закрыто; Data/ML evaluation поставлена в очередь'
            : 'Сбор обратной связи закрыт; обучение поставлено в очередь')
      } else if (action === 'evaluation') {
        setEvaluation(await loadCandidateEvaluation())
        onToast('Оценка candidate перечитана из backend')
      } else if (action === 'candidate-add') {
        const candidateVersion = candidateVersionInput.trim()
        const result = await registerLearningCandidate(learning.id, candidateVersion)
        setCandidateVersionInput('')
        onToast(`Кандидат ${result.candidate_model_version} зарегистрирован до начала общего shadow-окна`)
      } else if (action === 'promote' || action === 'reject') {
        if (!window.confirm(action === 'promote' ? 'Запустить candidate на 10% трафика для canary-наблюдения?' : 'Отклонить candidate?')) return
        if (action === 'promote') {
          await promoteCandidate(note, selectedCandidateVersion)
        } else {
          await rejectCandidate(note)
        }
        onToast(action === 'promote' ? 'Canary запущен на 10% трафика; production pointer не изменён' : 'Candidate отклонён; production не изменён')
        setNote('')
      }
      await onRefresh()
    } catch (error) {
      onToast(`Не удалось выполнить действие: ${error instanceof Error ? error.message : 'ошибка API'}`)
    } finally {
      setBusy(null)
    }
  }

  if (learning.id === 'нет данных') return <div className="analytics-page"><section className="panel"><NoData message={translateUi("Активного цикла обучения нет.")} /><button className="button button-primary" disabled={busy !== null} onClick={() => void runAction('create')}>{busy === 'create' ? 'Создаём…' : 'Открыть цикл COLLECT'}</button></section></div>

  const candidateComparisons = evaluation?.candidate_comparisons ?? []
  const evaluationForCandidate = (candidateVersion: string): CandidateEvaluation | null => {
    const comparison = candidateComparisons.find((candidate) => candidate.candidate_model_version === candidateVersion)
    if (comparison?.evaluation) return comparison.evaluation
    return evaluation?.candidate_model_version === candidateVersion ? evaluation : null
  }
  const detailedEvaluation = selectedCandidateVersion
    ? evaluationForCandidate(selectedCandidateVersion) ?? evaluation
    : evaluation
  const offlineStatus = detailedEvaluation?.offline_evaluation.status
  const readyToReview = learning.stage === 'DECISION'
    && Boolean(selectedCandidateVersion)
    && candidateEvaluationReady(evaluationForCandidate(selectedCandidateVersion))
  const candidateF1 = detailedEvaluation?.offline_evaluation.metrics.macro_f1
  const productionF1 = detailedEvaluation?.baseline_evaluation.metrics.macro_f1
  const classMetrics = detailedEvaluation?.offline_evaluation.metrics.per_class_f1 ?? {}
  const productionClassMetrics = detailedEvaluation?.offline_evaluation.metrics.production_per_class_f1
    ?? detailedEvaluation?.baseline_evaluation.metrics.per_class_f1
    ?? {}
  const classChanges = detailedEvaluation?.offline_evaluation.metrics.per_class_changes ?? {}
  const classSupport = detailedEvaluation?.offline_evaluation.metrics.per_class_support ?? {}
  const classLabels = [...new Set([
    ...Object.keys(classMetrics),
    ...Object.keys(productionClassMetrics),
    ...Object.keys(classChanges),
  ])].sort()
  const criticalRegressions = detailedEvaluation
    ? [...new Set([
      ...detailedEvaluation.offline_evaluation.critical_regressions,
      ...detailedEvaluation.shadow_evaluation.critical_regressions,
    ])]
    : []
  const collectEndTimestamp = Date.parse(learning.collectEndsAt)
  const collectEndReached = Number.isFinite(collectEndTimestamp) && collectEndTimestamp <= Date.now()
  const canCloseCollect = learning.manualCloseEnabled || collectEndReached
  const evaluationEndTimestamp = learning.evaluationEndsAt ? Date.parse(learning.evaluationEndsAt) : Number.NaN
  const evaluationEndReached = Number.isFinite(evaluationEndTimestamp) && evaluationEndTimestamp <= Date.now()
  const canCloseEvaluation = learning.manualCloseEnabled || evaluationEndReached
  const collectWindowLabel = learning.collectStartedAt && learning.collectEndsAt
    ? `${new Date(learning.collectStartedAt).toLocaleString(localeTag())} — ${new Date(learning.collectEndsAt).toLocaleString(localeTag())}`
    : translateUi('не задано')
  const evaluationWindowLabel = learning.evaluationStartedAt && learning.evaluationEndsAt
    ? `${new Date(learning.evaluationStartedAt).toLocaleString(localeTag())} — ${new Date(learning.evaluationEndsAt).toLocaleString(localeTag())}`
    : learning.evaluationStartedAt
      ? `${new Date(learning.evaluationStartedAt).toLocaleString(localeTag())} — ${translateUi('конец не задан')}`
      : translateUi('ещё не началось')
  return <div className="analytics-page">
    <section className="panel">
      <PanelHeading title={`${translateUi('Цикл')} ${learning.id}`} />
      <div className="learning-summary-grid">
      <div className="dataset-stat"><span>{translateUi("Состояние")}</span><strong>{learning.stage}</strong></div>
      <div className="dataset-stat"><span>{translateUi("Обратная связь")}</span><strong>{learning.feedbackCount}</strong></div>
      <div className="dataset-stat"><span>{translateUi("Порог обратной связи")}</span><strong>{learning.minFeedbackCount}</strong></div>
      <div className="dataset-stat"><span>{translateUi("Окно COLLECT")}</span><strong>{collectWindowLabel}</strong></div>
      <div className="dataset-stat"><span>{translateUi("Окно shadow evaluation")}</span><strong>{evaluationWindowLabel}</strong></div>
      <div className="dataset-stat"><span>Shadow predictions</span><strong>{learning.shadowPredictionCount ?? 0}</strong></div>
      <div className="dataset-stat"><span>{translateUi("Ошибки shadow inference")}</span><strong>{learning.shadowInferenceFailures ?? 0}</strong></div>
      <div className="dataset-stat"><span>{translateUi("Связанные решения оператора")}</span><strong>{learning.shadowOperatorDecisionCount ?? 0}</strong></div>
      <div className="dataset-stat"><span>Blind A/B signal</span><strong>{translateUi(learning.blindAbEnabled ? 'включён' : 'отключён')}</strong></div>
      <div className="dataset-stat"><span>Production baseline</span><strong>{learning.productionModelVersion ?? translateUi('не зафиксирована')}</strong></div>
      <div className="dataset-stat"><span>Frozen evaluation dataset</span><strong>{learning.frozenEvaluationDatasetVersion ?? translateUi('не настроен')}</strong></div>
      {learning.candidateDatasetChecksum && <div className="dataset-stat"><span>Candidate dataset SHA-256</span><strong>{learning.candidateDatasetChecksum}</strong></div>}
      <div className="dataset-stat"><span>{translateUi("Датасет")}</span><strong>{learning.dataset}</strong></div>
      <div className="dataset-stat"><span>{translateUi("Кандидат")}</span><strong>{learning.candidate}</strong></div>
      </div>
      {learning.decisionNote && <p className="panel-note learning-decision-note">{translateUi("Решение:")} {learning.decisionNote}</p>}
      <div className="learning-actions" aria-label={translateUi("Действия reviewer")}>
        {learning.stage === 'COLLECT' && learning.id !== 'нет данных' && <button className="button button-primary" disabled={busy !== null || !canCloseCollect} onClick={() => void runAction('close')}>{translateUi(busy === 'close' ? 'Закрываем…' : 'Закрыть цикл')}</button>}
        {learning.stage === 'COLLECT' && learning.id !== 'нет данных' && !canCloseCollect && <p className="panel-note">{translateUi("Сбор завершится автоматически по окончании окна COLLECT.")}</p>}
        {learning.stage === 'EVALUATE' && learning.id !== 'нет данных' && <button className="button button-primary" disabled={busy !== null || !canCloseEvaluation} onClick={() => void runAction('close')}>{translateUi(busy === 'close' ? 'Закрываем…' : 'Закрыть окно evaluation')}</button>}
        {learning.stage === 'EVALUATE' && learning.id !== 'нет данных' && !canCloseEvaluation && <p className="panel-note">{translateUi("Shadow evaluation завершится автоматически по окончании окна.")}</p>}
        {['CANARY', 'MONITORING'].includes(learning.stage) && <p className="panel-note">{translateUi("Candidate обслуживает 10% обращений в canary. Production pointer пока прежний; наблюдайте evidence на странице «Реестр моделей».")}</p>}
        {!learning.blindAbEnabled && <p className="panel-note">{translateUi("Слепое сравнение A/B отключено; предпочтения не собираются.")}</p>}
        {['PROMOTED', 'REJECTED', 'INSUFFICIENT_FEEDBACK', 'DATASET_BUILD_FAILED', 'TRAINING_FAILED'].includes(learning.stage) && <button className="button button-primary" disabled={busy !== null} onClick={() => void runAction('create')}>{translateUi(busy === 'create' ? 'Создаём…' : 'Открыть цикл COLLECT')}</button>}
        {['EVALUATE', 'DECISION'].includes(learning.stage) && <button className="button button-secondary" disabled={busy !== null} onClick={() => void runAction('evaluation')}>{translateUi(busy === 'evaluation' ? 'Читаем…' : 'Показать evaluation')}</button>}
        {['COLLECT', 'TRAINING'].includes(learning.stage) && <div className="learning-candidate-register">
          <label htmlFor="learning-candidate-version">{translateUi("Добавить версию из Model Registry")}</label>
          <input id="learning-candidate-version" className="learning-note" value={candidateVersionInput} maxLength={200} onChange={(event) => setCandidateVersionInput(event.target.value)} placeholder="candidate model version" disabled={busy !== null} />
          <button className="button button-secondary" disabled={busy !== null || candidateVersionInput.trim().length === 0} onClick={() => void runAction('candidate-add')}>{translateUi(busy === 'candidate-add' ? 'Добавляем…' : 'Добавить candidate')}</button>
        </div>}
        {learning.stage === 'DECISION' && <>
          <input className="learning-note" aria-label={translateUi("Комментарий reviewer")} placeholder={translateUi("Комментарий к решению (необязательно)")} value={note} onChange={(event) => setNote(event.target.value)} disabled={busy !== null} />
          <button className="button button-primary" disabled={busy !== null || !readyToReview} onClick={() => void runAction('promote')}>{translateUi(busy === 'promote' ? 'Запускаем canary…' : 'Запустить canary на 10%')}</button>
          <button className="button button-quiet" disabled={busy !== null} onClick={() => void runAction('reject')}>{translateUi(busy === 'reject' ? 'Отклоняем…' : 'Reject')}</button>
        </>}
      </div>
    </section>
    {['EVALUATE', 'DECISION'].includes(learning.stage) && <section className="panel learning-evaluation">
      <PanelHeading title="Evaluation candidate" />
      {evaluationError && <p className="panel-note">{translateUi("Оценка пока недоступна:")} {evaluationError}{translateUi(". Повторите запрос после восстановления backend.")}</p>}
      {!evaluation && !evaluationError && <p className="panel-note">{translateUi("Загружаем evaluation из backend…")}</p>}
      {evaluation && <>
        <div className="learning-candidate-comparison">
          <h3>{translateUi("Сравнение кандидатов на одном evidence set")}</h3>
          <p className="panel-note">Frozen dataset: {evaluation.evaluation_set?.dataset_version ?? evaluation.offline_evaluation.dataset_version}; {evaluation.evaluation_set?.sample_count ?? evaluation.offline_evaluation.sample_count} {translateUi("общих записей. Каждый кандидат оценивается отдельно.")}</p>
          {candidateComparisons.length > 0
            ? <div className="learning-class-table-wrap"><table className="learning-class-table learning-candidate-table">
              <thead><tr><th>{translateUi("Выбор")}</th><th>{translateUi("Версия")}</th><th>Candidate dataset</th><th>{translateUi("Статус")}</th><th>Macro-F1</th><th>Offline n</th><th>Shadow correction Δ</th><th>Policy gates</th></tr></thead>
              <tbody>{candidateComparisons.map((candidate) => {
                const candidateEvaluation = evaluationForCandidate(candidate.candidate_model_version)
                const candidateReady = candidateEvaluationReady(candidateEvaluation)
                return <tr key={candidate.candidate_model_version}>
                  <td><input
                    type="radio"
                    name="promotion-candidate"
                    aria-label={`Выбрать ${candidate.candidate_model_version} для promotion`}
                    checked={selectedCandidateVersion === candidate.candidate_model_version}
                    disabled={learning.stage !== 'DECISION' || !candidateReady || busy !== null}
                    onChange={() => setSelectedCandidateVersion(candidate.candidate_model_version)}
                  /></td>
                  <th scope="row">{candidate.candidate_model_version}</th>
                  <td>{candidate.candidate_dataset_version ?? candidateEvaluation?.candidate_dataset_version ?? '—'}</td>
                  <td>{candidateEvaluation?.decision ?? candidate.job_state ?? candidate.status}</td>
                  <td>{evaluationMetric(candidateEvaluation?.offline_evaluation.metrics.macro_f1)}</td>
                  <td>{candidateEvaluation?.offline_evaluation.sample_count ?? '—'}</td>
                  <td>{evaluationMetric(candidateEvaluation?.shadow_evaluation.correction_rate_delta, true)}</td>
                  <td>{translateUi(candidateReady ? 'Пройдены' : 'Не готовы')}</td>
                </tr>
              })}</tbody>
            </table></div>
            : <p className="panel-note">{translateUi("Список кандидатов появится после загрузки сравнения из backend.")}</p>}
          {learning.stage === 'DECISION' && !selectedCandidateVersion && <p className="panel-note">{translateUi("Выберите версию с пройденными gates, чтобы активировать promotion.")}</p>}
        </div>
        {detailedEvaluation && <>
        <div className="dataset-stat"><span>{translateUi("Подробные результаты")}</span><strong>{detailedEvaluation.candidate_model_version}</strong></div>
        <div className="dataset-stat"><span>{translateUi("Статус оценки")}</span><strong>{translateUi(detailedEvaluation.status === 'PENDING' ? 'В очереди или выполняется' : detailedEvaluation.status === 'FAILED' ? 'Фоновая оценка завершилась ошибкой' : 'Оценка завершена')}</strong></div>
        <div className="dataset-stat"><span>{translateUi("Решение policy")}</span><strong>{translateUi(detailedEvaluation.decision === 'PENDING_HUMAN_DECISION' ? 'Все gates пройдены; ожидает решения reviewer' : detailedEvaluation.decision === 'FAIL' ? 'Есть проваленные gates' : detailedEvaluation.decision === 'PASS' ? 'Policy пройдена; ожидает действия reviewer' : 'Недостаточно evidence')}</strong></div>
        <div className="dataset-stat"><span>{translateUi("Статус evidence")}</span><strong>{translateUi(String(offlineStatus ?? 'нет данных'))}</strong></div>
        <div className="learning-evaluation-grid">
          <article className="learning-evaluation-metric">
            <span>Candidate macro-F1</span>
            <strong>{evaluationMetric(candidateF1)}</strong>
            <small>{detailedEvaluation.offline_evaluation.sample_count} {translateUi("frozen offline записей")}</small>
          </article>
          <article className="learning-evaluation-metric">
            <span>Production macro-F1</span>
            <strong>{evaluationMetric(productionF1)}</strong>
            <small>{detailedEvaluation.baseline_evaluation.sample_count} {translateUi("offline записей")}</small>
          </article>
          <article className="learning-evaluation-metric">
            <span>Shadow agreement</span>
            <strong>{evaluationMetric(detailedEvaluation.shadow_evaluation.agreement_with_confirmed, true)}</strong>
            <small>{detailedEvaluation.shadow_evaluation.sample_count} {translateUi("подтверждённых решений")}</small>
          </article>
          <article className="learning-evaluation-metric">
            <span>{translateUi("Изменение correction rate")}</span>
            <strong>{evaluationMetric(detailedEvaluation.shadow_evaluation.correction_rate_delta, true)}</strong>
            <small>{translateUi("candidate минус production")}</small>
          </article>
        </div>
        <div className="dataset-stat"><span>Candidate</span><strong>{detailedEvaluation.candidate_model_version}</strong></div>
        <div className="dataset-stat"><span>Baseline production</span><strong>{detailedEvaluation.production_model_version}</strong></div>
        <div className="dataset-stat"><span>Frozen evaluation dataset</span><strong>{detailedEvaluation.offline_evaluation.dataset_version}</strong></div>
        <div className="dataset-stat"><span>Promotion policy</span><strong>{detailedEvaluation.policy_version}</strong></div>
        {Object.entries(detailedEvaluation.promotion_policy.thresholds).length > 0 && <div className="learning-policy-thresholds">
          <h3>{translateUi("Пороги до оценки candidate")}</h3>
          <ul>{Object.entries(detailedEvaluation.promotion_policy.thresholds).map(([key, value]) => <li key={key}>
            <span>{translateUi(promotionThresholdLabel(key))}</span>
            <strong>{key.includes('regression') || key.includes('delta') ? evaluationMetric(value, true) : value}</strong>
          </li>)}</ul>
        </div>}
        <div className="learning-gates">
          <h3>Promotion gates</h3>
          <ul>{detailedEvaluation.gates.map((item) => <li className={`learning-gate learning-gate-${item.status.toLowerCase()}`} key={item.key}>
            <span><strong>{translateUi(evaluationGateLabel(item.key))}</strong>{item.reason && <small>{translateUi(item.reason)}</small>}</span>
            <span className="learning-gate-result">
              <strong>{translateUi(evaluationGateStatus(item.status))}</strong>
              {(item.observed !== undefined || item.threshold !== undefined) && <small>{item.observed ?? '—'} / {item.threshold ?? '—'}</small>}
            </span>
          </li>)}</ul>
        </div>
        <div className="learning-gates">
          <h3>Critical regressions</h3>
          {criticalRegressions.length > 0
            ? <ul>{criticalRegressions.map((name) => <li className="learning-regression" key={name}><strong>{name}</strong></li>)}</ul>
            : offlineStatus === 'COMPLETED' && detailedEvaluation.offline_evaluation.sample_count > 0
              ? <p className="panel-note">{translateUi("Критические регрессии не обнаружены.")}</p>
              : <p className="panel-note">{translateUi("Не оценивались: offline evidence недостаточно.")}</p>}
        </div>
        <div className="learning-gates">
          <h3>Per-class F1</h3>
          {classLabels.length > 0
            ? <div className="learning-class-table-wrap"><table className="learning-class-table">
              <thead><tr><th>{translateUi("Класс")}</th><th>Production</th><th>Candidate</th><th>{translateUi("Изменение")}</th><th>Support</th></tr></thead>
              <tbody>{classLabels.map((label) => <tr key={label}>
                <th scope="row">{label}</th>
                <td>{evaluationMetric(productionClassMetrics[label])}</td>
                <td>{evaluationMetric(classMetrics[label])}</td>
                <td>{evaluationMetric(classChanges[label])}</td>
                <td>{classSupport[label] ?? '—'}</td>
              </tr>)}</tbody>
            </table></div>
            : <p className="panel-note">{translateUi("Per-class metrics появятся после получения достаточного offline evidence.")}</p>}
        </div>
        <div className="dataset-stat"><span>Blind A/B</span><strong>{translateUi(detailedEvaluation.shadow_evaluation.blind_ab === 'ENABLED' ? 'включён' : 'отключён')}</strong></div>
        {detailedEvaluation.synthetic && <p className="panel-note">{translateUi("Evidence синтетический и не допускается для production promotion.")}</p>}
        </>}
      </>}
    </section>}
  </div>
}

export function CleanModelsPage({ models, driftTriggers, canReview, onRefresh, onToast }: { models: ModelStatus[]; driftTriggers: NonNullable<DashboardData['driftTriggers']>; canReview: boolean; onRefresh: () => Promise<void>; onToast: (message: string) => void }) {
  const [busyTrigger, setBusyTrigger] = useState<string | null>(null)
  const [rollouts, setRollouts] = useState<BackendModelRollout[]>([])
  const [rolloutError, setRolloutError] = useState<string | null>(null)
  const [busyRollout, setBusyRollout] = useState<string | null>(null)
  const [rolloutReason, setRolloutReason] = useState('')
  const [busyRollback, setBusyRollback] = useState<string | null>(null)
  const [rollbackReason, setRollbackReason] = useState('')
  const refreshRollouts = useCallback(async () => {
    try {
      const result = await loadModelRollouts()
      setRollouts(result.items)
      setRolloutError(null)
    } catch {
      setRolloutError('Статус canary rollout пока недоступен через API.')
    }
  }, [])
  useEffect(() => {
    void refreshRollouts()
  }, [refreshRollouts])
  const completeRollout = async (rollout: BackendModelRollout) => {
    const reason = rolloutReason.trim()
    if (!reason || !window.confirm(`Перевести ${rollout.candidate_model_version} в полный production?`)) return
    setBusyRollout(rollout.rollout_id)
    try {
      await completeModelRollout(rollout.rollout_id, reason)
    } catch {
      onToast('Не удалось завершить rollout; проверьте текущие gates и состояние API')
      setBusyRollout(null)
      return
    }
    setRolloutReason('')
    onToast('Candidate переведён в полный production вручную')
    try {
      await refreshRollouts()
      await onRefresh()
    } catch {
      onToast('Rollout завершён, но данные не удалось обновить')
    } finally {
      setBusyRollout(null)
    }
  }
  const rollbackRollout = async (rollout: BackendModelRollout) => {
    const reason = rollbackReason.trim()
    if (!reason || !window.confirm(`Откатить rollout ${rollout.candidate_model_version} к предыдущей production-модели?`)) return
    setBusyRollback(rollout.rollout_id)
    let result: Awaited<ReturnType<typeof rollbackModelRollout>>
    try {
      result = await rollbackModelRollout(rollout.rollout_id, reason)
    } catch {
      onToast('Не удалось откатить rollout; проверьте его состояние и доступность предыдущей модели')
      setBusyRollback(null)
      return
    }
    setRollbackReason('')
    onToast(result.production_pointer_changed
      ? 'Предыдущая verified-модель восстановлена в production'
      : 'Canary остановлен; production-модель не менялась')
    try {
      await refreshRollouts()
      await onRefresh()
    } catch {
      onToast('Откат выполнен, но данные не удалось обновить')
    } finally {
      setBusyRollback(null)
    }
  }
  const review = async (evidenceId: string, decision: 'OPEN_CANDIDATE_CYCLE' | 'DISMISS') => {
    setBusyTrigger(evidenceId)
    try {
      await reviewDriftTrigger(evidenceId, decision)
    } catch {
      onToast('Не удалось сохранить решение reviewer')
      setBusyTrigger(null)
      return
    }
    onToast(decision === 'OPEN_CANDIDATE_CYCLE' ? 'Цикл COLLECT открыт; production-модель не изменена' : 'Drift trigger закрыт reviewer')
    try {
      await onRefresh()
    } catch {
      onToast('Решение сохранено, но список trigger не удалось обновить')
    } finally {
      setBusyTrigger(null)
    }
  }
  return <div className="analytics-page">
    <section className="panel">
      <PanelHeading title={translateUi("Реестр моделей")} />
      {models.length
        ? <><p className="models-scroll-hint">{translateUi('Прокрутите таблицу в сторону, чтобы увидеть остальные столбцы.')}</p><div className="models-table" role="region" aria-label={translateUi('Реестр моделей')} tabIndex={0}><div className="models-head"><span>{translateUi('Модель')}</span><span>{translateUi('Версия')}</span><span>{translateUi('Статус')}</span><span>{translateUi('Метрика')}</span><span>{translateUi('Обновлено')}</span></div>{models.map((model) => <div className="models-row" key={model.version}><strong>{model.name}</strong><code>{model.version}</code><span>{model.status}</span><span><strong>{model.metricValue}</strong><small>{translateUi(model.metric)}</small></span><span>{formatUiDateTime(model.updatedAt)}</span></div>)}</div></>
        : <NoData message={translateUi("Реестр моделей не предоставлен API.")} />}
    </section>
    <section className="panel">
      <PanelHeading title="Canary rollout" />
      <p className="panel-note">{translateUi("В canary поступает 10% обращений. Полный production доступен только после 100 успешных canary-тикетов, 20 решений оператора, нулевых ошибок inference и correction-rate delta не выше +5 п.п.; переключение выполняет reviewer вручную.")}</p>
      {rolloutError && <p className="panel-note">{rolloutError}</p>}
      {!rolloutError && rollouts.length === 0 && <p className="panel-note">{translateUi("Активных и завершённых rollout пока нет.")}</p>}
      {rollouts.map((rollout) => {
        const metrics = rollout.metrics
        const delta = metrics.correction_rate_delta
        const deltaLabel = delta == null ? translateUi('Недостаточно решений') : `${delta > 0 ? '+' : ''}${(delta * 100).toFixed(1)} ${translateUi('п.п.')}`
        const active = rollout.status === 'CANARY' || rollout.status === 'MONITORING'
        const rollbackAvailable = active || rollout.status === 'FULL_PRODUCTION'
        return <article className="drift-trigger-card" key={rollout.rollout_id}>
          <div className="dataset-stat"><span>{translateUi("Rollout / состояние")}</span><strong>{rollout.rollout_id} · {rollout.status}</strong></div>
          <div className="dataset-stat"><span>Candidate / production baseline</span><strong>{rollout.candidate_model_version} · {rollout.previous_production_model_version}</strong></div>
          <div className="dataset-stat"><span>Canary traffic</span><strong>{rollout.canary_traffic_percent}%</strong></div>
          <div className="dataset-stat"><span>Canary tickets / operator decisions</span><strong>{metrics.canary_ticket_count} / {metrics.canary_decision_count}</strong></div>
          <div className="dataset-stat"><span>Correction-rate delta</span><strong>{deltaLabel}</strong></div>
          <div className="dataset-stat"><span>Inference failures / policy</span><strong>{metrics.failed_inference_count} / {rollout.policy_version}</strong></div>
          {rollout.blocking_gates.length > 0 && <p className="panel-note">{translateUi("Ожидают выполнения:")} {rollout.blocking_gates.join(', ')}</p>}
          {active && canReview && <label className="learning-candidate-register">
            <span>{translateUi("Причина ручного полного rollout")}</span>
            <input className="learning-note" value={rolloutReason} maxLength={1000} onChange={(event) => setRolloutReason(event.target.value)} disabled={busyRollout !== null || busyRollback !== null} />
          </label>}
          {active && canReview && <button className="button button-primary" disabled={busyRollout !== null || busyRollback !== null || !rollout.full_rollout_eligible || rolloutReason.trim().length === 0} onClick={() => void completeRollout(rollout)}>
            {translateUi(busyRollout === rollout.rollout_id ? 'Переводим…' : 'Вручную перевести в полный production')}
          </button>}
          {rollbackAvailable && canReview && <label className="learning-candidate-register">
            <span>{translateUi("Причина ручного rollback")}</span>
            <input className="learning-note" value={rollbackReason} maxLength={1000} onChange={(event) => setRollbackReason(event.target.value)} disabled={busyRollback !== null || busyRollout !== null} />
          </label>}
          {rollbackAvailable && canReview && <button className="button button-quiet" disabled={busyRollback !== null || busyRollout !== null || rollbackReason.trim().length === 0} onClick={() => void rollbackRollout(rollout)}>
            {translateUi(busyRollback === rollout.rollout_id ? 'Откатываем…' : active ? 'Остановить canary и сохранить production' : 'Откатить к предыдущей production-модели')}
          </button>}
        </article>
      })}
    </section>
    <section className="panel">
      <PanelHeading title={translateUi("Drift evidence на проверке")} />
      {driftTriggers.length === 0
        ? <p className="panel-note">{translateUi("Новых агрегированных сигналов Data/ML нет.")}</p>
        : <div className="drift-trigger-list">{driftTriggers.map((trigger) => {
          const evidence = trigger.evidence
          return <article className="drift-trigger-card" key={evidence.evidence_id}>
            <div className="dataset-stat"><span>{translateUi("Состояние")}</span><strong>{trigger.state}</strong></div>
            <div className="dataset-stat"><span>Evidence ID</span><code>{evidence.evidence_id}</code></div>
            <div className="dataset-stat"><span>{translateUi("Модель / detector")}</span><strong>{evidence.model_version} · {evidence.detector_version}</strong></div>
            <div className="dataset-stat"><span>{translateUi("Метрика")}</span><strong>{evidence.metric_name}</strong></div>
            <div className="dataset-stat"><span>Baseline / observed</span><strong>{evidence.baseline_value.toFixed(3)} / {evidence.observed_value.toFixed(3)}</strong></div>
            <div className="dataset-stat"><span>Drift score / threshold</span><strong>{evidence.drift_score.toFixed(3)} / {evidence.threshold.toFixed(3)}</strong></div>
            <div className="dataset-stat"><span>{translateUi("Выборка")}</span><strong>{evidence.sample_count} {translateUi("/ минимум")} {evidence.minimum_sample_count}{evidence.synthetic ? ' · synthetic' : ''}</strong></div>
            <div className="dataset-stat"><span>{translateUi("Окно наблюдения")}</span><strong>{evidence.observed_window_start} — {evidence.observed_window_end}</strong></div>
            {trigger.learning_cycle_id && <div className="dataset-stat"><span>{translateUi("Цикл")}</span><strong>{trigger.learning_cycle_id}</strong></div>}
            {trigger.state === 'PENDING_REVIEW' && canReview && <div className="learning-actions" aria-label={translateUi("Действия reviewer")}>
              <button className="button button-primary" disabled={busyTrigger !== null} onClick={() => void review(evidence.evidence_id, 'OPEN_CANDIDATE_CYCLE')}>{translateUi(busyTrigger === evidence.evidence_id ? 'Сохраняем…' : 'Открыть цикл COLLECT')}</button>
              <button className="button button-quiet" disabled={busyTrigger !== null} onClick={() => void review(evidence.evidence_id, 'DISMISS')}>{translateUi("Отклонить сигнал")}</button>
            </div>}
          </article>
        })}</div>}
      <p className="panel-note">{translateUi("Открытие цикла требует текущую production-модель и frozen evaluation dataset. Оно не продвигает candidate и не меняет production pointer.")}</p>
    </section>
  </div>
}
