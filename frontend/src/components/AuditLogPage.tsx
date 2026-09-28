import { localeTag, translateUi } from '../uiSettings'
import { useEffect, useState } from 'react'
import { loadAuditLog, type AuditLogPage as AuditLogPageData } from '../api/client'

const AUDIT_PAGE_SIZE = 50

function auditLoadError(error: unknown): string {
  if (error instanceof Error && error.message === 'API 409') {
    return 'Постоянный журнал доступен при подключённом PostgreSQL.'
  }
  if (error instanceof Error && error.message === 'API 403') {
    return 'Для просмотра журнала нужны права руководителя или администратора.'
  }
  if (error instanceof Error && error.message === 'API 401') {
    return 'Войдите в систему, чтобы просматривать журнал.'
  }
  return 'Не удалось загрузить журнал. Повторите попытку.'
}

function formatAuditTime(value: string): string {
  return new Date(value).toLocaleString(localeTag())
}

export function AuditLogPage() {
  const [pageIndex, setPageIndex] = useState(0)
  const [reloadVersion, setReloadVersion] = useState(0)
  const [page, setPage] = useState<AuditLogPageData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    setLoading(true)
    setError(null)
    setPage(null)

    loadAuditLog(AUDIT_PAGE_SIZE, pageIndex * AUDIT_PAGE_SIZE)
      .then((result) => {
        if (active) setPage(result)
      })
      .catch((reason: unknown) => {
        if (active) setError(auditLoadError(reason))
      })
      .finally(() => {
        if (active) setLoading(false)
      })

    return () => { active = false }
  }, [pageIndex, reloadVersion])

  const total = page?.total ?? 0
  const firstItem = total > 0 ? pageIndex * AUDIT_PAGE_SIZE + 1 : 0
  const lastItem = Math.min(pageIndex * AUDIT_PAGE_SIZE + (page?.items.length ?? 0), total)
  const canGoBack = pageIndex > 0
  const canGoForward = page !== null && page.offset + page.items.length < page.total

  return (
    <div className="analytics-page">
      <section className="panel audit-log-panel">
        <div className="panel-heading">
          <div>
            <h2>{translateUi("События системы")}</h2>
            <p className="panel-note">{translateUi("В выдачу входят только учётные поля. Свободный текст и metadata закрыты.")}</p>
          </div>
          <span className="audit-log-total">{page ? `${total.toLocaleString(localeTag())} ${translateUi('записей')}` : ' '}</span>
        </div>

        {loading && <div className="audit-log-state" role="status">{translateUi("Загрузка журнала…")}</div>}
        {!loading && error && (
          <div className="audit-log-state" role="alert">
            <p>{translateUi(error)}</p>
            <button className="button button-secondary" onClick={() => setReloadVersion((version) => version + 1)}>{translateUi("Повторить")}</button>
          </div>
        )}
        {!loading && !error && page?.items.length === 0 && (
          <div className="audit-log-state">{translateUi("В журнале пока нет записей.")}</div>
        )}
        {!loading && !error && page && page.items.length > 0 && (
          <>
            <p className="audit-scroll-hint">{translateUi('Прокрутите таблицу в сторону, чтобы увидеть остальные столбцы.')}</p>
            <div className="audit-log-table-scroll" role="region" aria-label={translateUi('Таблица событий аудита')} tabIndex={0}>
              <table className="audit-log-table">
                <thead>
                  <tr>
                    <th scope="col">{translateUi("Время")}</th>
                    <th scope="col">{translateUi("Субъект")}</th>
                    <th scope="col">{translateUi("Действие")}</th>
                    <th scope="col">{translateUi("Тип объекта")}</th>
                    <th scope="col">{translateUi("ID объекта")}</th>
                    <th scope="col">Request ID</th>
                  </tr>
                </thead>
                <tbody>
                  {page.items.map((event) => (
                    <tr key={event.id}>
                      <td><time dateTime={event.created_at}>{formatAuditTime(event.created_at)}</time></td>
                      <td>{event.actor_id ?? '—'}</td>
                      <td><code>{event.action}</code></td>
                      <td>{event.entity_type}</td>
                      <td>{event.entity_id ?? '—'}</td>
                      <td><code>{event.request_id ?? '—'}</code></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="audit-log-pagination">
              <span>{translateUi("Показаны")} {firstItem}–{lastItem} {translateUi("из")} {total.toLocaleString(localeTag())}</span>
              <div>
                <button className="button button-secondary" disabled={!canGoBack} onClick={() => setPageIndex((index) => Math.max(0, index - 1))}>{translateUi("Назад")}</button>
                <button className="button button-secondary" disabled={!canGoForward} onClick={() => setPageIndex((index) => index + 1)}>{translateUi("Далее")}</button>
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  )
}
