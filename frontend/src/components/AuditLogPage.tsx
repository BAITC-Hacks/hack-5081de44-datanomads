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
  return new Date(value).toLocaleString('ru-RU')
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
            <h2>События системы</h2>
            <p className="panel-note">В выдачу входят только учётные поля. Свободный текст и metadata закрыты.</p>
          </div>
          <span className="audit-log-total">{page ? `${total.toLocaleString('ru-RU')} записей` : ' '}</span>
        </div>

        {loading && <div className="audit-log-state" role="status">Загрузка журнала…</div>}
        {!loading && error && (
          <div className="audit-log-state" role="alert">
            <p>{error}</p>
            <button className="button button-secondary" onClick={() => setReloadVersion((version) => version + 1)}>Повторить</button>
          </div>
        )}
        {!loading && !error && page?.items.length === 0 && (
          <div className="audit-log-state">В журнале пока нет записей.</div>
        )}
        {!loading && !error && page && page.items.length > 0 && (
          <>
            <div className="audit-log-table-scroll">
              <table className="audit-log-table">
                <thead>
                  <tr>
                    <th scope="col">Время</th>
                    <th scope="col">Субъект</th>
                    <th scope="col">Действие</th>
                    <th scope="col">Тип объекта</th>
                    <th scope="col">ID объекта</th>
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
              <span>Показаны {firstItem}–{lastItem} из {total.toLocaleString('ru-RU')}</span>
              <div>
                <button className="button button-secondary" disabled={!canGoBack} onClick={() => setPageIndex((index) => Math.max(0, index - 1))}>Назад</button>
                <button className="button button-secondary" disabled={!canGoForward} onClick={() => setPageIndex((index) => index + 1)}>Далее</button>
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  )
}
