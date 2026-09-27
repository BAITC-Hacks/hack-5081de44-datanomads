# Recurrence monitoring

Core labels a retrieval candidate as a possible repeat only when its similarity
passes the existing retrieval threshold, its topic matches the current
operator-confirmed topic or prediction, its structured object matches, and
PostgreSQL reports the earlier ticket as `CLOSED` or `RESOLVED` with a close
time from zero to 30 days before the new ticket was created. The rule snapshot
is versioned as `related-ticket-rules.v2`. Core compares structured object
values internally; candidate responses expose only the `object_match` factor,
not the object string.

If the structured object, official status, close time, or candidate metadata is
missing, Core leaves the candidate as ordinary similarity. When candidate
metadata cannot be read, duplicate/repeat classification is unavailable and
the assist preview requires review. The operator panel says “Возможный повтор
после недавнего закрытия. Проверьте предыдущую историю.”; operators can open
the prior ticket from its candidate row to inspect official status and close
time.

The rule creates no automatic link or CRM update and makes no assessment of
the prior assignee's work. The previous record remains available for manual
review, including its separate operator decision and outcome-verification
state.
