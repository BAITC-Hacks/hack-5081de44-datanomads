# Verified resolution memory

Operator actions and outcomes remain separate records. The operator history
shows `Action recorded; outcome UNKNOWN` when an action exists without outcome
evidence, and shows `Action recorded; outcome VERIFIED` only when a separate
verification event exists. A closed CRM status is not verification evidence.

When an operator opens a similar ticket, its latest outcome state and
source/channel/actor/time are shown with the prior action. `UNKNOWN` stays
visible. Outcome state does not change similarity ranking, automatically link
tickets, or replace the original ticket text. If the outcome endpoint is
unavailable, the related ticket remains visible and the UI says the outcome
could not be loaded rather than treating the ticket as verified.
