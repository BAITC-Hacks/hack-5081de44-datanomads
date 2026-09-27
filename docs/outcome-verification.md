# Outcome verification

Pulse keeps outcome verification separate from the official status imported from
109 CRM. Closing a CRM ticket does not make its Pulse outcome state
`VERIFIED`. If no outcome evidence exists, Pulse reports `UNKNOWN`; `UNKNOWN`
is not stored as an event and cannot be submitted to the write endpoint.

The current adapter is `DemoOutcomeVerificationAdapter`. It stores operator
simulations with source `DEMO_SIMULATION` and channel
`OPERATOR_PANEL_SIMULATION`, together with the authenticated actor and server
timestamp. The operator panel labels each entry as a simulation. It does not
send messages, update CRM status, or establish that a citizen confirmed the
result.

A real citizen feedback channel is not connected. Production citizen
verification remains an unmet external dependency; only the adapter contract
and demo simulation are available. Audit metadata contains the ticket ID,
state, source, and channel, never the ticket text.
