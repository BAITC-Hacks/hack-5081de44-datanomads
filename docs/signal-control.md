# Signal control

Signal control is a manager observation of an existing region/topic ticket-volume
alert. Starting it does not acknowledge or close the alert, and those actions do
not contribute to detector precision or signal outcomes.

`POST /api/v1/alerts/{alert_id}/monitor` accepts `monitoring_period_days`. The
period must contain one to twelve complete windows using the detector period
saved on the alert. The saved detector version, configuration, original count,
baseline and dispersion are copied into a separate monitoring record. A second
active period for the same alert is rejected; a later period can be started
after the previous one has finished.

When a manager reads the alert list or detail after the period ends, or runs
alert detection, Core reads the corresponding non-overlapping ticket windows,
persists their counts and source ticket IDs, and records one observed state:

| State | Rule |
| --- | --- |
| `STABILIZED` | The latest complete window is below the saved detector threshold. |
| `PERSISTING` | The latest window still crosses the threshold and does not exceed the original alert count. |
| `WORSENING` | The latest window still crosses the threshold and exceeds the original alert count. |
| `RECURRED` | A below-threshold window is followed by a later window that crosses the threshold again. |
| `INSUFFICIENT_HISTORY` | Saved detector evidence is missing/incompatible or complete observation windows cannot be evaluated. |

`RECURRED` takes precedence when its transition was observed during the period.
Otherwise the final window determines whether the signal is stabilized,
persisting or worsening. These labels describe only ticket counts currently
stored in PostgreSQL, compared with the original detector baseline and
thresholds. They do not establish physical incidents, root cause, official
service status, or whether a manager action solved anything. Raw alert evidence
is retained independently from this follow-up evidence.

There is no external ingestion watermark in the current ticket contract. The
evidence records the exact PostgreSQL ticket IDs and window bounds used, so the
calculation is reproducible against that stored snapshot; it cannot prove that
an upstream CRM feed was complete during the period. Demo memory mode therefore
returns `INSUFFICIENT_HISTORY` after the selected period rather than presenting
fixture data as an authoritative signal series.

## Advanced lifecycle gate

The separate `BACKGROUND → WATCH → VERIFY → ESCALATE` lifecycle is not
enabled. The repository has no detector-specific calibration or evaluation
evidence showing that those transitions reduce alert noise for the saved
detector versions. Candidate model promotion metrics are a separate contract
and do not establish detector quality. ACK/CLOSE actions are not used as
precision labels. Data/ML must provide versioned detector calibration and a
comparable evaluation showing the noise change before runtime lifecycle states
can be added. Until then, no advanced lifecycle state is created. Task-044
manager monitoring remains separate and does not imply `BACKGROUND`, `WATCH`,
`VERIFY` or `ESCALATE`.
