# Rolling forecast

`GET /api/v1/forecast` remains a read-only forecast calculation. Managers use
`POST /api/v1/forecast/reforecast` to save the current version. Core keeps one
version per normalized filter slice, horizon, model version and UTC day; repeat
requests for the same key return the saved version. PostgreSQL owns the
version chain and manager signals. The ML forecast endpoint remains stateless.

Each saved run contains its filter state, horizon, issue time, model version,
history, forecast points, expected peaks and backtest evidence. A new run links
to the previous run for that same slice and horizon. Core joins completed
previous forecast dates to actual PostgreSQL daily counts, and compares the
two versions on their shared future dates. The response and forecast page show
previous forecast → observed fact and previous forecast → updated forecast.
Only completed UTC dates are treated as actual observations.

Manager signal policy `peak-change-backtest-mae-v1` is deliberately tied to
observed model error. It compares the maximum forecast value on the shared
future dates and creates one persisted signal when the absolute change exceeds
the larger of the previous and updated backtest MAE values. Both versions must
have status `OK` and a finite MAE from at least one backtest sample. Without
those measurements the comparison is retained, but no signal is created.

A signal is also withheld unless every ticket in the forecast's 366-day source
slice is linked to a dataset explicitly marked non-synthetic. Synthetic or
unassigned ticket provenance does not become a manager conclusion. The policy
records its version, both peak values and dates, signed delta, threshold, and
the two run IDs. It does not infer staffing needs, causes, or official service
status. Memory/demo mode does not persist rolling runs or emit manager signals.

Reforecasting happens when the manager dashboard requests the explicit POST;
this task does not add a scheduler. The forecast runs and signals remain
separate from the recurring learning-cycle scheduler.
