# Capacity gap assessment

The current forecast reports ticket demand and expected peak days. It does not
estimate staffing needs or operational capacity. The repository has no verified
sources for staffing, handling time or throughput, work schedules, or a service
level target/SLA, so Core returns `capacity_assessment.status: DATA_UNAVAILABLE`
and lists all four missing input groups.

Synthetic tickets, average ticket volume, and forecast backtest error are not
substitutes for operational capacity inputs. The API and forecast screen must
not turn them into a staffing count, a capacity risk, or a service-level claim.

An assessment can be implemented only after those inputs have authoritative
sources and compatible time/service/region dimensions. The calculation must use
forecast demand minus planned capacity and must identify the input versions and
slice used. Missing or stale inputs must keep the result unavailable. The ML
forecast service remains a demand-only inference contract.
