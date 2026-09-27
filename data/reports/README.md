# Evidence from two local regional 109 CSV exports

These reports were regenerated on 2026-09-27 from two customer-provided regional
CSV files for Eastern Kazakhstan and Almaty Region. The raw files remain outside
Git. Reports contain only field presence and joint-presence counts, semantic
status, date ranges, checksums and aggregate forecast metrics; they contain no
ticket rows or field values.

| Export | SHA-256 | Rows | Calendar days | `original_text` column | Forecast status |
| --- | --- | ---: | ---: | --- | --- |
| Eastern Kazakhstan | `bfa29295829a3c7830ffa5d0389a2924676f804464e577931d98bf3d8ffa7716` | 83,385 | 1,035 | absent | `EVALUATED` for 30/60/90 days |
| Almaty Region | `9c423e6699010981ab636de15f16bcc41cd104e918c6641409f025079f73fd8f` | 19,912 | 315 | absent | `INSUFFICIENT_HISTORY` for 30/60/90 days |

The Eastern Kazakhstan weekly seasonal baseline has 21/20/19 rolling windows
and WAPE 0.4742/0.4661/0.4707 for the three horizons. The export has no rows
for 2023-01-13 and 2023-01-14. These dates are unobserved rather than assumed
zero. The backtest keeps its fixed calendar origins and excludes windows unless
the previous 365 days and the whole target horizon are observed. Almaty Region
has fewer than the required 365 training days plus each forecast horizon, so
its report contains no error metrics. These are daily aggregate results for two
regions, not evidence of hourly forecasts or quality across all regions.

The [candidate comparison](vko_109_forecast_candidates.json) uses the same
21/20/19 windows. Prophet WAPE is 0.4574/0.4857/0.5209, versus the seasonal
naive values above. It wins only the 30-day horizon under the fixed WAPE rule;
the 60- and 90-day baseline remains better. The
[Almaty comparison](almaty_109_forecast_candidates.json) correctly records
`INSUFFICIENT_HISTORY`. Neither report authorizes runtime promotion, and
uncertainty intervals or peak-detection quality were not evaluated. The
[spike exploration](vko_109_spike_exploration.json) excludes four days whose
observations or same-weekday history are incomplete because of those two gaps;
its alert counts remain exploratory without incident labels.

The customer confirmed that `com_exp` is not useful citizen appeal text. Neither
file contains original appeal text. Synthetic appeal texts were created
separately; they are not reconstructions of these customer rows. These are two
regional datasets, with no asserted mapping to the seven operational
`source_system` profiles. Neither export is a reviewed classifier or retrieval
corpus. `category`, `service` and `contractor` need separate semantic review
before training or routing use. Neither layout has an explicit priority column.
The CSV audit counts `com_exp` when present but does not require it.
The audit also counts rows with an exact date-time value in `application_number`,
`category` or `service`, where that value suggests a possible column shift. Both
exports have zero such rows. This narrow structural signal cannot rule out
other shifts or establish field semantics; the VKO report still records one
`region` value longer than 120 characters for review.
The [field availability report](customer_109_field_availability.json) and
[interpretation](../../docs/customer-field-availability.md) list safe structural
facts, missing fields and limits on within-file and cross-file comparisons.

To reproduce with the same local files:

```bash
python3 scripts/data_audit.py /path/to/export.csv --output /tmp/source-audit.json
python3 scripts/evaluate_field_availability.py \
  --vko /path/to/vko.csv --almaty /path/to/almaty.csv \
  --output /tmp/customer-109-field-availability.json
python3 scripts/evaluate_forecast_csv.py /path/to/export.csv --output /tmp/forecast-baseline.json
MPLCONFIGDIR=/tmp/pulse109-mpl .venv/bin/python scripts/evaluate_forecast_candidates.py \
  /path/to/export.csv --output /tmp/forecast-candidates.json
```

The output path must be new. Compare the output `sha256` or `source_sha256` with
the values above before comparing metrics.
