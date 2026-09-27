# Evidence from two local regional 109 CSV exports

These reports were regenerated on 2026-09-27 from two customer-provided regional
CSV files for Eastern Kazakhstan and Almaty Region. The raw files remain outside
Git. Reports contain only column presence counts, date ranges, checksums and
aggregate forecast metrics; they contain no ticket rows or field values.

| Export | SHA-256 | Rows | Calendar days | `original_text` column | Forecast status |
| --- | --- | ---: | ---: | --- | --- |
| Eastern Kazakhstan | `bfa29295829a3c7830ffa5d0389a2924676f804464e577931d98bf3d8ffa7716` | 83,385 | 1,035 | absent | `EVALUATED` for 30/60/90 days |
| Almaty Region | `9c423e6699010981ab636de15f16bcc41cd104e918c6641409f025079f73fd8f` | 19,912 | 315 | absent | `INSUFFICIENT_HISTORY` for 30/60/90 days |

The Eastern Kazakhstan weekly seasonal baseline has 22/21/20 rolling windows
and WAPE 0.4855/0.4860/0.4867 for the three horizons. Two dates without rows
were filled with zero by the current baseline; whether they are true zero days
or export gaps is unverified. Almaty Region has fewer than the required 365
training days plus each forecast horizon, so its report contains no error
metrics. These are daily aggregate results for two regions, not evidence of
hourly forecasts or quality across all regions.

The customer confirmed that `com_exp` is not useful citizen appeal text. Neither
file contains original appeal text. Synthetic appeal texts were created
separately; they are not reconstructions of these customer rows. These are two
regional datasets, with no asserted mapping to the seven operational
`source_system` profiles. Neither export is a reviewed classifier or retrieval
corpus. `category`, `service` and `contractor` need separate semantic review
before training or routing use. Neither layout has an explicit priority column.

To reproduce with the same local files:

```bash
python3 scripts/data_audit.py /path/to/export.csv --output /tmp/source-audit.json
python3 scripts/evaluate_forecast_csv.py /path/to/export.csv --output /tmp/forecast-baseline.json
```

The output path must be new. Compare the output `sha256` or `source_sha256` with
the values above before comparing metrics.
