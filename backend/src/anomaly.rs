use chrono::{DateTime, Duration, Utc};
use serde::Serialize;
use std::env;

use crate::Alert;

const DEFAULT_DETECTOR_VERSION: &str = "region-topic-median-mad-v1";
const DEFAULT_PERIOD_DAYS: u32 = 7;
const DEFAULT_BASELINE_PERIODS: u32 = 4;
const DEFAULT_MINIMUM_COUNT: u32 = 3;
const DEFAULT_ROBUST_Z_THRESHOLD: f64 = 3.5;
const DEFAULT_RATIO_THRESHOLD: f64 = 1.5;
const DEFAULT_HIGH_Z_THRESHOLD: f64 = 5.0;
const DEFAULT_CRITICAL_Z_THRESHOLD: f64 = 7.0;
const DEFAULT_HIGH_RATIO_THRESHOLD: f64 = 2.0;
const DEFAULT_CRITICAL_RATIO_THRESHOLD: f64 = 3.0;
const DEFAULT_BASELINE_FLOOR: f64 = 1.0;
const DEFAULT_COOLDOWN_HOURS: u32 = 168;
const MAD_NORMALIZATION: f64 = 1.4826;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SeasonalityMode {
    WeekdayAlignedWeekly,
    None,
}

#[derive(Clone, Debug, Serialize)]
pub struct AlertDetectorConfig {
    pub detector_version: String,
    pub period_days: u32,
    pub baseline_periods: u32,
    pub minimum_count: u32,
    pub robust_z_threshold: f64,
    pub ratio_threshold: f64,
    pub high_z_threshold: f64,
    pub critical_z_threshold: f64,
    pub high_ratio_threshold: f64,
    pub critical_ratio_threshold: f64,
    pub baseline_floor: f64,
    pub cooldown_hours: u32,
    pub seasonality: SeasonalityMode,
    pub baseline_method: &'static str,
    pub dispersion_method: &'static str,
}

impl Default for AlertDetectorConfig {
    fn default() -> Self {
        Self {
            detector_version: DEFAULT_DETECTOR_VERSION.to_owned(),
            period_days: DEFAULT_PERIOD_DAYS,
            baseline_periods: DEFAULT_BASELINE_PERIODS,
            minimum_count: DEFAULT_MINIMUM_COUNT,
            robust_z_threshold: DEFAULT_ROBUST_Z_THRESHOLD,
            ratio_threshold: DEFAULT_RATIO_THRESHOLD,
            high_z_threshold: DEFAULT_HIGH_Z_THRESHOLD,
            critical_z_threshold: DEFAULT_CRITICAL_Z_THRESHOLD,
            high_ratio_threshold: DEFAULT_HIGH_RATIO_THRESHOLD,
            critical_ratio_threshold: DEFAULT_CRITICAL_RATIO_THRESHOLD,
            baseline_floor: DEFAULT_BASELINE_FLOOR,
            cooldown_hours: DEFAULT_COOLDOWN_HOURS,
            seasonality: SeasonalityMode::WeekdayAlignedWeekly,
            baseline_method: "median",
            dispersion_method: "median_absolute_deviation",
        }
    }
}

impl AlertDetectorConfig {
    pub fn from_env() -> Self {
        let defaults = Self::default();
        let config = Self {
            detector_version: env::var("PULSE_ALERT_DETECTOR_VERSION")
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
                .unwrap_or(defaults.detector_version),
            period_days: positive_env("PULSE_ALERT_PERIOD_DAYS", defaults.period_days),
            baseline_periods: positive_env(
                "PULSE_ALERT_BASELINE_PERIODS",
                defaults.baseline_periods,
            ),
            minimum_count: positive_env("PULSE_ALERT_MINIMUM_COUNT", defaults.minimum_count),
            robust_z_threshold: positive_float_env(
                "PULSE_ALERT_ROBUST_Z_THRESHOLD",
                defaults.robust_z_threshold,
            ),
            ratio_threshold: positive_float_env(
                "PULSE_ALERT_RATIO_THRESHOLD",
                defaults.ratio_threshold,
            ),
            high_z_threshold: positive_float_env(
                "PULSE_ALERT_HIGH_Z_THRESHOLD",
                defaults.high_z_threshold,
            ),
            critical_z_threshold: positive_float_env(
                "PULSE_ALERT_CRITICAL_Z_THRESHOLD",
                defaults.critical_z_threshold,
            ),
            high_ratio_threshold: positive_float_env(
                "PULSE_ALERT_HIGH_RATIO_THRESHOLD",
                defaults.high_ratio_threshold,
            ),
            critical_ratio_threshold: positive_float_env(
                "PULSE_ALERT_CRITICAL_RATIO_THRESHOLD",
                defaults.critical_ratio_threshold,
            ),
            baseline_floor: positive_float_env(
                "PULSE_ALERT_BASELINE_FLOOR",
                defaults.baseline_floor,
            ),
            cooldown_hours: positive_env("PULSE_ALERT_COOLDOWN_HOURS", defaults.cooldown_hours),
            seasonality: seasonality_env(defaults.seasonality),
            baseline_method: defaults.baseline_method,
            dispersion_method: defaults.dispersion_method,
        };
        config.validate();
        config
    }

    pub fn validate(&self) {
        assert!(
            (1..=31).contains(&self.period_days),
            "PULSE_ALERT_PERIOD_DAYS must be between 1 and 31"
        );
        assert!(
            (1..=52).contains(&self.baseline_periods),
            "PULSE_ALERT_BASELINE_PERIODS must be between 1 and 52"
        );
        assert!(
            self.minimum_count <= i32::MAX as u32,
            "PULSE_ALERT_MINIMUM_COUNT must fit in a 32-bit integer"
        );
        assert!(
            self.minimum_count > 0,
            "PULSE_ALERT_MINIMUM_COUNT must be positive"
        );
        assert!(
            (1..=24 * 365).contains(&self.cooldown_hours),
            "PULSE_ALERT_COOLDOWN_HOURS must be between 1 and 8760"
        );
        assert!(
            !self.detector_version.trim().is_empty(),
            "PULSE_ALERT_DETECTOR_VERSION must not be empty"
        );
        assert!(
            self.robust_z_threshold.is_finite()
                && self.ratio_threshold.is_finite()
                && self.high_z_threshold.is_finite()
                && self.critical_z_threshold.is_finite()
                && self.high_ratio_threshold.is_finite()
                && self.critical_ratio_threshold.is_finite()
                && self.robust_z_threshold > 0.0
                && self.ratio_threshold >= 1.0
                && self.high_z_threshold >= self.robust_z_threshold
                && self.critical_z_threshold >= self.high_z_threshold
                && self.high_ratio_threshold >= self.ratio_threshold
                && self.critical_ratio_threshold >= self.high_ratio_threshold,
            "alert detector thresholds must be positive and ordered"
        );
        assert!(
            self.baseline_floor.is_finite() && self.baseline_floor > 0.0,
            "PULSE_ALERT_BASELINE_FLOOR must be positive"
        );
        if matches!(self.seasonality, SeasonalityMode::WeekdayAlignedWeekly) {
            assert_eq!(
                self.period_days, 7,
                "weekday-aligned seasonality requires seven-day periods"
            );
        }
    }
}

fn positive_env<T>(name: &str, default: T) -> T
where
    T: std::str::FromStr + PartialOrd + Default,
    T::Err: std::fmt::Debug,
{
    let Ok(value) = env::var(name) else {
        return default;
    };
    let parsed = value
        .parse::<T>()
        .unwrap_or_else(|_| panic!("{name} must be a positive integer"));
    assert!(parsed > T::default(), "{name} must be a positive integer");
    parsed
}

fn positive_float_env(name: &str, default: f64) -> f64 {
    let Ok(value) = env::var(name) else {
        return default;
    };
    let parsed = value
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("{name} must be a positive number"));
    assert!(
        parsed.is_finite() && parsed > 0.0,
        "{name} must be a positive finite number"
    );
    parsed
}

fn seasonality_env(default: SeasonalityMode) -> SeasonalityMode {
    let Ok(value) = env::var("PULSE_ALERT_SEASONALITY") else {
        return default;
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "weekday_aligned_weekly" => SeasonalityMode::WeekdayAlignedWeekly,
        "none" => SeasonalityMode::None,
        _ => panic!("PULSE_ALERT_SEASONALITY must be weekday_aligned_weekly or none"),
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AlertSeriesInput {
    pub region_id: String,
    pub topic_id: String,
    pub period_end: DateTime<Utc>,
    pub coverage_start: DateTime<Utc>,
    /// The current period followed by complete historical periods, newest first.
    pub counts: Vec<i64>,
    pub current_ticket_ids: Vec<i64>,
}

#[derive(Debug)]
pub(crate) struct AlertDetectionRun {
    pub status: &'static str,
    pub source: &'static str,
    pub evaluated_series: usize,
    pub insufficient_history_series: usize,
    pub new_alerts: usize,
    pub items: Vec<Alert>,
}

#[derive(Clone, Debug)]
pub(crate) struct AlertEvidence {
    pub region_id: String,
    pub topic_id: String,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub current_count: i64,
    pub historical_counts: Vec<i64>,
    pub baseline: f64,
    pub median_absolute_deviation: f64,
    pub robust_dispersion: f64,
    pub deviation: f64,
    pub robust_z: Option<f64>,
    pub ratio: f64,
    pub severity: &'static str,
    pub reasons: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub(crate) enum AlertEvaluation {
    InsufficientHistory,
    BelowThreshold,
    Anomaly(AlertEvidence),
}

pub(crate) fn evaluate_series(
    input: &AlertSeriesInput,
    config: &AlertDetectorConfig,
) -> AlertEvaluation {
    let period_start = input.period_end - Duration::days(i64::from(config.period_days));
    let required_start = period_start
        - Duration::days(i64::from(config.period_days) * i64::from(config.baseline_periods));
    if input.counts.len() != config.baseline_periods as usize + 1
        || input.coverage_start > required_start
    {
        return AlertEvaluation::InsufficientHistory;
    }

    let current_count = input.counts[0].max(0);
    let historical_counts = input.counts[1..].to_vec();
    let mut baseline_values = historical_counts
        .iter()
        .map(|value| *value as f64)
        .collect::<Vec<_>>();
    let baseline = median(&mut baseline_values);
    let mut absolute_deviations = historical_counts
        .iter()
        .map(|value| (*value as f64 - baseline).abs())
        .collect::<Vec<_>>();
    let median_absolute_deviation = median(&mut absolute_deviations);
    let robust_dispersion = median_absolute_deviation * MAD_NORMALIZATION;
    let robust_z =
        (robust_dispersion > 0.0).then(|| (current_count as f64 - baseline) / robust_dispersion);
    let deviation = current_count as f64 - baseline;
    let ratio = current_count as f64 / baseline.max(config.baseline_floor);
    if current_count < i64::from(config.minimum_count) {
        return AlertEvaluation::BelowThreshold;
    }
    let z_triggered = robust_z.is_some_and(|value| value >= config.robust_z_threshold);
    let ratio_triggered = ratio >= config.ratio_threshold;
    if !z_triggered && !ratio_triggered {
        return AlertEvaluation::BelowThreshold;
    }

    let severity = if ratio >= config.critical_ratio_threshold
        || robust_z.is_some_and(|value| value >= config.critical_z_threshold)
    {
        "CRITICAL"
    } else if ratio >= config.high_ratio_threshold
        || robust_z.is_some_and(|value| value >= config.high_z_threshold)
    {
        "HIGH"
    } else {
        "MEDIUM"
    };
    let mut reasons = Vec::new();
    if z_triggered {
        reasons.push("ROBUST_Z_THRESHOLD");
    }
    if ratio_triggered {
        reasons.push("RATIO_THRESHOLD");
    }

    AlertEvaluation::Anomaly(AlertEvidence {
        region_id: input.region_id.clone(),
        topic_id: input.topic_id.clone(),
        period_start,
        period_end: input.period_end,
        current_count,
        historical_counts,
        baseline,
        median_absolute_deviation,
        robust_dispersion,
        deviation,
        robust_z,
        ratio,
        severity,
        reasons,
    })
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::{evaluate_series, AlertDetectorConfig, AlertEvaluation, AlertSeriesInput};
    use chrono::{DateTime, Duration, Utc};

    fn period_end() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-09-28T00:00:00Z")
            .expect("test timestamp is valid")
            .with_timezone(&Utc)
    }

    fn input(current_count: i64, history: &[i64], coverage_weeks: i64) -> AlertSeriesInput {
        let period_end = period_end();
        AlertSeriesInput {
            region_id: "R01".to_owned(),
            topic_id: "TOPIC-WATER".to_owned(),
            period_end,
            coverage_start: period_end - Duration::days(coverage_weeks * 7),
            counts: std::iter::once(current_count)
                .chain(history.iter().copied())
                .collect(),
            current_ticket_ids: (0..current_count.max(0)).collect(),
        }
    }

    #[test]
    fn insufficient_region_coverage_is_not_scored_as_a_zero_baseline() {
        let config = AlertDetectorConfig::default();
        let result = evaluate_series(&input(12, &[0, 0, 0, 0], 4), &config);
        assert!(matches!(result, AlertEvaluation::InsufficientHistory));
    }

    #[test]
    fn missing_baseline_period_is_reported_as_insufficient_history() {
        let config = AlertDetectorConfig::default();
        let result = evaluate_series(&input(12, &[0, 0, 0], 6), &config);
        assert!(matches!(result, AlertEvaluation::InsufficientHistory));
    }

    #[test]
    fn a_count_increase_over_a_stable_zero_baseline_uses_the_ratio_gate() {
        let config = AlertDetectorConfig::default();
        let result = evaluate_series(&input(3, &[0, 0, 0, 0], 6), &config);
        let AlertEvaluation::Anomaly(evidence) = result else {
            panic!("expected a reproducible anomaly");
        };
        assert_eq!(evidence.baseline, 0.0);
        assert_eq!(evidence.ratio, 3.0);
        assert_eq!(evidence.severity, "CRITICAL");
        assert_eq!(evidence.reasons, vec!["RATIO_THRESHOLD"]);
    }

    #[test]
    fn median_mad_detects_a_spike_and_keeps_historical_counts() {
        let config = AlertDetectorConfig::default();
        let result = evaluate_series(&input(12, &[1, 2, 1, 2], 6), &config);
        let AlertEvaluation::Anomaly(evidence) = result else {
            panic!("expected a robust-z anomaly");
        };
        assert_eq!(evidence.baseline, 1.5);
        assert_eq!(evidence.historical_counts, vec![1, 2, 1, 2]);
        assert!(evidence.robust_z.unwrap() >= config.robust_z_threshold);
        assert!(evidence.reasons.contains(&"ROBUST_Z_THRESHOLD"));
    }

    #[test]
    fn minimum_count_and_stable_variation_do_not_emit_alerts() {
        let config = AlertDetectorConfig::default();
        assert!(matches!(
            evaluate_series(&input(2, &[1, 1, 2, 1], 6), &config),
            AlertEvaluation::BelowThreshold
        ));
        assert!(matches!(
            evaluate_series(&input(2, &[1, 2, 1, 2], 6), &config),
            AlertEvaluation::BelowThreshold
        ));
    }
}
