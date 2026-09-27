//! PostgreSQL/Qdrant/ML integration for the real Core runtime.
//!
//! The demo store in `lib.rs` is intentionally kept for unit tests and for
//! an explicitly selected `PULSE_STORAGE=memory` run.  Compose uses this
//! repository instead.  PostgreSQL is the source of truth; Qdrant only keeps
//! the vector index and the ML service owns classification/embedding.

use crate::anomaly::{
    evaluate_series, AlertDetectionRun, AlertDetectorConfig, AlertEvaluation, AlertSeriesInput,
};
use crate::{
    actionable_context_for_candidates, actionable_context_manual_review,
    actionable_context_needs_candidates, build_context_handoff_package, build_query_intent_result,
    known_routing_value, manual_response_template, metric_rate, query_analytics_filters,
    related_ticket_candidate, render_template_body, validate_query_intent, ActionableContextOption,
    Alert, AlertQuery, AlternativePrediction, AnalyticsDrilldownQuery, AnalyticsDrilldownResponse,
    AnalyticsDrilldownTicket, AnalyticsQuery, AnalyticsResponse, AssistOrchestration,
    AssistPreviewResponse, AssistStage, CloseLearningCycleRequest, Config,
    ContextHandoffLocationSource, ContextHandoffPackage, CreateLearningCycleRequest,
    DatasetProvenance, DecisionRequest, DecisionResponse, ForecastQuery, ForecastResponse,
    ImportRequest, ImportResponse, LearningCycle, LearningFeedback, LearningFeedbackRequest,
    LearningMetrics, LearningOverview, MetricBucket, ModelQuery, ModelVersion, OperatorDecision,
    Prediction, QueryIntentRequest, RelationSuggestionSnapshot, ResponseTemplate,
    ResponseTemplateInput, ResponseTemplateRecord, ResponseTemplatesResponse,
    RoutingFeedbackRecord, RuleProvenance, RuleSource, RuntimeMetrics, Ticket,
    TicketDetailResponse, TicketListResponse, TicketQuery, TimeSeriesPoint, Topic,
    ROUTING_FEEDBACK_DEMO_SOURCE_SYSTEM, ROUTING_FEEDBACK_PENDING_STATUS,
};
use chrono::{DateTime, NaiveDate, Utc};
use reqwest::{Client, StatusCode as HttpStatus};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sqlx::{
    migrate::Migrator, postgres::PgPoolOptions, FromRow, PgPool, Postgres, QueryBuilder, Row,
};
use std::{
    env,
    time::{Duration, Instant},
};

const DEFAULT_EMBEDDING_DIMENSION: usize = 32;
const DEFAULT_QDRANT_COLLECTION: &str = "pulse109_tickets_v1";
const DEFAULT_EMBEDDER_VERSION: &str = "embedder-demo-2026-09-21-001";
const DEFAULT_FORECAST_MODEL_VERSION: &str = "forecast-statsforecast-seasonal-naive-2026-09-24-001";
const READINESS_DEPENDENCY_TIMEOUT: Duration = Duration::from_secs(2);
static MIGRATOR: Migrator = sqlx::migrate!("../migrations");

fn production_runtime_mode(runtime_mode: &str) -> bool {
    matches!(
        runtime_mode.trim().to_ascii_lowercase().as_str(),
        "prod" | "production"
    )
}

fn synthetic_candidate_can_be_promoted(is_synthetic: bool, runtime_mode: &str) -> bool {
    !is_synthetic || !production_runtime_mode(runtime_mode)
}

fn candidate_evaluation_is_promotable(
    evaluation: &Value,
    cycle_id: &str,
    candidate_model_version: &str,
    production_model_version: &str,
    candidate_dataset_version: &str,
    frozen_evaluation_dataset_version: &str,
    policy_version: &str,
) -> bool {
    let expected_thresholds = match policy_version {
        "policy-v1" => json!({
            "minimum_offline_samples": 30,
            "minimum_shadow_samples": 20,
            "maximum_macro_f1_regression": 0.02,
            "maximum_class_f1_regression": 0.05,
            "maximum_shadow_correction_rate_delta": 0.05,
            "maximum_shadow_inference_failures": 0
        }),
        _ => return false,
    };
    if evaluation.get("schema_version").and_then(Value::as_str) != Some("candidate-evaluation.v1")
        || evaluation.get("evaluation_version").and_then(Value::as_str)
            != Some("candidate-evaluation.v1")
        || evaluation.get("status").and_then(Value::as_str) != Some("COMPLETED")
        || evaluation.get("decision").and_then(Value::as_str) != Some("PENDING_HUMAN_DECISION")
        || evaluation.get("synthetic").and_then(Value::as_bool) != Some(false)
        || evaluation.get("cycle_id").and_then(Value::as_str) != Some(cycle_id)
        || evaluation
            .get("candidate_model_version")
            .and_then(Value::as_str)
            != Some(candidate_model_version)
        || evaluation
            .get("production_model_version")
            .and_then(Value::as_str)
            != Some(production_model_version)
        || evaluation
            .get("candidate_dataset_version")
            .and_then(Value::as_str)
            != Some(candidate_dataset_version)
        || evaluation.get("policy_version").and_then(Value::as_str) != Some(policy_version)
        || evaluation
            .get("promotion_policy")
            .and_then(|policy| policy.get("version"))
            .and_then(Value::as_str)
            != Some(policy_version)
        || evaluation
            .get("promotion_policy")
            .and_then(|policy| policy.get("thresholds"))
            != Some(&expected_thresholds)
    {
        return false;
    }

    let Some(gates) = evaluation.get("gates").and_then(Value::as_array) else {
        return false;
    };
    let mut gate_by_key = std::collections::BTreeMap::new();
    for gate in gates {
        let Some(key) = gate.get("key").and_then(Value::as_str) else {
            return false;
        };
        if gate.get("status").and_then(Value::as_str) != Some("PASSED")
            || gate_by_key.insert(key, gate).is_some()
        {
            return false;
        }
    }
    let observed = |key: &str| gate_by_key.get(key).and_then(|gate| gate.get("observed"));
    let threshold = |key: &str| gate_by_key.get(key).and_then(|gate| gate.get("threshold"));
    let candidate_offline = evaluation.get("offline_evaluation");
    let baseline_offline = evaluation.get("baseline_evaluation");
    let shadow_evaluation = evaluation.get("shadow_evaluation");
    let macro_f1_delta = candidate_offline
        .and_then(|offline| offline.get("metrics"))
        .and_then(|metrics| metrics.get("macro_f1"))
        .and_then(Value::as_f64)
        .zip(
            baseline_offline
                .and_then(|offline| offline.get("metrics"))
                .and_then(|metrics| metrics.get("macro_f1"))
                .and_then(Value::as_f64),
        )
        .map(|(candidate, baseline)| candidate - baseline);
    let class_changes_pass = candidate_offline
        .and_then(|offline| offline.get("metrics"))
        .and_then(|metrics| metrics.get("per_class_changes"))
        .and_then(Value::as_object)
        .is_some_and(|changes| {
            !changes.is_empty()
                && changes
                    .values()
                    .all(|change| change.as_f64().is_some_and(|delta| delta >= -0.05))
        });
    let no_critical_regressions = [candidate_offline, evaluation.get("shadow_evaluation")]
        .iter()
        .all(|section| {
            section
                .and_then(|section| section.get("critical_regressions"))
                .and_then(Value::as_array)
                .is_some_and(Vec::is_empty)
        });
    [
        "offline_sample_count",
        "shadow_sample_count",
        "macro_f1_non_inferiority",
        "critical_class_regressions",
        "shadow_correction_rate_delta",
        "candidate_shadow_inference_failures",
        "synthetic_evidence",
    ]
    .iter()
    .all(|key| gate_by_key.contains_key(key))
        && candidate_offline
            .and_then(|offline| offline.get("status"))
            .and_then(Value::as_str)
            == Some("COMPLETED")
        && candidate_offline
            .and_then(|offline| offline.get("model_version"))
            .and_then(Value::as_str)
            == Some(candidate_model_version)
        && candidate_offline
            .and_then(|offline| offline.get("dataset_version"))
            .and_then(Value::as_str)
            == Some(frozen_evaluation_dataset_version)
        && candidate_offline
            .and_then(|offline| offline.get("synthetic"))
            .and_then(Value::as_bool)
            == Some(false)
        && baseline_offline
            .and_then(|offline| offline.get("status"))
            .and_then(Value::as_str)
            == Some("COMPLETED")
        && baseline_offline
            .and_then(|offline| offline.get("model_version"))
            .and_then(Value::as_str)
            == Some(production_model_version)
        && baseline_offline
            .and_then(|offline| offline.get("dataset_version"))
            .and_then(Value::as_str)
            == Some(frozen_evaluation_dataset_version)
        && baseline_offline
            .and_then(|offline| offline.get("synthetic"))
            .and_then(Value::as_bool)
            == Some(false)
        && candidate_offline
            .and_then(|offline| offline.get("sample_count"))
            .and_then(Value::as_u64)
            == observed("offline_sample_count").and_then(Value::as_u64)
        && baseline_offline
            .and_then(|offline| offline.get("sample_count"))
            .and_then(Value::as_u64)
            == observed("offline_sample_count").and_then(Value::as_u64)
        && macro_f1_delta.is_some_and(|delta| delta >= -0.02)
        && class_changes_pass
        && no_critical_regressions
        && shadow_evaluation
            .and_then(|shadow| shadow.get("correction_rate_delta"))
            .and_then(Value::as_f64)
            .is_some_and(|delta| delta <= 0.05)
        && shadow_evaluation
            .and_then(|shadow| shadow.get("sample_count"))
            .and_then(Value::as_u64)
            == observed("shadow_sample_count").and_then(Value::as_u64)
        && shadow_evaluation
            .and_then(|shadow| shadow.get("metrics"))
            .and_then(|metrics| metrics.get("candidate_inference_failures"))
            .and_then(Value::as_u64)
            == Some(0)
        && observed("offline_sample_count")
            .and_then(Value::as_u64)
            .is_some_and(|count| count >= 30)
        && threshold("offline_sample_count").and_then(Value::as_u64) == Some(30)
        && observed("shadow_sample_count")
            .and_then(Value::as_u64)
            .is_some_and(|count| count >= 20)
        && threshold("shadow_sample_count").and_then(Value::as_u64) == Some(20)
        && observed("macro_f1_non_inferiority")
            .and_then(Value::as_f64)
            .is_some_and(|delta| delta >= -0.02)
        && threshold("macro_f1_non_inferiority")
            .and_then(Value::as_f64)
            .is_some_and(|limit| (limit + 0.02).abs() < f64::EPSILON)
        && observed("critical_class_regressions").and_then(Value::as_u64) == Some(0)
        && threshold("critical_class_regressions").and_then(Value::as_u64) == Some(0)
        && observed("shadow_correction_rate_delta")
            .and_then(Value::as_f64)
            .is_some_and(|delta| delta <= 0.05)
        && threshold("shadow_correction_rate_delta")
            .and_then(Value::as_f64)
            .is_some_and(|limit| (limit - 0.05).abs() < f64::EPSILON)
        && observed("candidate_shadow_inference_failures").and_then(Value::as_u64) == Some(0)
        && threshold("candidate_shadow_inference_failures").and_then(Value::as_u64) == Some(0)
}

#[cfg(test)]
mod promotion_safety_tests {
    use super::{candidate_evaluation_is_promotable, synthetic_candidate_can_be_promoted};
    use serde_json::{json, Value};

    fn passing_evaluation() -> Value {
        json!({
            "schema_version": "candidate-evaluation.v1",
            "status": "COMPLETED",
            "cycle_id": "cycle-1",
            "candidate_model_version": "candidate-1",
            "production_model_version": "production-1",
            "candidate_dataset_version": "dataset-1",
            "evaluation_version": "candidate-evaluation.v1",
            "policy_version": "policy-v1",
            "promotion_policy": {
                "version": "policy-v1",
                "thresholds": {
                    "minimum_offline_samples": 30,
                    "minimum_shadow_samples": 20,
                    "maximum_macro_f1_regression": 0.02,
                    "maximum_class_f1_regression": 0.05,
                    "maximum_shadow_correction_rate_delta": 0.05,
                    "maximum_shadow_inference_failures": 0
                }
            },
            "offline_evaluation": {
                "status": "COMPLETED",
                "model_version": "candidate-1",
                "dataset_version": "holdout-1",
                "sample_count": 30,
                "synthetic": false,
                "critical_regressions": [],
                "metrics": {"macro_f1": 0.8, "per_class_changes": {"water": 0.0}}
            },
            "baseline_evaluation": {
                "status": "COMPLETED",
                "model_version": "production-1",
                "dataset_version": "holdout-1",
                "sample_count": 30,
                "synthetic": false,
                "critical_regressions": [],
                "metrics": {"macro_f1": 0.8}
            },
            "shadow_evaluation": {
                "sample_count": 20,
                "critical_regressions": [],
                "correction_rate_delta": 0.0,
                "metrics": {"candidate_inference_failures": 0}
            },
            "decision": "PENDING_HUMAN_DECISION",
            "synthetic": false,
            "gates": [
                {"key": "offline_sample_count", "status": "PASSED", "observed": 30, "threshold": 30},
                {"key": "shadow_sample_count", "status": "PASSED", "observed": 20, "threshold": 20},
                {"key": "macro_f1_non_inferiority", "status": "PASSED", "observed": 0.0, "threshold": -0.02},
                {"key": "critical_class_regressions", "status": "PASSED", "observed": 0, "threshold": 0},
                {"key": "shadow_correction_rate_delta", "status": "PASSED", "observed": 0.0, "threshold": 0.05},
                {"key": "candidate_shadow_inference_failures", "status": "PASSED", "observed": 0, "threshold": 0},
                {"key": "synthetic_evidence", "status": "PASSED"}
            ]
        })
    }

    #[test]
    fn synthetic_candidates_cannot_be_promoted_in_production() {
        assert!(!synthetic_candidate_can_be_promoted(true, "production"));
        assert!(!synthetic_candidate_can_be_promoted(true, "PROD"));
        assert!(synthetic_candidate_can_be_promoted(true, "demo"));
        assert!(synthetic_candidate_can_be_promoted(false, "production"));
    }

    #[test]
    fn candidate_evaluation_requires_every_policy_gate_and_matching_lineage() {
        let mut evaluation = passing_evaluation();
        assert!(candidate_evaluation_is_promotable(
            &evaluation,
            "cycle-1",
            "candidate-1",
            "production-1",
            "dataset-1",
            "holdout-1",
            "policy-v1"
        ));
        assert!(!candidate_evaluation_is_promotable(
            &evaluation,
            "cycle-1",
            "candidate-1",
            "production-1",
            "dataset-1",
            "different-holdout",
            "policy-v1"
        ));

        evaluation["gates"][0]["status"] = json!("INSUFFICIENT_EVIDENCE");
        assert!(!candidate_evaluation_is_promotable(
            &evaluation,
            "cycle-1",
            "candidate-1",
            "production-1",
            "dataset-1",
            "holdout-1",
            "policy-v1"
        ));
    }
}

#[derive(Debug)]
pub enum ImportError {
    Invalid(String),
    Conflict(String),
    Internal(String),
}

impl From<String> for ImportError {
    fn from(_message: String) -> Self {
        Self::Internal("dataset import failed".to_owned())
    }
}

fn import_row_error(row_number: usize, reason: &str, field: &str) -> ImportError {
    ImportError::Invalid(format!("row {row_number}: {reason}: {field}"))
}

pub fn default_qdrant_collection(embedder_version: &str, dimension: usize) -> String {
    if embedder_version == DEFAULT_EMBEDDER_VERSION && dimension == DEFAULT_EMBEDDING_DIMENSION {
        return DEFAULT_QDRANT_COLLECTION.to_owned();
    }
    let suffix = embedder_version
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_owned();
    format!(
        "pulse109_{}_d{dimension}",
        if suffix.is_empty() {
            "embedder"
        } else {
            &suffix
        }
    )
}

fn qdrant_distance_name(value: &str) -> Result<String, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "cosine" => Ok("Cosine".to_owned()),
        "dot" => Ok("Dot".to_owned()),
        "euclid" | "euclidean" => Ok("Euclid".to_owned()),
        _ => Err("embedding distance must be cosine, dot, or euclidean".to_owned()),
    }
}

pub(crate) fn safe_trace_id(value: &str) -> String {
    const PREFIX: &str = "trace-";
    if let Some(digest) = value.strip_prefix(PREFIX) {
        if digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return value.to_owned();
        }
    }
    format!("{PREFIX}{:x}", Sha256::digest(value.as_bytes()))
}

fn readiness_check(status: &str, error_code: Option<&str>) -> Value {
    let mut check = json!({"status": status});
    if let Some(error_code) = error_code {
        check["error_code"] = json!(error_code);
    }
    check
}

fn migration_readiness(rows: &[(i64, bool)], expected_versions: &[i64]) -> Value {
    let expected = expected_versions
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    let applied = rows
        .iter()
        .filter_map(|(version, success)| success.then_some(*version))
        .collect::<std::collections::BTreeSet<_>>();
    let failed_migrations = rows.iter().filter(|(_, success)| !success).count();
    let pending_migrations = expected.difference(&applied).count();
    let unexpected_migrations = applied.difference(&expected).count();
    let error_code = if failed_migrations > 0 {
        Some("DATABASE_MIGRATION_FAILED")
    } else if unexpected_migrations > 0 {
        Some("DATABASE_MIGRATION_VERSION_MISMATCH")
    } else if pending_migrations > 0 {
        Some("DATABASE_MIGRATIONS_PENDING")
    } else {
        None
    };
    let status = if error_code.is_some() {
        "not_ready"
    } else {
        "ready"
    };
    json!({
        "status": status,
        "error_code": error_code,
        "applied_migrations": applied.len(),
        "failed_migrations": failed_migrations,
        "expected_migrations": expected.len(),
        "pending_migrations": pending_migrations,
        "unexpected_migrations": unexpected_migrations,
        "latest_applied_version": applied.iter().next_back(),
    })
}

fn readiness_checks_are_ready(checks: &serde_json::Map<String, Value>) -> bool {
    !checks.is_empty()
        && checks
            .values()
            .all(|check| check.get("status").and_then(Value::as_str) == Some("ready"))
}

fn qdrant_collection_error_code(error: &str) -> &'static str {
    if error.contains("missing") {
        "QDRANT_COLLECTION_MISSING"
    } else if error.contains("does not match") || error.contains("config") {
        "QDRANT_COLLECTION_MISMATCH"
    } else {
        "QDRANT_COLLECTION_UNAVAILABLE"
    }
}

fn remote_model_check(body: &Value) -> Value {
    let Some(check) = body
        .get("checks")
        .and_then(|checks| checks.get("model_artifact"))
    else {
        return readiness_check("not_ready", Some("ML_READINESS_CHECK_MISSING"));
    };
    let status = match check.get("status").and_then(Value::as_str) {
        Some("ready") => "ready",
        Some("not_ready") => "not_ready",
        _ => return readiness_check("not_ready", Some("ML_READINESS_CHECK_INVALID")),
    };
    let error_code = check
        .get("error_code")
        .and_then(Value::as_str)
        .filter(|code| {
            matches!(
                *code,
                "MODEL_MANIFEST_UNAVAILABLE_OR_INVALID"
                    | "MODEL_RUNTIME_ADAPTER_NOT_CONFIGURED"
                    | "MODEL_MANIFEST_RUNTIME_MISMATCH"
                    | "INVALID_PULSE_ENV"
            )
        });
    readiness_check(status, error_code)
}

#[cfg(test)]
mod observability_readiness_tests {
    use super::{
        migration_readiness, qdrant_collection_error_code, remote_model_check, safe_trace_id,
    };
    use serde_json::json;

    #[test]
    fn trace_ids_are_pseudonymized_idempotently() {
        let safe = safe_trace_id("user-provided-trace");
        assert!(safe.starts_with("trace-"));
        assert_eq!(safe.len(), "trace-".len() + 64);
        assert_eq!(safe_trace_id(&safe), safe);
        assert_ne!(safe, "user-provided-trace");
    }

    #[test]
    fn migrations_must_match_the_embedded_version_set() {
        let pending = migration_readiness(&[(1, true)], &[1, 2]);
        assert_eq!(pending["status"], "not_ready");
        assert_eq!(pending["error_code"], "DATABASE_MIGRATIONS_PENDING");

        let failed = migration_readiness(&[(1, true), (2, false)], &[1, 2]);
        assert_eq!(failed["error_code"], "DATABASE_MIGRATION_FAILED");

        let unexpected = migration_readiness(&[(1, true), (3, true)], &[1, 2]);
        assert_eq!(
            unexpected["error_code"],
            "DATABASE_MIGRATION_VERSION_MISMATCH"
        );

        let ready = migration_readiness(&[(1, true), (2, true)], &[1, 2]);
        assert_eq!(
            ready,
            json!({
                "status": "ready",
                "error_code": null,
                "applied_migrations": 2,
                "failed_migrations": 0,
                "expected_migrations": 2,
                "pending_migrations": 0,
                "unexpected_migrations": 0,
                "latest_applied_version": 2
            })
        );
    }

    #[test]
    fn dependency_errors_are_reduced_to_safe_codes() {
        assert_eq!(
            qdrant_collection_error_code(
                "Qdrant collection dimension does not match active embedder"
            ),
            "QDRANT_COLLECTION_MISMATCH"
        );
        assert_eq!(
            qdrant_collection_error_code("active Qdrant collection is missing"),
            "QDRANT_COLLECTION_MISSING"
        );
        let model = remote_model_check(&json!({
            "checks": {
                "model_artifact": {
                    "status": "not_ready",
                    "error_code": "MODEL_RUNTIME_ADAPTER_NOT_CONFIGURED"
                }
            }
        }));
        assert_eq!(model["status"], "not_ready");
        assert_eq!(model["error_code"], "MODEL_RUNTIME_ADAPTER_NOT_CONFIGURED");
        assert_eq!(
            remote_model_check(&json!({"detail": "do not expose this"}))["error_code"],
            "ML_READINESS_CHECK_MISSING"
        );
    }
}

fn qdrant_score_to_similarity(score: f32, distance_metric: &str) -> Option<f32> {
    if !score.is_finite() {
        return None;
    }
    match distance_metric {
        "Cosine" | "Dot" => Some(score.clamp(0.0, 1.0)),
        // The embedder normalizes vectors, so Euclidean distance maps to cosine similarity.
        "Euclid" => Some((1.0 - score.powi(2) / 2.0).clamp(0.0, 1.0)),
        _ => None,
    }
}

fn ml_distance_name(value: &str) -> Result<&'static str, String> {
    match value {
        "Cosine" => Ok("cosine"),
        "Dot" => Ok("dot"),
        "Euclid" => Ok("euclidean"),
        _ => Err("active vector index has an unsupported distance metric".to_owned()),
    }
}

fn qdrant_vector_config(info: &Value) -> Result<(usize, String), String> {
    let vectors = info
        .pointer("/result/config/params/vectors")
        .ok_or_else(|| "Qdrant collection has no single unnamed vector config".to_owned())?;
    let dimension = vectors
        .get("size")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| "Qdrant collection vector dimension is unavailable".to_owned())?;
    let distance = vectors
        .get("distance")
        .and_then(Value::as_str)
        .ok_or_else(|| "Qdrant collection vector distance is unavailable".to_owned())?;
    let distance = qdrant_distance_name(distance)?;
    Ok((dimension, distance))
}

fn validate_qdrant_vector_config(info: &Value, expected: &VectorIndexConfig) -> Result<(), String> {
    let (dimension, distance) = qdrant_vector_config(info)?;
    if dimension != expected.dimension() {
        return Err("Qdrant collection dimension does not match active embedder".to_owned());
    }
    if distance != expected.distance_metric {
        return Err("Qdrant collection distance does not match active embedder".to_owned());
    }
    Ok(())
}

#[derive(Clone)]
pub struct PgRepository {
    pub pool: PgPool,
    pub qdrant_url: String,
    pub ml_service_url: String,
    pub qdrant_collection: String,
    pub embedding_dimension: usize,
    pub embedding_distance: String,
    pub embedder_version: String,
    pub forecast_model_version: String,
    client: Client,
}

#[derive(Debug, Serialize)]
pub struct AuditLogEvent {
    pub id: i64,
    pub actor_id: Option<String>,
    pub action: String,
    pub entity_type: String,
    pub entity_id: Option<String>,
    pub request_id: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
pub struct AuditLogPage {
    pub items: Vec<AuditLogEvent>,
    pub total: i64,
    pub limit: i64,
    pub offset: i64,
}

// Caller-provided request IDs may contain user data; only Core IDs are exposed in the audit API.
pub(crate) fn safe_audit_request_id(value: Option<String>) -> Option<String> {
    let value = value?;
    let suffix = value.strip_prefix("core-")?;
    let (timestamp, sequence) = suffix.split_once('-')?;
    let is_decimal =
        |part: &str| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit());
    (is_decimal(timestamp) && is_decimal(sequence)).then_some(value)
}

#[derive(Debug, Deserialize)]
struct MlClassifyResponse {
    model_version: String,
    predictions: Vec<MlClassification>,
}

#[derive(Debug)]
struct CandidateShadowWindow {
    learning_cycle_id: i64,
    candidate_model_version: String,
    production_model_version: Option<String>,
    artifact_checksum: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MlClassification {
    language: String,
    topic_id: String,
    topic: String,
    #[serde(default)]
    confidence: f32,
    #[serde(default)]
    confidence_state: String,
    #[serde(default)]
    needs_review: bool,
    #[serde(default)]
    alternatives: Vec<MlAlternative>,
}

#[derive(Debug, Deserialize, Serialize)]
struct MlAlternative {
    topic_id: String,
    #[serde(default)]
    topic: String,
    #[serde(default)]
    score: f32,
    #[serde(default)]
    confidence: f32,
}

#[derive(Debug, Deserialize)]
struct MlEmbedResponse {
    model_version: String,
    dimension: usize,
    #[serde(default)]
    embeddings: Vec<Vec<f32>>,
    embedding: Option<Vec<f32>>,
}

#[derive(Debug, FromRow)]
struct DbTicket {
    id: i64,
    external_ticket_id: String,
    original_text: String,
    language: String,
    region_id: String,
    region_name: String,
    topic_id: String,
    topic_label: String,
    priority: String,
    status: String,
    source_system: String,
    created_at: DateTime<Utc>,
    closed_at: Option<DateTime<Utc>>,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, FromRow)]
struct DbContextHandoffLocation {
    district: Option<String>,
    address: Option<String>,
    object: Option<String>,
    attachment_count: i64,
}

#[derive(Debug, FromRow)]
struct DbAnalyticsDrilldownTicket {
    id: i64,
    region_id: String,
    region_name: String,
    topic_id: String,
    topic_label: String,
    priority: String,
    status: String,
    created_at: DateTime<Utc>,
}

#[derive(Debug, FromRow)]
struct DbPrediction {
    ticket_id: i64,
    model_version: String,
    topic_id: Option<String>,
    topic_label: String,
    service_name: String,
    priority: String,
    confidence: Option<f64>,
    alternatives: Value,
    prediction: Value,
    needs_review: bool,
    created_at: DateTime<Utc>,
}

#[derive(Debug, FromRow)]
struct DbDecision {
    id: i64,
    ticket_id: i64,
    decision: String,
    predicted_topic_id: Option<String>,
    confirmed_topic_id: Option<String>,
    confirmed_topic_label: Option<String>,
    confirmed_service_id: Option<String>,
    confirmed_priority: String,
    service: Option<String>,
    feedback: Value,
    user_id: String,
    note: Option<String>,
    created_at: DateTime<Utc>,
}

#[derive(Debug, FromRow)]
struct DbRoutingFeedback {
    id: i64,
    ticket_id: i64,
    operator_decision_id: i64,
    original_route_recommendation: String,
    operator_confirmed_route: String,
    service_feedback: String,
    corrected_target_service: Option<String>,
    actor_user_id: String,
    source_system: String,
    evaluation_status: String,
    created_at: DateTime<Utc>,
}

impl From<DbRoutingFeedback> for RoutingFeedbackRecord {
    fn from(row: DbRoutingFeedback) -> Self {
        Self {
            id: row.id.to_string(),
            ticket_id: row.ticket_id.to_string(),
            operator_decision_id: format!("decision-{}", row.operator_decision_id),
            original_route_recommendation: row.original_route_recommendation,
            operator_confirmed_route: row.operator_confirmed_route,
            service_feedback: row.service_feedback,
            corrected_target_service: row.corrected_target_service,
            actor_user_id: row.actor_user_id,
            source_system: row.source_system,
            evaluation_status: row.evaluation_status,
            created_at: row.created_at.to_rfc3339(),
        }
    }
}

#[derive(Debug, FromRow)]
struct DbResponseTemplate {
    id: i64,
    template_key: String,
    language: String,
    topic_id: Option<String>,
    service_id: Option<String>,
    body: String,
    approved: bool,
    version: i32,
    created_by: Option<String>,
    updated_by: Option<String>,
    approved_by: Option<String>,
    approved_at: Option<DateTime<Utc>>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

impl From<DbResponseTemplate> for ResponseTemplateRecord {
    fn from(row: DbResponseTemplate) -> Self {
        Self {
            id: row.id.to_string(),
            template_key: row.template_key,
            language: row.language,
            topic_id: row.topic_id,
            service_id: row.service_id,
            body: row.body,
            approved: row.approved,
            version: row.version,
            created_by: row.created_by,
            updated_by: row.updated_by,
            approved_by: row.approved_by,
            approved_at: row.approved_at.map(|value| value.to_rfc3339()),
            created_at: row.created_at.to_rfc3339(),
            updated_at: row.updated_at.to_rfc3339(),
        }
    }
}

#[derive(Debug)]
struct QdrantHit {
    id: i64,
    score: f32,
    topic_id: String,
    region_id: String,
    created_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, FromRow)]
struct VectorIndexConfig {
    embedder_version: String,
    embedding_dimension: i32,
    distance_metric: String,
    collection_name: String,
    generation: i64,
}

impl VectorIndexConfig {
    fn dimension(&self) -> usize {
        self.embedding_dimension as usize
    }
}

fn assist_language_state(value: Option<&str>) -> String {
    match value
        .unwrap_or_default()
        .trim()
        .to_ascii_uppercase()
        .as_str()
    {
        "RU" | "RUS" => "RU".to_owned(),
        "KZ" | "KK" | "KAZ" => "KZ".to_owned(),
        "MIXED" => "MIXED".to_owned(),
        _ => "UNKNOWN".to_owned(),
    }
}

fn response_template_language_supported(language_state: &str) -> bool {
    matches!(language_state, "RU" | "KZ")
}

fn assist_stage(
    name: &str,
    status: &str,
    latency_ms: f64,
    model_version: Option<String>,
    error_code: Option<&str>,
) -> AssistStage {
    AssistStage {
        name: name.to_owned(),
        status: status.to_owned(),
        latency_ms,
        model_version,
        error_code: error_code.map(ToOwned::to_owned),
    }
}

fn unavailable_assist_prediction(ticket: &Ticket) -> Prediction {
    Prediction {
        ticket_id: ticket.id.clone(),
        model_version: "unavailable".to_owned(),
        topic_id: "unknown".to_owned(),
        topic_label: "Не определено".to_owned(),
        confidence: 0.0,
        confidence_state: "low".to_owned(),
        recommended_service: "UNKNOWN".to_owned(),
        predicted_priority: "UNKNOWN".to_owned(),
        routing_reason: "Требуется ручная проверка".to_owned(),
        service_provenance: RuleProvenance::manual(
            "Маршрутизация недоступна; выберите службу вручную",
        ),
        priority_provenance: RuleProvenance::manual("Приоритет недоступен; выберите его вручную"),
        alternatives: Vec::new(),
        created_at: ticket.created_at.clone(),
    }
}

#[derive(Debug, Deserialize)]
struct MlEmbedderMetadata {
    model_version: String,
    dimension: Option<usize>,
    distance_metric: Option<String>,
}

#[derive(Debug, Clone)]
struct RoutingDecision {
    service_id: String,
    service_name: String,
    service_provenance: RuleProvenance,
    priority: String,
    priority_provenance: RuleProvenance,
    reason: String,
}

const ROUTING_RULE_LOOKUP: &str = "SELECT s.id, s.name_ru, rr.source, rr.reason, rr.version FROM routing_rules rr JOIN services s ON s.id = rr.service_id WHERE rr.active AND (rr.topic_id = $1 OR rr.topic_id IS NULL) AND (rr.region_id = $2 OR rr.region_id IS NULL) ORDER BY CASE rr.source WHEN 'OFFICIAL' THEN 0 WHEN 'LABEL_HISTORY' THEN 1 WHEN 'MANUAL' THEN 2 ELSE 3 END, CASE WHEN rr.region_id IS NULL THEN 1 ELSE 0 END, rr.version DESC, rr.precedence ASC, rr.id ASC LIMIT 1";
const PRIORITY_RULE_LOOKUP: &str = "SELECT priority, source, reason, version FROM priority_rules WHERE active AND (topic_id = $1 OR topic_id IS NULL) AND (region_id = $2 OR region_id IS NULL) ORDER BY CASE source WHEN 'OFFICIAL' THEN 0 WHEN 'LABEL_HISTORY' THEN 1 WHEN 'MANUAL' THEN 2 ELSE 3 END, CASE WHEN region_id IS NULL THEN 1 ELSE 0 END, version DESC, precedence ASC, id ASC LIMIT 1";

const TICKET_SELECT: &str = r#"
SELECT
    t.id,
    t.external_ticket_id,
    t.original_text,
    t.language,
    t.region_id,
    COALESCE(r.name_ru, r.name_en, t.region_id) AS region_name,
    t.topic_id,
    COALESCE(tp.name_ru, tp.name_kk, t.topic_id) AS topic_label,
    COALESCE(t.priority, 'normal') AS priority,
    t.status,
    t.source_system,
    t.created_at,
    t.closed_at,
    t.updated_in_pulse_at AS updated_at
FROM tickets t
LEFT JOIN regions r ON r.id = t.region_id
LEFT JOIN topics tp ON tp.id = t.topic_id
"#;

const PREDICTION_SELECT: &str = r#"
SELECT
    p.ticket_id,
    p.model_version,
    p.topic_id,
    COALESCE(tp.name_ru, tp.name_kk, p.topic_id, 'Не определено') AS topic_label,
    COALESCE(s.name_ru, 'Другая служба') AS service_name,
    COALESCE(p.priority, 'normal') AS priority,
    p.confidence::float8 AS confidence,
    p.alternatives,
    p.prediction,
    p.needs_review,
    p.created_at
FROM ticket_predictions p
LEFT JOIN topics tp ON tp.id = p.topic_id
LEFT JOIN services s ON s.id = p.service_id
"#;

impl PgRepository {
    pub fn connect_lazy(
        database_url: &str,
        qdrant_url: impl Into<String>,
        ml_service_url: impl Into<String>,
    ) -> Result<Self, String> {
        Self::connect_lazy_with_options(
            database_url,
            qdrant_url,
            ml_service_url,
            DEFAULT_QDRANT_COLLECTION,
            DEFAULT_EMBEDDING_DIMENSION,
            DEFAULT_EMBEDDER_VERSION,
        )
    }

    pub fn connect_lazy_with_options(
        database_url: &str,
        qdrant_url: impl Into<String>,
        ml_service_url: impl Into<String>,
        qdrant_collection: impl Into<String>,
        embedding_dimension: usize,
        embedder_version: impl Into<String>,
    ) -> Result<Self, String> {
        if !(8..=1024).contains(&embedding_dimension) {
            return Err("embedding dimension must be between 8 and 1024".to_owned());
        }
        let embedding_distance = qdrant_distance_name(
            &env::var("EMBEDDING_DISTANCE_METRIC").unwrap_or_else(|_| "cosine".to_owned()),
        )?;
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .min_connections(1)
            .connect_lazy(database_url)
            .map_err(|error| format!("invalid DATABASE_URL: {error}"))?;
        Ok(Self {
            pool,
            qdrant_url: qdrant_url.into().trim_end_matches('/').to_owned(),
            ml_service_url: ml_service_url.into().trim_end_matches('/').to_owned(),
            qdrant_collection: qdrant_collection.into(),
            embedding_dimension,
            embedding_distance,
            embedder_version: embedder_version.into(),
            forecast_model_version: env::var("FORECAST_MODEL_VERSION")
                .unwrap_or_else(|_| DEFAULT_FORECAST_MODEL_VERSION.to_owned()),
            client: Client::builder()
                .build()
                .map_err(|error| format!("build HTTP client: {error}"))?,
        })
    }

    pub async fn initialize(&self) -> Result<(), String> {
        MIGRATOR
            .run(&self.pool)
            .await
            .map_err(|error| format!("apply migrations: {error}"))?;
        self.initialize_vector_index_state().await?;
        let active_index = self.active_vector_index().await?;
        self.ensure_qdrant_collection(&active_index, true).await
    }

    pub async fn readiness(&self) -> Value {
        let mut checks = serde_json::Map::new();
        let postgres_ready = matches!(
            tokio::time::timeout(
                READINESS_DEPENDENCY_TIMEOUT,
                sqlx::query("SELECT 1").fetch_one(&self.pool),
            )
            .await,
            Ok(Ok(_))
        );
        checks.insert(
            "postgres".to_owned(),
            readiness_check(
                if postgres_ready { "ready" } else { "not_ready" },
                (!postgres_ready).then_some("POSTGRES_UNAVAILABLE"),
            ),
        );

        let expected_versions = MIGRATOR
            .iter()
            .map(|migration| migration.version)
            .collect::<Vec<_>>();
        if postgres_ready {
            match tokio::time::timeout(
                READINESS_DEPENDENCY_TIMEOUT,
                sqlx::query_as::<_, (i64, bool)>(
                    "SELECT version, success FROM _sqlx_migrations ORDER BY version",
                )
                .fetch_all(&self.pool),
            )
            .await
            {
                Ok(Ok(rows)) => {
                    checks.insert(
                        "database_migrations".to_owned(),
                        migration_readiness(&rows, &expected_versions),
                    );
                }
                _ => {
                    checks.insert(
                        "database_migrations".to_owned(),
                        readiness_check("not_ready", Some("DATABASE_MIGRATION_STATUS_UNAVAILABLE")),
                    );
                }
            }
        } else {
            checks.insert(
                "database_migrations".to_owned(),
                readiness_check("not_evaluated", Some("POSTGRES_UNAVAILABLE")),
            );
        }

        let active_index = if postgres_ready {
            match tokio::time::timeout(
                READINESS_DEPENDENCY_TIMEOUT,
                self.active_vector_index_optional(),
            )
            .await
            {
                Ok(Ok(Some(index))) => {
                    let matches_configuration = index.embedder_version == self.embedder_version
                        && index.dimension() == self.embedding_dimension
                        && index.distance_metric == self.embedding_distance;
                    checks.insert(
                        "vector_index".to_owned(),
                        readiness_check(
                            if matches_configuration {
                                "ready"
                            } else {
                                "not_ready"
                            },
                            (!matches_configuration)
                                .then_some("VECTOR_INDEX_CONFIGURATION_MISMATCH"),
                        ),
                    );
                    Some(index)
                }
                Ok(Ok(None)) => {
                    checks.insert(
                        "vector_index".to_owned(),
                        readiness_check("not_ready", Some("VECTOR_INDEX_NOT_CONFIGURED")),
                    );
                    None
                }
                _ => {
                    checks.insert(
                        "vector_index".to_owned(),
                        readiness_check("not_ready", Some("VECTOR_INDEX_UNAVAILABLE")),
                    );
                    None
                }
            }
        } else {
            checks.insert(
                "vector_index".to_owned(),
                readiness_check("not_evaluated", Some("POSTGRES_UNAVAILABLE")),
            );
            None
        };

        let qdrant_ready = match tokio::time::timeout(
            READINESS_DEPENDENCY_TIMEOUT,
            self.client
                .get(format!("{}/readyz", self.qdrant_url))
                .send(),
        )
        .await
        {
            Ok(Ok(response)) if response.status().is_success() => {
                checks.insert("qdrant".to_owned(), readiness_check("ready", None));
                true
            }
            _ => {
                checks.insert(
                    "qdrant".to_owned(),
                    readiness_check("not_ready", Some("QDRANT_UNAVAILABLE")),
                );
                false
            }
        };
        match (active_index.as_ref(), qdrant_ready) {
            (Some(index), true) => match tokio::time::timeout(
                READINESS_DEPENDENCY_TIMEOUT,
                self.ensure_qdrant_collection(index, false),
            )
            .await
            {
                Ok(Ok(())) => {
                    checks.insert(
                        "qdrant_collection".to_owned(),
                        readiness_check("ready", None),
                    );
                }
                Ok(Err(error)) => {
                    let code = qdrant_collection_error_code(&error);
                    checks.insert(
                        "qdrant_collection".to_owned(),
                        readiness_check("not_ready", Some(code)),
                    );
                }
                Err(_) => {
                    checks.insert(
                        "qdrant_collection".to_owned(),
                        readiness_check("not_ready", Some("QDRANT_COLLECTION_UNAVAILABLE")),
                    );
                }
            },
            (None, _) => {
                checks.insert(
                    "qdrant_collection".to_owned(),
                    readiness_check("not_evaluated", Some("VECTOR_INDEX_UNAVAILABLE")),
                );
            }
            (_, false) => {
                checks.insert(
                    "qdrant_collection".to_owned(),
                    readiness_check("not_evaluated", Some("QDRANT_UNAVAILABLE")),
                );
            }
        }

        let ml_response = tokio::time::timeout(
            READINESS_DEPENDENCY_TIMEOUT,
            self.client
                .get(format!("{}/readyz", self.ml_service_url))
                .send(),
        )
        .await;
        let mut ml_body = None;
        let ml_ready = match ml_response {
            Ok(Ok(response)) => {
                let status_is_success = response.status().is_success();
                match tokio::time::timeout(READINESS_DEPENDENCY_TIMEOUT, response.json::<Value>())
                    .await
                {
                    Ok(Ok(body)) => {
                        let status_is_ready =
                            body.get("status").and_then(Value::as_str) == Some("ready");
                        let ready = status_is_success && status_is_ready;
                        checks.insert(
                            "ml_service".to_owned(),
                            readiness_check(
                                if ready { "ready" } else { "not_ready" },
                                (!ready).then_some("ML_SERVICE_NOT_READY"),
                            ),
                        );
                        ml_body = Some(body);
                        ready
                    }
                    _ => {
                        checks.insert(
                            "ml_service".to_owned(),
                            readiness_check("not_ready", Some("ML_READINESS_RESPONSE_INVALID")),
                        );
                        false
                    }
                }
            }
            _ => {
                checks.insert(
                    "ml_service".to_owned(),
                    readiness_check("not_ready", Some("ML_SERVICE_UNAVAILABLE")),
                );
                false
            }
        };
        let model_artifact = ml_body
            .as_ref()
            .map(remote_model_check)
            .unwrap_or_else(|| readiness_check("not_evaluated", Some("ML_SERVICE_UNAVAILABLE")));
        checks.insert("model_artifact".to_owned(), model_artifact);

        if let (Some(index), true) = (active_index.as_ref(), ml_ready) {
            let metadata_response = tokio::time::timeout(
                READINESS_DEPENDENCY_TIMEOUT,
                self.client
                    .get(format!(
                        "{}/internal/v1/models/embedder",
                        self.ml_service_url
                    ))
                    .send(),
            )
            .await;
            match metadata_response {
                Ok(Ok(response)) if response.status().is_success() => {
                    match tokio::time::timeout(
                        READINESS_DEPENDENCY_TIMEOUT,
                        response.json::<MlEmbedderMetadata>(),
                    )
                    .await
                    {
                        Ok(Ok(embedder)) => {
                            let expected_distance =
                                ml_distance_name(&index.distance_metric).unwrap_or_default();
                            let matches_index = embedder.model_version == index.embedder_version
                                && embedder.dimension == Some(index.dimension())
                                && embedder.distance_metric.as_deref() == Some(expected_distance);
                            checks.insert(
                                "ml_embedder".to_owned(),
                                readiness_check(
                                    if matches_index { "ready" } else { "not_ready" },
                                    (!matches_index)
                                        .then_some("ML_EMBEDDER_CONFIGURATION_MISMATCH"),
                                ),
                            );
                        }
                        _ => {
                            checks.insert(
                                "ml_embedder".to_owned(),
                                readiness_check("not_ready", Some("ML_EMBEDDER_METADATA_INVALID")),
                            );
                        }
                    }
                }
                _ => {
                    checks.insert(
                        "ml_embedder".to_owned(),
                        readiness_check("not_ready", Some("ML_EMBEDDER_METADATA_UNAVAILABLE")),
                    );
                }
            }
        } else {
            checks.insert(
                "ml_embedder".to_owned(),
                readiness_check("not_evaluated", Some("ML_SERVICE_NOT_READY")),
            );
        };

        json!({
            "status": if readiness_checks_are_ready(&checks) { "ready" } else { "not_ready" },
            "checks": checks,
        })
    }

    async fn initialize_vector_index_state(&self) -> Result<(), String> {
        if self.active_vector_index_optional().await?.is_some() {
            return Ok(());
        }

        let legacy = sqlx::query(
            "SELECT model_versions->>'embedder' AS embedder_version, split_part(embedding_ref, ':', 2) AS collection_name FROM tickets WHERE embedding_ref LIKE 'qdrant:%' AND model_versions ? 'embedder' GROUP BY 1, 2 ORDER BY COUNT(*) DESC, 1, 2 LIMIT 1",
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("read legacy vector index lineage: {error}"))?;

        let (embedder_version, embedding_dimension, distance_metric, collection_name) =
            if let Some(legacy) = legacy {
                let embedder_version: String = legacy
                    .try_get("embedder_version")
                    .map_err(|error| format!("read legacy embedder version: {error}"))?;
                let collection_name: String = legacy
                    .try_get("collection_name")
                    .map_err(|error| format!("read legacy collection name: {error}"))?;
                let info = self
                    .client
                    .get(format!("{}/collections/{collection_name}", self.qdrant_url))
                    .send()
                    .await
                    .map_err(|error| format!("read legacy Qdrant collection: {error}"))?
                    .error_for_status()
                    .map_err(|error| format!("read legacy Qdrant collection: {error}"))?
                    .json::<Value>()
                    .await
                    .map_err(|error| format!("legacy Qdrant collection JSON: {error}"))?;
                let (embedding_dimension, distance_metric) = qdrant_vector_config(&info)?;
                (
                    embedder_version,
                    embedding_dimension,
                    distance_metric,
                    collection_name,
                )
            } else {
                (
                    self.embedder_version.clone(),
                    self.embedding_dimension,
                    self.embedding_distance.clone(),
                    self.qdrant_collection.clone(),
                )
            };

        sqlx::query(
            "INSERT INTO vector_index_state (singleton_id, embedder_version, embedding_dimension, distance_metric, collection_name) VALUES (1, $1, $2, $3, $4) ON CONFLICT (singleton_id) DO NOTHING",
        )
        .bind(embedder_version)
        .bind(embedding_dimension as i32)
        .bind(distance_metric)
        .bind(collection_name)
        .execute(&self.pool)
        .await
        .map_err(|error| format!("initialize vector index state: {error}"))?;
        Ok(())
    }

    async fn active_vector_index_optional(&self) -> Result<Option<VectorIndexConfig>, String> {
        sqlx::query_as::<_, VectorIndexConfig>(
            "SELECT embedder_version, embedding_dimension, distance_metric, collection_name, generation FROM vector_index_state WHERE singleton_id = 1",
        )
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("read active vector index: {error}"))
    }

    async fn active_vector_index(&self) -> Result<VectorIndexConfig, String> {
        self.active_vector_index_optional()
            .await?
            .ok_or_else(|| "active vector index is not configured".to_owned())
    }

    async fn locked_vector_index(
        &self,
        transaction: &mut sqlx::Transaction<'_, Postgres>,
    ) -> Result<VectorIndexConfig, String> {
        sqlx::query_as::<_, VectorIndexConfig>(
            "SELECT embedder_version, embedding_dimension, distance_metric, collection_name, generation FROM vector_index_state WHERE singleton_id = 1 FOR SHARE",
        )
        .fetch_one(&mut **transaction)
        .await
        .map_err(|error| format!("lock active vector index: {error}"))
    }

    async fn ensure_qdrant_collection(
        &self,
        index: &VectorIndexConfig,
        create_missing: bool,
    ) -> Result<(), String> {
        let url = format!("{}/collections/{}", self.qdrant_url, index.collection_name);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|error| format!("qdrant collection check: {error}"))?;
        if response.status().is_success() {
            let info = response
                .json::<Value>()
                .await
                .map_err(|error| format!("qdrant collection JSON: {error}"))?;
            return validate_qdrant_vector_config(&info, index);
        }
        if response.status() != HttpStatus::NOT_FOUND {
            return Err(format!(
                "qdrant collection check returned {}",
                response.status()
            ));
        }
        if !create_missing {
            return Err("active Qdrant collection is missing".to_owned());
        }
        self.client
            .put(&url)
            .json(&json!({
                "vectors": {
                    "size": index.dimension(),
                    "distance": index.distance_metric,
                }
            }))
            .send()
            .await
            .map_err(|error| format!("qdrant collection create: {error}"))?
            .error_for_status()
            .map_err(|error| format!("qdrant collection create: {error}"))?;
        let info = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|error| format!("qdrant collection verify: {error}"))?
            .error_for_status()
            .map_err(|error| format!("qdrant collection verify: {error}"))?
            .json::<Value>()
            .await
            .map_err(|error| format!("qdrant collection verify JSON: {error}"))?;
        validate_qdrant_vector_config(&info, index)
    }

    async fn classify(
        &self,
        text: &str,
        language: Option<&str>,
        request_id: &str,
        trace_id: &str,
    ) -> Result<MlClassificationWithModel, String> {
        self.classify_versioned(text, language, request_id, trace_id, None, None)
            .await
    }

    async fn classify_versioned(
        &self,
        text: &str,
        language: Option<&str>,
        request_id: &str,
        trace_id: &str,
        model_version: Option<&str>,
        expected_artifact_checksum: Option<&str>,
    ) -> Result<MlClassificationWithModel, String> {
        let response = self
            .client
            .post(format!("{}/internal/v1/classify", self.ml_service_url))
            .header("x-request-id", request_id)
            .header("x-trace-id", safe_trace_id(trace_id))
            .json(&json!({
                "text": text,
                "language": language,
                "top_k": 3,
                "model_version": model_version,
                "expected_artifact_checksum": expected_artifact_checksum,
                "request_id": request_id,
            }))
            .send()
            .await
            .map_err(|error| format!("ML classify request: {error}"))?
            .error_for_status()
            .map_err(|error| format!("ML classify response: {error}"))?
            .json::<MlClassifyResponse>()
            .await
            .map_err(|error| format!("ML classify JSON: {error}"))?;
        let prediction = response
            .predictions
            .into_iter()
            .next()
            .ok_or_else(|| "ML classify returned no predictions".to_owned())?;
        Ok(MlClassificationWithModel {
            model_version: response.model_version,
            prediction,
        })
    }

    async fn shadow_window_for_ticket(
        &self,
        ticket_created_at: DateTime<Utc>,
    ) -> Result<Option<CandidateShadowWindow>, String> {
        let row = sqlx::query(
            "SELECT lc.id, lc.candidate_model_version, lc.production_model_version, mv.artifact_checksum FROM learning_cycles lc LEFT JOIN model_versions mv ON mv.model_version = lc.candidate_model_version WHERE lc.evaluation_started_at IS NOT NULL AND lc.evaluation_ends_at IS NOT NULL AND lc.evaluation_started_at <= $1 AND lc.evaluation_ends_at >= $1 ORDER BY lc.evaluation_started_at DESC, lc.id DESC LIMIT 1",
        )
        .bind(ticket_created_at)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("find candidate shadow window: {error}"))?;

        row.map(|row| {
            Ok(CandidateShadowWindow {
                learning_cycle_id: row
                    .try_get("id")
                    .map_err(|error| format!("shadow cycle id: {error}"))?,
                candidate_model_version: row
                    .try_get::<Option<String>, _>("candidate_model_version")
                    .map_err(|error| format!("shadow candidate version: {error}"))?
                    .ok_or_else(|| "shadow candidate version is unavailable".to_owned())?,
                production_model_version: row
                    .try_get("production_model_version")
                    .map_err(|error| format!("shadow production version: {error}"))?,
                artifact_checksum: row
                    .try_get("artifact_checksum")
                    .map_err(|error| format!("shadow artifact checksum: {error}"))?,
            })
        })
        .transpose()
    }

    async fn embed(
        &self,
        index: &VectorIndexConfig,
        text: &str,
        request_id: &str,
        trace_id: &str,
    ) -> Result<Vec<f32>, String> {
        let response = self
            .client
            .post(format!("{}/internal/v1/embed", self.ml_service_url))
            .header("x-request-id", request_id)
            .header("x-trace-id", safe_trace_id(trace_id))
            .json(&json!({
                "text": text,
                "dimension": index.dimension(),
                "normalize": true,
                "model_version": index.embedder_version,
            }))
            .send()
            .await
            .map_err(|error| format!("ML embed request: {error}"))?
            .error_for_status()
            .map_err(|error| format!("ML embed response: {error}"))?
            .json::<MlEmbedResponse>()
            .await
            .map_err(|error| format!("ML embed JSON: {error}"))?;
        if response.model_version != index.embedder_version {
            return Err("ML embed returned a different model version".to_owned());
        }
        if response.dimension != index.dimension() {
            return Err(format!(
                "ML embed returned dimension {}, expected {}",
                response.dimension,
                index.dimension()
            ));
        }
        let vector = response
            .embedding
            .or_else(|| response.embeddings.into_iter().next())
            .ok_or_else(|| "ML embed returned no vector".to_owned())?;
        if vector.len() != index.dimension() {
            return Err(format!(
                "ML embed returned dimension {}, expected {}",
                vector.len(),
                index.dimension()
            ));
        }
        Ok(vector)
    }

    async fn qdrant_upsert(
        &self,
        index: &VectorIndexConfig,
        ticket_id: i64,
        vector: &[f32],
        topic_id: &str,
        region_id: &str,
        created_at: &str,
    ) -> Result<(), String> {
        self.ensure_qdrant_collection(index, false).await?;
        self.client
            .put(format!(
                "{}/collections/{}/points?wait=true",
                self.qdrant_url, index.collection_name
            ))
            .json(&json!({
                "points": [{
                    "id": ticket_id,
                    "vector": vector,
                    "payload": {
                        "ticket_id": ticket_id.to_string(),
                        "topic_id": topic_id,
                        "region_id": region_id,
                        "created_at": created_at,
                    }
                }]
            }))
            .send()
            .await
            .map_err(|error| format!("qdrant upsert: {error}"))?
            .error_for_status()
            .map_err(|error| format!("qdrant upsert: {error}"))?;
        Ok(())
    }

    async fn qdrant_update_payload(
        &self,
        index: &VectorIndexConfig,
        ticket_id: i64,
        topic_id: &str,
        region_id: &str,
    ) -> Result<(), String> {
        self.ensure_qdrant_collection(index, false).await?;
        self.client
            .post(format!(
                "{}/collections/{}/points/payload?wait=true",
                self.qdrant_url, index.collection_name
            ))
            .json(&json!({
                "payload": {
                    "ticket_id": ticket_id.to_string(),
                    "topic_id": topic_id,
                    "region_id": region_id,
                },
                "points": [ticket_id]
            }))
            .send()
            .await
            .map_err(|error| format!("qdrant payload update: {error}"))?
            .error_for_status()
            .map_err(|error| format!("qdrant payload update: {error}"))?;
        Ok(())
    }

    async fn qdrant_delete(&self, index: &VectorIndexConfig, ticket_id: i64) -> Result<(), String> {
        self.ensure_qdrant_collection(index, false).await?;
        self.client
            .post(format!(
                "{}/collections/{}/points/delete?wait=true",
                self.qdrant_url, index.collection_name
            ))
            .json(&json!({"points": [ticket_id]}))
            .send()
            .await
            .map_err(|error| format!("qdrant vector delete: {error}"))?
            .error_for_status()
            .map_err(|error| format!("qdrant vector delete: {error}"))?;
        Ok(())
    }

    async fn qdrant_search(
        &self,
        index: &VectorIndexConfig,
        vector: &[f32],
        exclude_id: Option<i64>,
        request_id: &str,
        trace_id: &str,
    ) -> Result<Vec<QdrantHit>, String> {
        self.ensure_qdrant_collection(index, false).await?;
        let result = self
            .client
            .post(format!(
                "{}/collections/{}/points/search",
                self.qdrant_url, index.collection_name
            ))
            .header("x-request-id", request_id)
            .header("x-trace-id", safe_trace_id(trace_id))
            .json(&json!({
                "vector": vector,
                "limit": 5,
                "with_payload": true,
                "with_vector": false,
            }))
            .send()
            .await
            .map_err(|error| format!("qdrant search: {error}"))?
            .error_for_status()
            .map_err(|error| format!("qdrant search: {error}"))?
            .json::<Value>()
            .await
            .map_err(|error| format!("qdrant search JSON: {error}"))?;
        let mut hits = Vec::new();
        for item in result["result"].as_array().into_iter().flatten() {
            let id = item["id"].as_i64().or_else(|| {
                item["id"]
                    .as_str()
                    .and_then(|value| value.parse::<i64>().ok())
            });
            let Some(id) = id else { continue };
            if exclude_id == Some(id) {
                continue;
            }
            let score = item["score"].as_f64().unwrap_or_default() as f32;
            let payload = &item["payload"];
            hits.push(QdrantHit {
                id,
                score,
                topic_id: payload["topic_id"].as_str().unwrap_or("unknown").to_owned(),
                region_id: payload["region_id"]
                    .as_str()
                    .unwrap_or("unknown")
                    .to_owned(),
                created_at: payload["created_at"]
                    .as_str()
                    .and_then(|value| DateTime::parse_from_rfc3339(value).ok())
                    .map(|value| value.with_timezone(&Utc)),
            });
        }
        Ok(hits)
    }

    async fn search_assist_candidates(
        &self,
        text: &str,
        exclude_id: Option<i64>,
        request_id: &str,
        trace_id: &str,
    ) -> Result<(VectorIndexConfig, Vec<QdrantHit>), String> {
        let mut index_tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin similarity index read: {error}"))?;
        let active_index = self.locked_vector_index(&mut index_tx).await?;
        let vector = self
            .embed(&active_index, text, request_id, trace_id)
            .await?;
        let hits = self
            .qdrant_search(&active_index, &vector, exclude_id, request_id, trace_id)
            .await?;
        index_tx
            .commit()
            .await
            .map_err(|error| format!("finish similarity index read: {error}"))?;
        Ok((active_index, hits))
    }

    async fn resolve_routing(
        &self,
        topic_id: &str,
        region_id: &str,
        service_label: Option<&str>,
        requested_priority: Option<&str>,
        input_source: RuleSource,
    ) -> Result<RoutingDecision, String> {
        let service_row = sqlx::query(ROUTING_RULE_LOOKUP)
            .bind(topic_id)
            .bind(region_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("resolve routing rule: {error}"))?;
        let service_label_match = if let Some(label) =
            service_label.filter(|value| !value.trim().is_empty())
        {
            let row = sqlx::query(
                "SELECT id, name_ru FROM services WHERE active AND (id = $1 OR lower(name_ru) = lower($1)) LIMIT 1",
            )
            .bind(label.trim())
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("resolve service label: {error}"))?;
            row.map(|row| {
                Ok::<_, String>((
                    row.try_get::<String, _>("id")
                        .map_err(|error| format!("service id: {error}"))?,
                    row.try_get::<String, _>("name_ru")
                        .map_err(|error| format!("service name: {error}"))?,
                ))
            })
            .transpose()?
        } else {
            None
        };
        let (service_id, service_name, service_provenance) = if let Some(row) = service_row {
            let source = row
                .try_get::<String, _>("source")
                .map_err(|error| format!("routing source: {error}"))?;
            let rule_source = RuleSource::from_db(&source);
            if rule_source == RuleSource::Official {
                (
                    row.try_get::<String, _>("id")
                        .map_err(|error| format!("routing service id: {error}"))?,
                    row.try_get::<String, _>("name_ru")
                        .map_err(|error| format!("routing service name: {error}"))?,
                    RuleProvenance {
                        source: rule_source,
                        version: Some(
                            row.try_get::<i32, _>("version")
                                .map_err(|error| format!("routing rule version: {error}"))?,
                        ),
                        reason: row
                            .try_get::<String, _>("reason")
                            .map_err(|error| format!("routing reason: {error}"))?,
                        facts_used: routing_lookup_facts(topic_id, region_id),
                    },
                )
            } else if input_source == RuleSource::LabelHistory {
                if let Some((service_id, service_name)) = service_label_match {
                    (
                        service_id,
                        service_name,
                        RuleProvenance {
                            source: RuleSource::LabelHistory,
                            version: None,
                            reason: "Служба из исторической метки обращения".to_owned(),
                            facts_used: routing_lookup_facts(topic_id, region_id),
                        },
                    )
                } else {
                    (
                        row.try_get::<String, _>("id")
                            .map_err(|error| format!("routing service id: {error}"))?,
                        row.try_get::<String, _>("name_ru")
                            .map_err(|error| format!("routing service name: {error}"))?,
                        RuleProvenance {
                            source: rule_source,
                            version: Some(
                                row.try_get::<i32, _>("version")
                                    .map_err(|error| format!("routing rule version: {error}"))?,
                            ),
                            reason: row
                                .try_get::<String, _>("reason")
                                .map_err(|error| format!("routing reason: {error}"))?,
                            facts_used: routing_lookup_facts(topic_id, region_id),
                        },
                    )
                }
            } else {
                (
                    row.try_get::<String, _>("id")
                        .map_err(|error| format!("routing service id: {error}"))?,
                    row.try_get::<String, _>("name_ru")
                        .map_err(|error| format!("routing service name: {error}"))?,
                    RuleProvenance {
                        source: rule_source,
                        version: Some(
                            row.try_get::<i32, _>("version")
                                .map_err(|error| format!("routing rule version: {error}"))?,
                        ),
                        reason: row
                            .try_get::<String, _>("reason")
                            .map_err(|error| format!("routing reason: {error}"))?,
                        facts_used: routing_lookup_facts(topic_id, region_id),
                    },
                )
            }
        } else if let Some((service_id, service_name)) = service_label_match {
            (
                service_id,
                service_name,
                RuleProvenance {
                    source: input_source,
                    version: None,
                    reason: match input_source {
                        RuleSource::LabelHistory => {
                            "Служба из исторической метки обращения".to_owned()
                        }
                        RuleSource::Manual => "Служба введена вручную".to_owned(),
                        RuleSource::Official => "Официальная метка службы".to_owned(),
                    },
                    facts_used: routing_lookup_facts(topic_id, region_id),
                },
            )
        } else {
            (
                "service_other".to_owned(),
                "Другая служба".to_owned(),
                RuleProvenance::manual(
                    "Проверенное правило маршрутизации не предоставлено; выберите службу вручную",
                )
                .with_facts(routing_lookup_facts(topic_id, region_id)),
            )
        };

        let priority_row = sqlx::query(PRIORITY_RULE_LOOKUP)
            .bind(topic_id)
            .bind(region_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("resolve priority rule: {error}"))?;
        let (priority, priority_provenance) = if let Some(row) = priority_row {
            let source = row
                .try_get::<String, _>("source")
                .map_err(|error| format!("priority source: {error}"))?;
            let rule_source = RuleSource::from_db(&source);
            let historical_priority = requested_priority
                .filter(|value| !value.trim().is_empty())
                .filter(|_| input_source == RuleSource::LabelHistory);
            if rule_source != RuleSource::Official {
                if let Some(value) = historical_priority {
                    (
                        normalize_priority(value),
                        RuleProvenance {
                            source: RuleSource::LabelHistory,
                            version: None,
                            reason: "Приоритет из исторического значения обращения".to_owned(),
                            facts_used: routing_lookup_facts(topic_id, region_id),
                        },
                    )
                } else {
                    (
                        row.try_get::<String, _>("priority")
                            .map_err(|error| format!("priority value: {error}"))?,
                        RuleProvenance {
                            source: rule_source,
                            version: Some(
                                row.try_get::<i32, _>("version")
                                    .map_err(|error| format!("priority rule version: {error}"))?,
                            ),
                            reason: row
                                .try_get::<String, _>("reason")
                                .map_err(|error| format!("priority reason: {error}"))?,
                            facts_used: routing_lookup_facts(topic_id, region_id),
                        },
                    )
                }
            } else {
                (
                    row.try_get::<String, _>("priority")
                        .map_err(|error| format!("priority value: {error}"))?,
                    RuleProvenance {
                        source: rule_source,
                        version: Some(
                            row.try_get::<i32, _>("version")
                                .map_err(|error| format!("priority rule version: {error}"))?,
                        ),
                        reason: row
                            .try_get::<String, _>("reason")
                            .map_err(|error| format!("priority reason: {error}"))?,
                        facts_used: routing_lookup_facts(topic_id, region_id),
                    },
                )
            }
        } else if let Some(value) = requested_priority.filter(|value| !value.trim().is_empty()) {
            (
                normalize_priority(value),
                RuleProvenance {
                    source: input_source,
                    version: None,
                    reason: match input_source {
                        RuleSource::LabelHistory => {
                            "Приоритет из исторического значения обращения".to_owned()
                        }
                        RuleSource::Manual => "Приоритет введён вручную".to_owned(),
                        RuleSource::Official => "Официальное значение приоритета".to_owned(),
                    },
                    facts_used: routing_lookup_facts(topic_id, region_id),
                },
            )
        } else {
            (
                "medium".to_owned(),
                RuleProvenance::manual(
                    "Правило приоритета не предоставлено; используется ручной средний уровень",
                )
                .with_facts(routing_lookup_facts(topic_id, region_id)),
            )
        };
        let reason = format!(
            "{}; {}",
            service_provenance.reason, priority_provenance.reason
        );
        Ok(RoutingDecision {
            service_id,
            service_name,
            service_provenance,
            priority,
            priority_provenance,
            reason,
        })
    }

    pub async fn list_response_templates(&self) -> Result<ResponseTemplatesResponse, String> {
        let rows = sqlx::query_as::<_, DbResponseTemplate>(
            "SELECT id, template_key, language, topic_id, service_id, body, approved, version, created_by, updated_by, approved_by, approved_at, created_at, updated_at FROM response_templates ORDER BY template_key, language, version DESC, id DESC",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|error| format!("list response templates: {error}"))?;
        Ok(ResponseTemplatesResponse {
            items: rows.into_iter().map(Into::into).collect(),
        })
    }

    pub async fn get_response_template(
        &self,
        template_id: &str,
    ) -> Result<ResponseTemplateRecord, String> {
        let id = response_template_database_id(template_id)?;
        let row = sqlx::query_as::<_, DbResponseTemplate>(
            "SELECT id, template_key, language, topic_id, service_id, body, approved, version, created_by, updated_by, approved_by, approved_at, created_at, updated_at FROM response_templates WHERE id = $1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("get response template: {error}"))?
        .ok_or_else(|| format!("not found: response template {template_id} not found"))?;
        Ok(row.into())
    }

    pub async fn create_response_templates(
        &self,
        inputs: &[ResponseTemplateInput],
        actor_id: &str,
    ) -> Result<Vec<ResponseTemplateRecord>, String> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin response template import: {error}"))?;
        for input in inputs {
            let references_valid: bool = sqlx::query_scalar(
                "SELECT EXISTS (SELECT 1 FROM topics WHERE id = $1 AND active) AND EXISTS (SELECT 1 FROM services WHERE id = $2 AND active)",
            )
            .bind(&input.topic_id)
            .bind(&input.service_id)
            .fetch_one(&mut *tx)
            .await
            .map_err(|error| format!("validate response template references: {error}"))?;
            if !references_valid {
                return Err(format!(
                    "bad request: unknown or inactive topic_id/service_id for template {}",
                    input.template_key
                ));
            }
        }
        let lock_keys = inputs
            .iter()
            .map(|input| format!("{}:{}", input.template_key, input.language))
            .collect::<std::collections::BTreeSet<_>>();
        for lock_key in lock_keys {
            sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))")
                .bind(lock_key)
                .execute(&mut *tx)
                .await
                .map_err(|error| format!("lock response template version: {error}"))?;
        }

        let mut records = Vec::with_capacity(inputs.len());
        for input in inputs {
            let latest_version: Option<i32> = sqlx::query_scalar(
                "SELECT MAX(version) FROM response_templates WHERE template_key = $1 AND language = $2",
            )
            .bind(&input.template_key)
            .bind(&input.language)
            .fetch_one(&mut *tx)
            .await
            .map_err(|error| format!("read response template version: {error}"))?;
            let row = sqlx::query_as::<_, DbResponseTemplate>(
                "INSERT INTO response_templates (template_key, language, topic_id, service_id, body, approved, version, created_by, updated_by) VALUES ($1, $2, $3, $4, $5, FALSE, $6, $7, $7) RETURNING id, template_key, language, topic_id, service_id, body, approved, version, created_by, updated_by, approved_by, approved_at, created_at, updated_at",
            )
            .bind(&input.template_key)
            .bind(&input.language)
            .bind(&input.topic_id)
            .bind(&input.service_id)
            .bind(&input.body)
            .bind(latest_version.unwrap_or(0) + 1)
            .bind(actor_id)
            .fetch_one(&mut *tx)
            .await
            .map_err(|error| format!("create response template: {error}"))?;
            records.push(ResponseTemplateRecord::from(row));
        }
        tx.commit()
            .await
            .map_err(|error| format!("commit response template import: {error}"))?;
        Ok(records)
    }

    pub async fn update_response_template(
        &self,
        template_id: &str,
        body: &str,
        actor_id: &str,
    ) -> Result<ResponseTemplateRecord, String> {
        let id = response_template_database_id(template_id)?;
        let row = sqlx::query_as::<_, DbResponseTemplate>(
            "UPDATE response_templates SET body = $2, updated_by = $3, updated_at = now() WHERE id = $1 AND approved = FALSE RETURNING id, template_key, language, topic_id, service_id, body, approved, version, created_by, updated_by, approved_by, approved_at, created_at, updated_at",
        )
        .bind(id)
        .bind(body)
        .bind(actor_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("update response template: {error}"))?;
        if let Some(row) = row {
            return Ok(row.into());
        }
        let existing = self.get_response_template(template_id).await?;
        if existing.approved {
            return Err(
                "conflict: approved template versions are immutable; create a new version"
                    .to_owned(),
            );
        }
        Err(format!(
            "not found: response template {template_id} not found"
        ))
    }

    pub async fn delete_response_template(&self, template_id: &str) -> Result<(), String> {
        let id = response_template_database_id(template_id)?;
        let deleted: Option<i64> = sqlx::query_scalar(
            "DELETE FROM response_templates WHERE id = $1 AND approved = FALSE RETURNING id",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("delete response template: {error}"))?;
        if deleted.is_some() {
            return Ok(());
        }
        let existing = self.get_response_template(template_id).await?;
        if existing.approved {
            return Err("conflict: approved template versions cannot be deleted".to_owned());
        }
        Err(format!(
            "not found: response template {template_id} not found"
        ))
    }

    pub async fn approve_response_template(
        &self,
        template_id: &str,
        actor_id: &str,
    ) -> Result<ResponseTemplateRecord, String> {
        let id = response_template_database_id(template_id)?;
        let row = sqlx::query_as::<_, DbResponseTemplate>(
            "UPDATE response_templates SET approved = TRUE, approved_by = $2, approved_at = now(), updated_by = $2, updated_at = now() WHERE id = $1 AND approved = FALSE RETURNING id, template_key, language, topic_id, service_id, body, approved, version, created_by, updated_by, approved_by, approved_at, created_at, updated_at",
        )
        .bind(id)
        .bind(actor_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("approve response template: {error}"))?;
        if let Some(row) = row {
            return Ok(row.into());
        }
        let existing = self.get_response_template(template_id).await?;
        if existing.approved {
            return Err("conflict: template version is already approved".to_owned());
        }
        Err(format!(
            "not found: response template {template_id} not found"
        ))
    }

    async fn response_template(
        &self,
        language: &str,
        topic_id: &str,
        service_id: &str,
        region_name: &str,
    ) -> Result<ResponseTemplate, String> {
        let normalized = normalize_language(language);
        let row = sqlx::query(
            "SELECT rt.id, rt.template_key, rt.version, rt.body, rt.language, rt.topic_id, rt.service_id, CASE WHEN $1 = 'KZ' THEN COALESCE(tp.name_kk, tp.name_ru, tp.id) ELSE COALESCE(tp.name_ru, tp.name_kk, tp.id) END AS topic_label, CASE WHEN $1 = 'KZ' THEN COALESCE(s.name_kk, s.name_ru, s.id) ELSE COALESCE(s.name_ru, s.name_kk, s.id) END AS service_label FROM response_templates rt JOIN topics tp ON tp.id = rt.topic_id JOIN services s ON s.id = rt.service_id WHERE rt.approved = TRUE AND upper(rt.language) = $1 AND rt.topic_id = $2 AND rt.service_id = $3 ORDER BY rt.version DESC, rt.id DESC LIMIT 1",
        )
        .bind(&normalized)
        .bind(topic_id)
        .bind(service_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("fetch approved response template: {error}"))?;
        let Some(row) = row else {
            return Ok(response_template_for(language, topic_id));
        };
        let topic_label = row
            .try_get::<String, _>("topic_label")
            .map_err(|error| format!("template topic: {error}"))?;
        let service_label = row
            .try_get::<String, _>("service_label")
            .map_err(|error| format!("template service: {error}"))?;
        let stored_body = row
            .try_get::<String, _>("body")
            .map_err(|error| format!("template body: {error}"))?;
        let Some(body) =
            render_template_body(&stored_body, &topic_label, &service_label, region_name)
        else {
            return Ok(response_template_for(language, &topic_label));
        };
        Ok(ResponseTemplate {
            id: row
                .try_get::<i64, _>("id")
                .map_err(|error| format!("template id: {error}"))?
                .to_string(),
            title: format!("Ответ: {topic_label} · {service_label}"),
            body,
            language: row
                .try_get::<String, _>("language")
                .map_err(|error| format!("template language: {error}"))?
                .to_ascii_lowercase(),
            approved: true,
            source: "APPROVED_TEMPLATE".to_owned(),
            template_key: Some(
                row.try_get("template_key")
                    .map_err(|error| format!("template key: {error}"))?,
            ),
            topic_id: Some(
                row.try_get("topic_id")
                    .map_err(|error| format!("template topic id: {error}"))?,
            ),
            service_id: Some(
                row.try_get("service_id")
                    .map_err(|error| format!("template service id: {error}"))?,
            ),
            version: Some(
                row.try_get("version")
                    .map_err(|error| format!("template version: {error}"))?,
            ),
        })
    }

    pub async fn create_ticket(
        &self,
        text: &str,
        language: Option<&str>,
        region_id: Option<&str>,
        priority: Option<&str>,
        source: Option<&str>,
        request_id: &str,
    ) -> Result<TicketDetailResponse, String> {
        let classification = self
            .classify(text, language, request_id, request_id)
            .await?;
        let db_topic = normalize_topic_id(&classification.prediction.topic_id);
        let db_region = normalize_region_id(region_id.unwrap_or("KZ-ASTANA"));
        let db_language =
            normalize_language(language.unwrap_or(&classification.prediction.language));
        let routing = self
            .resolve_routing(&db_topic, &db_region, None, priority, RuleSource::Manual)
            .await?;
        let db_priority = routing.priority.clone();
        let external_id = format!(
            "api-{}",
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        );
        let now = Utc::now();
        let shadow_window = self.shadow_window_for_ticket(now).await?;
        let (candidate_shadow_prediction, candidate_shadow_error) = if let Some(window) =
            shadow_window.as_ref()
        {
            if window.production_model_version.as_deref() != Some(&classification.model_version) {
                (None, Some("PRODUCTION_MODEL_VERSION_MISMATCH"))
            } else if window.candidate_model_version == classification.model_version {
                (None, Some("CANDIDATE_MODEL_VERSION_CONFLICT"))
            } else if let Some(checksum) = window.artifact_checksum.as_deref() {
                match self
                    .classify_versioned(
                        text,
                        language,
                        request_id,
                        request_id,
                        Some(&window.candidate_model_version),
                        Some(checksum),
                    )
                    .await
                {
                    Ok(candidate) if candidate.model_version == window.candidate_model_version => {
                        (Some(candidate), None)
                    }
                    Ok(_) => (None, Some("CANDIDATE_MODEL_VERSION_MISMATCH")),
                    Err(_) => (None, Some("CANDIDATE_INFERENCE_FAILED")),
                }
            } else {
                (None, Some("CANDIDATE_ARTIFACT_CHECKSUM_UNAVAILABLE"))
            }
        } else {
            (None, None)
        };
        let mut index_tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin vector index update: {error}"))?;
        let active_index = self.locked_vector_index(&mut index_tx).await?;
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin ticket transaction: {error}"))?;
        let ticket_id: i64 = sqlx::query_scalar(
            r#"
            INSERT INTO tickets (
                external_ticket_id, source_system, region_id, created_at,
                original_text, language, topic_raw, topic_id, service_raw,
                service_id, priority, status, pulse_prediction, model_versions
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, 'OPEN', $12, $13)
            RETURNING id
            "#,
        )
        .bind(&external_id)
        .bind(source.unwrap_or("api"))
        .bind(&db_region)
        .bind(now)
        .bind(text)
        .bind(&db_language)
        .bind(&classification.prediction.topic_id)
        .bind(&db_topic)
        .bind(&classification.prediction.topic)
        .bind(&routing.service_id)
        .bind(&db_priority)
        .bind(json!({
            "topic_id": db_topic,
            "confidence": classification.prediction.confidence,
            "model_version": classification.model_version,
            "routing_reason": routing.reason,
            "service_provenance": routing.service_provenance,
            "priority_provenance": routing.priority_provenance,
        }))
        .bind(json!({"classifier": classification.model_version}))
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("insert ticket: {error}"))?;

        let prediction = prediction_for_db(ticket_id, &classification, &routing);
        sqlx::query(
            "INSERT INTO ticket_predictions (ticket_id, model_version, topic_id, service_id, priority, confidence, alternatives, prediction, needs_review) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
        )
        .bind(ticket_id)
        .bind(&classification.model_version)
        .bind(&db_topic)
        .bind(&routing.service_id)
        .bind(&db_priority)
        .bind(classification.prediction.confidence as f64)
        .bind(alternatives_json(&classification.prediction))
        .bind(json!({
            "topic_id": db_topic,
            "topic": classification.prediction.topic,
            "confidence": classification.prediction.confidence,
            "confidence_state": classification.prediction.confidence_state,
            "needs_review": classification.prediction.needs_review,
            "routing_reason": &routing.reason,
            "service_provenance": &routing.service_provenance,
            "priority_provenance": &routing.priority_provenance,
        }))
        .bind(classification.prediction.needs_review)
        .execute(&mut *tx)
        .await
        .map_err(|error| format!("insert prediction: {error}"))?;

        if let Some(window) = shadow_window.as_ref() {
            let candidate_prediction_value =
                candidate_shadow_prediction.as_ref().map(|candidate| {
                    json!({
                        "model_version": candidate.model_version,
                        "language": candidate.prediction.language,
                        "topic_id": candidate.prediction.topic_id,
                        "topic": candidate.prediction.topic,
                        "confidence": candidate.prediction.confidence,
                        "confidence_state": candidate.prediction.confidence_state,
                        "needs_review": candidate.prediction.needs_review,
                        "alternatives": candidate.prediction.alternatives,
                    })
                });
            sqlx::query(
                "INSERT INTO learning_cycle_shadow_predictions (learning_cycle_id, ticket_id, predicted_at, production_model_version, candidate_model_version, production_prediction, candidate_prediction, candidate_inference_status, candidate_error_code) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) ON CONFLICT (learning_cycle_id, ticket_id) DO NOTHING",
            )
            .bind(window.learning_cycle_id)
            .bind(ticket_id)
            .bind(now)
            .bind(&classification.model_version)
            .bind(&window.candidate_model_version)
            .bind(json!(&prediction))
            .bind(candidate_prediction_value)
            .bind(if candidate_shadow_prediction.is_some() {
                "COMPLETED"
            } else {
                "FAILED"
            })
            .bind(candidate_shadow_error)
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("persist candidate shadow prediction: {error}"))?;
        }
        tx.commit()
            .await
            .map_err(|error| format!("commit ticket transaction: {error}"))?;

        let vector = self
            .embed(&active_index, text, request_id, request_id)
            .await?;
        self.qdrant_upsert(
            &active_index,
            ticket_id,
            &vector,
            &db_topic,
            &db_region,
            &now.to_rfc3339(),
        )
        .await?;
        sqlx::query("UPDATE tickets SET embedding_ref = $2, model_versions = model_versions || $3::jsonb, updated_in_pulse_at = now() WHERE id = $1")
            .bind(ticket_id)
            .bind(format!("qdrant:{}:{ticket_id}", active_index.collection_name))
            .bind(json!({"embedder": active_index.embedder_version}))
            .execute(&mut *index_tx)
            .await
            .map_err(|error| format!("store embedding reference: {error}"))?;
        index_tx
            .commit()
            .await
            .map_err(|error| format!("commit vector index update: {error}"))?;

        let ticket = self.fetch_ticket_by_id(ticket_id).await?;
        Ok(TicketDetailResponse {
            ticket,
            prediction,
            latest_decision: None,
        })
    }

    pub async fn dataset_provenance(&self) -> Result<DatasetProvenance, String> {
        let row = sqlx::query(
            "WITH ticket_provenance AS (SELECT dtl.ticket_id, BOOL_OR(dv.is_synthetic) AS has_synthetic, BOOL_OR(NOT dv.is_synthetic) AS has_real FROM dataset_ticket_links dtl JOIN dataset_versions dv USING (dataset_version) GROUP BY dtl.ticket_id) SELECT COUNT(*) FILTER (WHERE has_synthetic)::bigint AS synthetic_ticket_count, COUNT(*) FILTER (WHERE has_real)::bigint AS real_ticket_count, (SELECT COUNT(*) FROM tickets t LEFT JOIN ticket_provenance p ON p.ticket_id = t.id WHERE p.ticket_id IS NULL)::bigint AS unassigned_ticket_count, COALESCE((SELECT SUM(quarantine_record_count) FROM dataset_versions), 0)::bigint AS quarantined_row_count, (SELECT COUNT(*) FROM dataset_versions)::bigint AS dataset_version_count FROM ticket_provenance",
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|error| format!("read dataset provenance: {error}"))?;
        Ok(DatasetProvenance {
            synthetic_ticket_count: row
                .try_get("synthetic_ticket_count")
                .map_err(|error| format!("read synthetic dataset count: {error}"))?,
            real_ticket_count: row
                .try_get("real_ticket_count")
                .map_err(|error| format!("read real dataset count: {error}"))?,
            unassigned_ticket_count: row
                .try_get("unassigned_ticket_count")
                .map_err(|error| format!("read unassigned ticket count: {error}"))?,
            quarantined_row_count: row
                .try_get("quarantined_row_count")
                .map_err(|error| format!("read quarantined row count: {error}"))?,
            dataset_version_count: row
                .try_get("dataset_version_count")
                .map_err(|error| format!("read dataset version count: {error}"))?,
        })
    }

    pub async fn import_tickets(
        &self,
        request: &ImportRequest,
        request_id: &str,
    ) -> Result<ImportResponse, ImportError> {
        if request.source_system.trim().is_empty() {
            return Err(ImportError::Invalid("source_system is required".to_owned()));
        }
        let known_regions = sqlx::query_scalar::<_, String>("SELECT id FROM regions WHERE active")
            .fetch_all(&self.pool)
            .await
            .map_err(|error| format!("read import regions: {error}"))?;
        for (row_index, item) in request.tickets.iter().enumerate() {
            let row_number = row_index + 1;
            if !item.is_object() {
                return Err(ImportError::Invalid(format!(
                    "row {row_number}: UNKNOWN_SCHEMA"
                )));
            }
            if json_text(item, "external_ticket_id").is_none() {
                return Err(import_row_error(
                    row_number,
                    "MISSING_REQUIRED_FIELD",
                    "external_ticket_id",
                ));
            }
            if json_text(item, "original_text").is_none() {
                return Err(import_row_error(
                    row_number,
                    "MISSING_REQUIRED_FIELD",
                    "original_text",
                ));
            }
            let region_id = json_text(item, "region_id").ok_or_else(|| {
                import_row_error(row_number, "MISSING_REQUIRED_FIELD", "region_id")
            })?;
            if !known_regions.contains(&normalize_region_id(&region_id)) {
                return Err(import_row_error(row_number, "INVALID_VALUE", "region_id"));
            }
            json_datetime(item, "created_at")
                .map_err(|_| import_row_error(row_number, "INVALID_DATE", "created_at"))?;
            for field in ["closed_at", "deadline_at"] {
                json_datetime_optional(item, field)
                    .map_err(|_| import_row_error(row_number, "INVALID_DATE", field))?;
            }
        }
        for (row_index, item) in request.quarantine.iter().enumerate() {
            let row_number = row_index + 1;
            let reason = json_text(item, "reason").unwrap_or_else(|| "UNKNOWN_SCHEMA".to_owned());
            if !matches!(
                reason.as_str(),
                "BAD_CSV_STRUCTURE"
                    | "INVALID_DATE"
                    | "MISSING_REQUIRED_FIELD"
                    | "UNKNOWN_SCHEMA"
                    | "PII_REVIEW"
                    | "INVALID_VALUE"
            ) {
                return Err(import_row_error(row_number, "INVALID_VALUE", "reason"));
            }
            if item
                .get("row_number")
                .and_then(Value::as_i64)
                .is_some_and(|number| !(1..=i32::MAX as i64).contains(&number))
            {
                return Err(import_row_error(row_number, "INVALID_VALUE", "row_number"));
            }
        }
        let mut index_tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin import vector index update: {error}"))?;
        let active_index = self.locked_vector_index(&mut index_tx).await?;
        let source_system = request.source_system.trim().to_owned();
        let dataset_version = request.dataset_version.clone().unwrap_or_else(|| {
            format!("{}-import-{}", source_system, Utc::now().timestamp_millis())
        });
        let manifest_uri = request
            .manifest_uri
            .clone()
            .or_else(|| request.source_uri.clone())
            .unwrap_or_else(|| format!("runtime://{source_system}/{dataset_version}"));
        let manifest_sha256 = request
            .manifest_sha256
            .clone()
            .unwrap_or_else(|| "runtime-unhashed".to_owned());
        let content = serde_json::to_vec(&json!({
            "source_system": source_system,
            "tickets": request.tickets,
            "quarantine": request.quarantine,
        }))
        .map_err(|error| format!("serialize dataset content: {error}"))?;
        let content_sha256 = format!("sha256:{:x}", Sha256::digest(&content));
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin import transaction: {error}"))?;
        sqlx::query(
            "INSERT INTO dataset_versions (dataset_version, schema_version, manifest_uri, manifest_sha256, content_sha256, is_synthetic, record_count, quarantine_record_count) VALUES ($1, 'unified-ticket.v1', $2, $3, $4, $5, $6, $7) ON CONFLICT (dataset_version) DO NOTHING",
        )
        .bind(&dataset_version)
        .bind(&manifest_uri)
        .bind(&manifest_sha256)
        .bind(&content_sha256)
        .bind(request.is_synthetic)
        .bind(request.tickets.len() as i32)
        .bind(request.quarantine.len() as i32)
        .execute(&mut *tx)
        .await
        .map_err(|error| format!("register dataset version: {error}"))?;
        let registered = sqlx::query(
            "SELECT schema_version, manifest_uri, manifest_sha256, content_sha256, is_synthetic, record_count, quarantine_record_count FROM dataset_versions WHERE dataset_version = $1 FOR UPDATE",
        )
        .bind(&dataset_version)
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("read dataset version: {error}"))?;
        let registered_content = registered
            .try_get::<Option<String>, _>("content_sha256")
            .map_err(|error| format!("read dataset checksum: {error}"))?;
        let registered_schema: String = registered
            .try_get("schema_version")
            .map_err(|error| format!("read dataset schema: {error}"))?;
        let registered_uri: String = registered
            .try_get("manifest_uri")
            .map_err(|error| format!("read dataset manifest URI: {error}"))?;
        let registered_manifest_sha256: String = registered
            .try_get("manifest_sha256")
            .map_err(|error| format!("read dataset manifest checksum: {error}"))?;
        let registered_synthetic: bool = registered
            .try_get("is_synthetic")
            .map_err(|error| format!("read dataset type: {error}"))?;
        let registered_count: i32 = registered
            .try_get("record_count")
            .map_err(|error| format!("read dataset record count: {error}"))?;
        let registered_quarantine_count: i32 = registered
            .try_get("quarantine_record_count")
            .map_err(|error| format!("read dataset quarantine count: {error}"))?;
        if registered_schema != "unified-ticket.v1"
            || registered_uri != manifest_uri
            || registered_manifest_sha256 != manifest_sha256
            || registered_synthetic != request.is_synthetic
            || registered_count != request.tickets.len() as i32
            || registered_quarantine_count != request.quarantine.len() as i32
            || registered_content
                .as_deref()
                .is_some_and(|value| value != content_sha256.as_str())
        {
            return Err(ImportError::Conflict(
                "dataset_version already has different content or manifest".to_owned(),
            ));
        }
        if registered_content.is_none() {
            sqlx::query(
                "UPDATE dataset_versions SET content_sha256 = $2 WHERE dataset_version = $1",
            )
            .bind(&dataset_version)
            .bind(&content_sha256)
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("backfill dataset checksum: {error}"))?;
        }
        let import_run_id: i64 = sqlx::query_scalar(
            "INSERT INTO data_import_runs (source_system, source_uri, dataset_version, status, total_rows, valid_rows, quarantined_rows) VALUES ($1, $2, $3, 'RUNNING', $4, 0, $5) RETURNING id",
        )
        .bind(&source_system)
        .bind(request.source_uri.as_deref())
        .bind(&dataset_version)
        .bind((request.tickets.len() + request.quarantine.len()) as i32)
        .bind(request.quarantine.len() as i32)
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("create import run: {error}"))?;
        let mut imported_rows = 0usize;
        let mut duplicate_rows = 0usize;
        let mut index_queue: Vec<(i64, String, String, String, String)> = Vec::new();
        for (row_index, item) in request.tickets.iter().enumerate() {
            let row_number = row_index + 1;
            let external_id = json_text(item, "external_ticket_id").ok_or_else(|| {
                import_row_error(row_number, "MISSING_REQUIRED_FIELD", "external_ticket_id")
            })?;
            let text = json_text(item, "original_text").ok_or_else(|| {
                import_row_error(row_number, "MISSING_REQUIRED_FIELD", "original_text")
            })?;
            let region_id =
                normalize_region_id(&json_text(item, "region_id").ok_or_else(|| {
                    import_row_error(row_number, "MISSING_REQUIRED_FIELD", "region_id")
                })?);
            let language =
                normalize_language(json_text(item, "language").as_deref().unwrap_or("UNKNOWN"));
            let classification = self
                .classify(&text, Some(&language), request_id, request_id)
                .await?;
            let topic_id = normalize_topic_id(
                json_text(item, "topic_id")
                    .filter(|value| value != "unknown" && !value.is_empty())
                    .as_deref()
                    .unwrap_or(&classification.prediction.topic_id),
            );
            let routing = self
                .resolve_routing(
                    &topic_id,
                    &region_id,
                    json_text(item, "service_raw").as_deref(),
                    json_text(item, "priority").as_deref(),
                    RuleSource::LabelHistory,
                )
                .await?;
            let created_at = json_datetime(item, "created_at")
                .map_err(|_| import_row_error(row_number, "INVALID_DATE", "created_at"))?;
            let status =
                normalize_status(json_text(item, "status").as_deref().unwrap_or("UNKNOWN"));
            let inserted_id: Option<i64> = sqlx::query_scalar(
                r#"
                INSERT INTO tickets (
                    external_ticket_id, source_system, region_id, created_at,
                    original_text, language, topic_raw, topic_id, service_raw,
                    service_id, priority, status, district, address, object,
                    channel, closed_at, deadline_at, resolution_text,
                    official_response, pulse_prediction, model_versions,
                    needs_review, text_redaction_count
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                    $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24)
                ON CONFLICT (source_system, external_ticket_id) DO NOTHING
                RETURNING id
                "#,
            )
            .bind(&external_id)
            .bind(&source_system)
            .bind(&region_id)
            .bind(created_at)
            .bind(&text)
            .bind(&language)
            .bind(json_text(item, "topic_raw").unwrap_or_else(|| topic_id.clone()))
            .bind(&topic_id)
            .bind(json_text(item, "service_raw"))
            .bind(&routing.service_id)
            .bind(&routing.priority)
            .bind(&status)
            .bind(json_text(item, "district"))
            .bind(json_text(item, "address"))
            .bind(json_text(item, "object"))
            .bind(json_text(item, "channel"))
            .bind(
                json_datetime_optional(item, "closed_at")
                    .map_err(|_| import_row_error(row_number, "INVALID_DATE", "closed_at"))?,
            )
            .bind(
                json_datetime_optional(item, "deadline_at")
                    .map_err(|_| import_row_error(row_number, "INVALID_DATE", "deadline_at"))?,
            )
            .bind(json_text(item, "resolution_text"))
            .bind(json_text(item, "official_response"))
            .bind(json!({
                "topic_id": topic_id,
                "confidence": classification.prediction.confidence,
                "model_version": classification.model_version,
                "routing_reason": routing.reason,
                "service_provenance": routing.service_provenance,
                "priority_provenance": routing.priority_provenance,
            }))
            .bind(json!({
                "classifier": classification.model_version,
                "dataset_version": dataset_version,
            }))
            .bind(classification.prediction.needs_review)
            .bind(
                item.get("text_redaction_count")
                    .and_then(Value::as_i64)
                    .unwrap_or(0) as i32,
            )
            .fetch_optional(&mut *tx)
            .await
            .map_err(|error| format!("insert imported ticket {external_id}: {error}"))?;
            let is_new_ticket = inserted_id.is_some();
            let ticket_id = if let Some(id) = inserted_id {
                imported_rows += 1;
                sqlx::query(
                    "INSERT INTO ticket_predictions (ticket_id, model_version, topic_id, service_id, priority, confidence, alternatives, prediction, needs_review) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
                )
                .bind(id)
                .bind(&classification.model_version)
                .bind(&topic_id)
                .bind(&routing.service_id)
                .bind(&routing.priority)
                .bind(classification.prediction.confidence as f64)
                .bind(alternatives_json(&classification.prediction))
                .bind(json!({
                    "topic_id": topic_id,
                    "topic": classification.prediction.topic,
                    "confidence": classification.prediction.confidence,
                    "confidence_state": classification.prediction.confidence_state,
                    "routing_reason": &routing.reason,
                    "service_provenance": &routing.service_provenance,
                    "priority_provenance": &routing.priority_provenance,
                }))
                .bind(classification.prediction.needs_review)
                .execute(&mut *tx)
                .await
                .map_err(|error| format!("insert imported prediction {external_id}: {error}"))?;
                id
            } else {
                let existing = sqlx::query(
                    "SELECT id, original_text, region_id, created_at, language, status FROM tickets WHERE source_system = $1 AND external_ticket_id = $2 FOR UPDATE",
                )
                .bind(&source_system)
                .bind(&external_id)
                .fetch_one(&mut *tx)
                .await
                .map_err(|error| format!("find duplicate imported ticket {external_id}: {error}"))?;
                if existing.get::<String, _>("original_text") != text
                    || existing.get::<String, _>("region_id") != region_id
                    || existing.get::<DateTime<Utc>, _>("created_at") != created_at
                    || existing.get::<String, _>("language") != language
                    || existing.get::<String, _>("status") != status
                {
                    return Err(ImportError::Conflict(
                        "source ticket already exists with different canonical fields".to_owned(),
                    ));
                }
                duplicate_rows += 1;
                existing.get::<i64, _>("id")
            };
            sqlx::query(
                "INSERT INTO dataset_ticket_links (dataset_version, ticket_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
            )
            .bind(&dataset_version)
            .bind(ticket_id)
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("link imported ticket to dataset: {error}"))?;
            if is_new_ticket {
                index_queue.push((
                    ticket_id,
                    region_id,
                    topic_id,
                    text,
                    created_at.to_rfc3339(),
                ));
            }
        }
        for (row_number, item) in request.quarantine.iter().enumerate() {
            let reason = json_text(item, "reason").unwrap_or_else(|| "UNKNOWN_SCHEMA".to_owned());
            let field = json_text(item, "field").filter(|value| {
                matches!(
                    value.as_str(),
                    "external_ticket_id"
                        | "source_system"
                        | "region_id"
                        | "created_at"
                        | "original_text"
                        | "language"
                        | "topic_raw"
                        | "topic_id"
                        | "service_raw"
                        | "service_id"
                        | "priority"
                        | "status"
                        | "district"
                        | "address"
                        | "channel"
                        | "closed_at"
                        | "deadline_at"
                )
            });
            sqlx::query(
                "INSERT INTO quarantine_rows (import_run_id, source_system, row_number, reason, field, detail, row_snapshot) VALUES ($1, $2, $3, $4, $5, $6, $7)",
            )
            .bind(import_run_id)
            .bind(&source_system)
            .bind(item.get("row_number").and_then(Value::as_i64).unwrap_or((row_number + 1) as i64) as i32)
            .bind(&reason)
            .bind(field)
            .bind(&reason)
            .bind(json!({"content_redacted": true}))
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("insert quarantine row: {error}"))?;
        }
        sqlx::query(
            "UPDATE data_import_runs SET status = 'COMPLETED', valid_rows = $2, quarantined_rows = $3, finished_at = now() WHERE id = $1",
        )
        .bind(import_run_id)
        .bind(imported_rows as i32)
        .bind(request.quarantine.len() as i32)
        .execute(&mut *tx)
        .await
        .map_err(|error| format!("complete import run: {error}"))?;
        tx.commit()
            .await
            .map_err(|error| format!("commit import: {error}"))?;

        let mut indexed_rows = 0usize;
        for (ticket_id, region_id, topic_id, text, created_at) in index_queue {
            let vector = self
                .embed(&active_index, &text, request_id, request_id)
                .await?;
            self.qdrant_upsert(
                &active_index,
                ticket_id,
                &vector,
                &topic_id,
                &region_id,
                &created_at,
            )
            .await?;
            sqlx::query("UPDATE tickets SET embedding_ref = $2, model_versions = model_versions || $3::jsonb, updated_in_pulse_at = now() WHERE id = $1")
                .bind(ticket_id)
                .bind(format!("qdrant:{}:{ticket_id}", active_index.collection_name))
                .bind(json!({"embedder": active_index.embedder_version}))
                .execute(&mut *index_tx)
                .await
                .map_err(|error| format!("store imported embedding reference: {error}"))?;
            indexed_rows += 1;
        }
        index_tx
            .commit()
            .await
            .map_err(|error| format!("commit imported vector index update: {error}"))?;
        Ok(ImportResponse {
            import_run_id: import_run_id.to_string(),
            source_system,
            dataset_version,
            total_rows: request.tickets.len() + request.quarantine.len(),
            imported_rows,
            duplicate_rows,
            quarantined_rows: request.quarantine.len(),
            indexed_rows,
            is_synthetic: request.is_synthetic,
            source: "postgres+ml+qdrant".to_owned(),
        })
    }

    pub async fn queue_reindex(&self) -> Result<Value, String> {
        let payload = json!({
            "kind": "reindex_qdrant",
            "collection_base": self.qdrant_collection,
            "embedding_dimension": self.embedding_dimension,
            "distance_metric": self.embedding_distance,
            "embedder_version": self.embedder_version,
        });
        let job_id: i64 = sqlx::query_scalar(
            "INSERT INTO background_jobs (job_type, payload, state) VALUES ('REINDEX_QDRANT', $1, 'QUEUED') RETURNING id",
        )
        .bind(payload)
        .fetch_one(&self.pool)
        .await
        .map_err(|error| format!("queue Qdrant reindex: {error}"))?;
        Ok(json!({
            "job_id": job_id.to_string(),
            "job_type": "REINDEX_QDRANT",
            "state": "QUEUED",
            "collection": self.qdrant_collection,
            "target_collection_base": self.qdrant_collection,
            "embedding_dimension": self.embedding_dimension,
            "distance_metric": self.embedding_distance,
            "embedder_version": self.embedder_version,
            "source": "postgres+qdrant+worker",
        }))
    }

    pub async fn delete_vector(&self, ticket_id: &str) -> Result<Value, String> {
        let ticket = self.fetch_ticket(ticket_id).await?;
        let numeric_id = ticket
            .id
            .parse::<i64>()
            .map_err(|_| "stored ticket has invalid database id".to_owned())?;
        let mut index_tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin vector deletion: {error}"))?;
        let active_index = self.locked_vector_index(&mut index_tx).await?;
        self.qdrant_delete(&active_index, numeric_id).await?;
        sqlx::query(
            "UPDATE tickets SET embedding_ref = NULL, updated_in_pulse_at = now() WHERE id = $1",
        )
        .bind(numeric_id)
        .execute(&mut *index_tx)
        .await
        .map_err(|error| format!("clear embedding reference: {error}"))?;
        index_tx
            .commit()
            .await
            .map_err(|error| format!("commit vector deletion: {error}"))?;
        Ok(json!({
            "ticket_id": ticket.id,
            "collection": active_index.collection_name,
            "state": "DELETED",
            "source": "postgres+qdrant",
        }))
    }

    pub async fn get_ticket(&self, ticket_id: &str) -> Result<TicketDetailResponse, String> {
        let ticket = self.fetch_ticket(ticket_id).await?;
        let numeric_id = ticket
            .id
            .parse::<i64>()
            .map_err(|_| "stored ticket has invalid database id".to_owned())?;
        let prediction = self.fetch_prediction(numeric_id).await?;
        let decision = self.fetch_latest_decision(numeric_id).await?;
        Ok(TicketDetailResponse {
            ticket,
            prediction,
            latest_decision: decision,
        })
    }

    pub async fn context_handoff_package(
        &self,
        ticket_id: &str,
    ) -> Result<ContextHandoffPackage, String> {
        let ticket = self.fetch_ticket(ticket_id).await?;
        let numeric_id = ticket
            .id
            .parse::<i64>()
            .map_err(|_| "stored ticket has invalid database id".to_owned())?;
        let decision = self
            .fetch_latest_decision(numeric_id)
            .await?
            .ok_or_else(|| {
                "an operator decision is required before preparing a handoff".to_owned()
            })?;
        let location: DbContextHandoffLocation = sqlx::query_as(
            "SELECT district, address, object, CASE WHEN jsonb_typeof(attachments) = 'array' THEN jsonb_array_length(attachments) ELSE 0 END::bigint AS attachment_count FROM tickets WHERE id = $1",
        )
        .bind(numeric_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|error| format!("fetch context handoff location: {error}"))?;
        let attachment_count = usize::try_from(location.attachment_count)
            .map_err(|_| "stored ticket has invalid attachment count".to_owned())?;

        build_context_handoff_package(
            &ticket,
            &decision,
            ContextHandoffLocationSource {
                district: location.district,
                address: location.address,
                object: location.object,
                attachment_count,
            },
        )
    }

    pub async fn get_prediction(&self, ticket_id: &str) -> Result<Prediction, String> {
        let ticket = self.fetch_ticket(ticket_id).await?;
        let numeric_id = ticket
            .id
            .parse::<i64>()
            .map_err(|_| "stored ticket has invalid database id".to_owned())?;
        self.fetch_prediction(numeric_id).await
    }

    pub async fn list_tickets(&self, query: &TicketQuery) -> Result<TicketListResponse, String> {
        let mut builder = QueryBuilder::<Postgres>::new(TICKET_SELECT);
        builder.push(" WHERE TRUE");
        let mut count = QueryBuilder::<Postgres>::new("SELECT COUNT(*) FROM tickets t WHERE TRUE");
        if let Some(region_id) = &query.region_id {
            let region = normalize_region_id(region_id);
            builder
                .push(" AND t.region_id = ")
                .push_bind(region.clone());
            count.push(" AND t.region_id = ").push_bind(region);
        }
        if let Some(topic_id) = &query.topic_id {
            let topic = normalize_topic_id(topic_id);
            builder.push(" AND t.topic_id = ").push_bind(topic.clone());
            count.push(" AND t.topic_id = ").push_bind(topic);
        }
        if let Some(language) = &query.language {
            let language = normalize_language(language);
            builder
                .push(" AND t.language = ")
                .push_bind(language.clone());
            count.push(" AND t.language = ").push_bind(language);
        }
        if let Some(status) = &query.status {
            builder
                .push(" AND t.status = ")
                .push_bind(status.to_ascii_uppercase());
            count
                .push(" AND t.status = ")
                .push_bind(status.to_ascii_uppercase());
        }
        if let Some(search) = &query.q {
            let pattern = format!("%{}%", search.replace('%', "\\%"));
            builder
                .push(" AND t.original_text ILIKE ")
                .push_bind(pattern.clone());
            count.push(" AND t.original_text ILIKE ").push_bind(pattern);
        }
        let total: i64 = count
            .build_query_scalar()
            .fetch_one(&self.pool)
            .await
            .map_err(|error| format!("count tickets: {error}"))?;
        let limit = query.limit.unwrap_or(50).clamp(1, 100);
        let offset = query.offset.unwrap_or(0);
        builder
            .push(" ORDER BY t.created_at DESC, t.id DESC LIMIT ")
            .push_bind(limit as i64)
            .push(" OFFSET ")
            .push_bind(offset as i64);
        let rows: Vec<DbTicket> = builder
            .build_query_as()
            .fetch_all(&self.pool)
            .await
            .map_err(|error| format!("list tickets: {error}"))?;
        Ok(TicketListResponse {
            items: rows.into_iter().map(ticket_from_db).collect(),
            total: total.max(0) as usize,
            limit,
            offset,
        })
    }

    pub async fn analytics(&self, query: &AnalyticsQuery) -> Result<AnalyticsResponse, String> {
        let days = analytics_days(query);
        let now = Utc::now();
        let current_since = now - chrono::Duration::days(days);
        let previous_since = current_since - chrono::Duration::days(days);

        let mut overview_query =
            QueryBuilder::<Postgres>::new("SELECT COUNT(*) FILTER (WHERE t.created_at >= ");
        overview_query
            .push_bind(current_since)
            .push(")::bigint AS total_tickets, COUNT(*) FILTER (WHERE t.created_at >= ")
            .push_bind(current_since)
            .push(" AND t.status IN ('OPEN', 'IN_PROGRESS', 'TRIAGED'))::bigint AS open_tickets, COUNT(*) FILTER (WHERE t.created_at >= ")
            .push_bind(current_since)
            .push(" AND t.status IN ('RESOLVED', 'CLOSED'))::bigint AS resolved_tickets, COUNT(*) FILTER (WHERE t.created_at >= ")
            .push_bind(current_since)
            .push(" AND lower(COALESCE(t.priority, '')) IN ('high', 'critical'))::bigint AS high_priority_tickets, COUNT(*) FILTER (WHERE t.created_at >= ")
            .push_bind(current_since)
            .push(" AND t.operator_confirmed_decision IS NOT NULL)::bigint AS operator_decisions, COUNT(*) FILTER (WHERE t.created_at >= ")
            .push_bind(current_since)
            .push(" AND d.decision = 'CONFIRMED')::bigint AS confirmed_decisions, COUNT(*) FILTER (WHERE t.created_at >= ")
            .push_bind(current_since)
            .push(" AND d.decision = 'CORRECTED')::bigint AS corrected_decisions, COUNT(*) FILTER (WHERE t.created_at >= ")
            .push_bind(previous_since)
            .push(" AND t.created_at < ")
            .push_bind(current_since)
            .push(")::bigint AS previous_total_tickets, AVG(EXTRACT(EPOCH FROM (first_d.created_at - t.created_at)) / 60.0) FILTER (WHERE t.created_at >= ")
            .push_bind(current_since)
            .push(" AND first_d.created_at >= t.created_at)::float8 AS avg_decision_minutes, COUNT(*) FILTER (WHERE t.created_at >= ")
            .push_bind(current_since)
            .push(" AND first_d.created_at >= t.created_at)::bigint AS decision_time_samples, COALESCE(AVG(p.confidence) FILTER (WHERE t.created_at >= ")
            .push_bind(current_since)
            .push("), 0)::float8 AS average_confidence FROM tickets t LEFT JOIN LATERAL (SELECT confidence FROM ticket_predictions WHERE ticket_id = t.id ORDER BY created_at DESC LIMIT 1) p ON TRUE LEFT JOIN LATERAL (SELECT decision, created_at FROM operator_decisions WHERE ticket_id = t.id ORDER BY created_at DESC LIMIT 1) d ON TRUE LEFT JOIN LATERAL (SELECT created_at FROM operator_decisions WHERE ticket_id = t.id ORDER BY created_at ASC, id ASC LIMIT 1) first_d ON TRUE WHERE t.created_at >= ")
            .push_bind(previous_since)
            .push(" AND t.created_at <= now()");
        push_analytics_filters(&mut overview_query, query, "t");
        let overview_row = overview_query
            .build()
            .fetch_one(&self.pool)
            .await
            .map_err(|error| format!("analytics overview: {error}"))?;
        let total_tickets: i64 = overview_row.try_get("total_tickets").unwrap_or(0);
        let open_tickets: i64 = overview_row.try_get("open_tickets").unwrap_or(0);
        let resolved_tickets: i64 = overview_row.try_get("resolved_tickets").unwrap_or(0);
        let high_priority_tickets: i64 = overview_row.try_get("high_priority_tickets").unwrap_or(0);
        let operator_decisions: i64 = overview_row.try_get("operator_decisions").unwrap_or(0);
        let confirmed_decisions: i64 = overview_row.try_get("confirmed_decisions").unwrap_or(0);
        let corrected_decisions: i64 = overview_row.try_get("corrected_decisions").unwrap_or(0);
        let avg_decision_minutes: Option<f64> =
            overview_row.try_get("avg_decision_minutes").unwrap_or(None);
        let decision_time_samples: i64 = overview_row.try_get("decision_time_samples").unwrap_or(0);
        let previous_total: i64 = overview_row.try_get("previous_total_tickets").unwrap_or(0);
        let average_confidence: f64 = overview_row.try_get("average_confidence").unwrap_or(0.0);
        let total_change = total_tickets - previous_total;
        let total_change_pct = percent_change(total_tickets, previous_total);

        let mut decision_metrics_query = QueryBuilder::<Postgres>::new(
            "SELECT \
                COUNT(*) FILTER (WHERE NULLIF(BTRIM(d.feedback->>'predicted_topic_id'), '') IS NOT NULL AND lower(BTRIM(d.feedback->>'predicted_topic_id')) NOT IN ('unknown', 'unavailable'))::bigint AS classification_decisions, \
                COUNT(*) FILTER (WHERE NULLIF(BTRIM(d.feedback->>'predicted_topic_id'), '') IS NOT NULL AND lower(BTRIM(d.feedback->>'predicted_topic_id')) NOT IN ('unknown', 'unavailable') AND lower(BTRIM(d.feedback->>'predicted_topic_id')) IS DISTINCT FROM lower(BTRIM(d.confirmed_topic_id)))::bigint AS classification_corrections, \
                COUNT(*) FILTER (WHERE NULLIF(BTRIM(d.feedback->>'predicted_service'), '') IS NOT NULL AND lower(BTRIM(d.feedback->>'predicted_service')) NOT IN ('unknown', 'unavailable'))::bigint AS routing_decisions, \
                COUNT(*) FILTER (WHERE NULLIF(BTRIM(d.feedback->>'predicted_service'), '') IS NOT NULL AND lower(BTRIM(d.feedback->>'predicted_service')) NOT IN ('unknown', 'unavailable') AND lower(BTRIM(d.feedback->>'predicted_service')) IS DISTINCT FROM lower(BTRIM(d.feedback->>'confirmed_service')))::bigint AS routing_corrections, \
                COUNT(*) FILTER (WHERE NULLIF(BTRIM(d.feedback->>'predicted_priority'), '') IS NOT NULL AND lower(BTRIM(d.feedback->>'predicted_priority')) NOT IN ('unknown', 'unavailable'))::bigint AS priority_decisions, \
                COUNT(*) FILTER (WHERE NULLIF(BTRIM(d.feedback->>'predicted_priority'), '') IS NOT NULL AND lower(BTRIM(d.feedback->>'predicted_priority')) NOT IN ('unknown', 'unavailable') AND lower(BTRIM(d.feedback->>'predicted_priority')) IS DISTINCT FROM lower(BTRIM(d.confirmed_priority)))::bigint AS priority_corrections \
             FROM operator_decisions d JOIN tickets t ON t.id = d.ticket_id WHERE t.created_at >= ",
        );
        decision_metrics_query
            .push_bind(current_since)
            .push(" AND t.created_at <= now()");
        push_analytics_filters(&mut decision_metrics_query, query, "t");
        let decision_metrics_row = decision_metrics_query
            .build()
            .fetch_one(&self.pool)
            .await
            .map_err(|error| format!("analytics decision metrics: {error}"))?;
        let decision_metric_count = |column: &str| -> usize {
            decision_metrics_row
                .try_get::<i64, _>(column)
                .unwrap_or(0)
                .max(0) as usize
        };
        let classification_decisions = decision_metric_count("classification_decisions");
        let classification_corrections = decision_metric_count("classification_corrections");
        let routing_decisions = decision_metric_count("routing_decisions");
        let routing_corrections = decision_metric_count("routing_corrections");
        let priority_decisions = decision_metric_count("priority_decisions");
        let priority_corrections = decision_metric_count("priority_corrections");

        let mut relation_metrics_query = QueryBuilder::<Postgres>::new(
            "SELECT \
                COUNT(*)::bigint AS similarity_feedback_count, \
                COUNT(*) FILTER (WHERE rf.decision = 'CONFIRMED')::bigint AS similarity_confirmations, \
                COUNT(*) FILTER (WHERE rf.relation = 'DUPLICATE')::bigint AS duplicate_feedback_count, \
                COUNT(*) FILTER (WHERE rf.relation = 'DUPLICATE' AND rf.decision = 'CONFIRMED')::bigint AS duplicate_confirmations \
             FROM relation_feedback rf JOIN tickets t ON t.id = rf.ticket_id WHERE rf.created_at >= ",
        );
        relation_metrics_query
            .push_bind(current_since)
            .push(" AND rf.created_at <= now()");
        push_analytics_filters(&mut relation_metrics_query, query, "t");
        let relation_metrics_row = relation_metrics_query
            .build()
            .fetch_one(&self.pool)
            .await
            .map_err(|error| format!("analytics relation metrics: {error}"))?;
        let relation_metric_count = |column: &str| -> usize {
            relation_metrics_row
                .try_get::<i64, _>(column)
                .unwrap_or(0)
                .max(0) as usize
        };
        let similarity_feedback_count = relation_metric_count("similarity_feedback_count");
        let similarity_confirmations = relation_metric_count("similarity_confirmations");
        let duplicate_feedback_count = relation_metric_count("duplicate_feedback_count");
        let duplicate_confirmations = relation_metric_count("duplicate_confirmations");
        let runtime_metrics = RuntimeMetrics {
            operator_decision_time_minutes: avg_decision_minutes,
            operator_decision_time_samples: decision_time_samples.max(0) as usize,
            classification_correction_rate: metric_rate(
                classification_corrections,
                classification_decisions,
            ),
            classification_corrections,
            classification_decisions,
            routing_correction_rate: metric_rate(routing_corrections, routing_decisions),
            routing_corrections,
            routing_decisions,
            priority_correction_rate: metric_rate(priority_corrections, priority_decisions),
            priority_corrections,
            priority_decisions,
            similarity_usefulness: metric_rate(similarity_confirmations, similarity_feedback_count),
            similarity_feedback_count,
            duplicate_precision: metric_rate(duplicate_confirmations, duplicate_feedback_count),
            duplicate_feedback_count,
        };

        let mut region_query = QueryBuilder::<Postgres>::new(
            "SELECT r.id, COALESCE(r.name_ru, r.name_en, r.id) AS label, COUNT(t.id) FILTER (WHERE t.created_at >= ",
        );
        region_query
            .push_bind(current_since)
            .push(")::bigint AS tickets, COUNT(t.id) FILTER (WHERE t.created_at >= ")
            .push_bind(previous_since)
            .push(" AND t.created_at < ")
            .push_bind(current_since)
            .push(")::bigint AS previous_tickets, COUNT(t.id) FILTER (WHERE t.created_at >= ")
            .push_bind(current_since)
            .push(" AND lower(COALESCE(t.priority, '')) IN ('high', 'critical'))::bigint AS high_priority, COALESCE(AVG(p.confidence) FILTER (WHERE t.created_at >= ")
            .push_bind(current_since)
            .push("), 0)::float8 AS avg_confidence FROM regions r LEFT JOIN tickets t ON t.region_id = r.id AND t.created_at >= ")
            .push_bind(previous_since)
            .push(" AND t.created_at <= now() LEFT JOIN LATERAL (SELECT confidence FROM ticket_predictions WHERE ticket_id = t.id ORDER BY created_at DESC LIMIT 1) p ON TRUE WHERE TRUE");
        push_analytics_filters(&mut region_query, query, "t");
        region_query.push(" GROUP BY r.id, r.name_ru, r.name_en ORDER BY label, r.id");
        let region_rows = region_query
            .build()
            .fetch_all(&self.pool)
            .await
            .map_err(|error| format!("analytics regions: {error}"))?;
        let by_region = region_rows
            .into_iter()
            .map(|row| {
                let tickets = row.try_get::<i64, _>("tickets").unwrap_or(0).max(0);
                let previous = row
                    .try_get::<i64, _>("previous_tickets")
                    .unwrap_or(0)
                    .max(0);
                MetricBucket {
                    id: row.try_get("id").unwrap_or_default(),
                    label: row.try_get("label").unwrap_or_default(),
                    tickets: tickets as usize,
                    high_priority: row.try_get::<i64, _>("high_priority").unwrap_or(0).max(0)
                        as usize,
                    avg_confidence: row.try_get::<f64, _>("avg_confidence").unwrap_or(0.0) as f32,
                    change_abs: Some(tickets - previous),
                    change_pct: percent_change(tickets, previous).map(|value| value as f32),
                }
            })
            .collect::<Vec<_>>();

        let mut topic_query = QueryBuilder::<Postgres>::new(
            "SELECT tp.id, COALESCE(tp.name_ru, tp.name_kk, tp.id) AS label, COUNT(t.id) FILTER (WHERE t.created_at >= ",
        );
        topic_query
            .push_bind(current_since)
            .push(")::bigint AS tickets, COUNT(t.id) FILTER (WHERE t.created_at >= ")
            .push_bind(previous_since)
            .push(" AND t.created_at < ")
            .push_bind(current_since)
            .push(")::bigint AS previous_tickets, COUNT(t.id) FILTER (WHERE t.created_at >= ")
            .push_bind(current_since)
            .push(" AND lower(COALESCE(t.priority, '')) IN ('high', 'critical'))::bigint AS high_priority, COALESCE(AVG(p.confidence) FILTER (WHERE t.created_at >= ")
            .push_bind(current_since)
            .push("), 0)::float8 AS avg_confidence FROM topics tp LEFT JOIN tickets t ON t.topic_id = tp.id AND t.created_at >= ")
            .push_bind(previous_since)
            .push(" AND t.created_at <= now() LEFT JOIN LATERAL (SELECT confidence FROM ticket_predictions WHERE ticket_id = t.id ORDER BY created_at DESC LIMIT 1) p ON TRUE WHERE TRUE");
        push_analytics_filters(&mut topic_query, query, "t");
        topic_query.push(" GROUP BY tp.id, tp.name_ru, tp.name_kk ORDER BY tickets DESC, tp.id");
        let topic_rows = topic_query
            .build()
            .fetch_all(&self.pool)
            .await
            .map_err(|error| format!("analytics topics: {error}"))?;
        let by_topic = topic_rows
            .into_iter()
            .map(|row| {
                let tickets = row.try_get::<i64, _>("tickets").unwrap_or(0).max(0);
                let previous = row
                    .try_get::<i64, _>("previous_tickets")
                    .unwrap_or(0)
                    .max(0);
                MetricBucket {
                    id: row.try_get("id").unwrap_or_default(),
                    label: row.try_get("label").unwrap_or_default(),
                    tickets: tickets as usize,
                    high_priority: row.try_get::<i64, _>("high_priority").unwrap_or(0).max(0)
                        as usize,
                    avg_confidence: row.try_get::<f64, _>("avg_confidence").unwrap_or(0.0) as f32,
                    change_abs: Some(tickets - previous),
                    change_pct: percent_change(tickets, previous).map(|value| value as f32),
                }
            })
            .collect::<Vec<_>>();

        let mut series_query = QueryBuilder::<Postgres>::new(
            "SELECT to_char(date_trunc('day', t.created_at AT TIME ZONE 'UTC'), 'YYYY-MM-DD') AS date, COUNT(*)::bigint AS tickets, COUNT(*) FILTER (WHERE t.status IN ('RESOLVED', 'CLOSED'))::bigint AS resolved FROM tickets t WHERE t.created_at >= ",
        );
        series_query
            .push_bind(current_since)
            .push(" AND t.created_at <= now()");
        push_analytics_filters(&mut series_query, query, "t");
        series_query.push(
            " GROUP BY date_trunc('day', t.created_at AT TIME ZONE 'UTC') ORDER BY date_trunc('day', t.created_at AT TIME ZONE 'UTC')",
        );
        let series_rows = series_query
            .build()
            .fetch_all(&self.pool)
            .await
            .map_err(|error| format!("analytics time series: {error}"))?;
        let series = series_rows
            .into_iter()
            .map(|row| {
                (
                    row.try_get::<String, _>("date").unwrap_or_default(),
                    TimeSeriesPoint {
                        date: row.try_get("date").unwrap_or_default(),
                        tickets: row.try_get::<i64, _>("tickets").unwrap_or(0).max(0) as u32,
                        resolved: row.try_get::<i64, _>("resolved").unwrap_or(0).max(0) as u32,
                    },
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        let time_series = (0..=days)
            .map(|offset| {
                let date =
                    (current_since.date_naive() + chrono::Duration::days(offset)).to_string();
                series.get(&date).cloned().unwrap_or(TimeSeriesPoint {
                    date,
                    tickets: 0,
                    resolved: 0,
                })
            })
            .collect();
        Ok(AnalyticsResponse {
            generated_at: Utc::now().to_rfc3339(),
            source: "postgres".to_owned(),
            range: query.range.clone().unwrap_or_else(|| format!("{days}d")),
            overview: json!({
                "total_tickets": total_tickets.max(0),
                "open_tickets": open_tickets.max(0),
                "resolved_tickets": resolved_tickets.max(0),
                "high_priority_tickets": high_priority_tickets.max(0),
                "operator_decisions": operator_decisions.max(0),
                "confirmed_decisions": confirmed_decisions.max(0),
                "corrected_decisions": corrected_decisions.max(0),
                "avg_decision_minutes": avg_decision_minutes,
                "average_confidence": average_confidence,
                "previous_total_tickets": previous_total.max(0),
                "change_abs": total_change,
                "change_pct": total_change_pct,
            }),
            runtime_metrics,
            by_region,
            by_topic,
            time_series,
        })
    }

    pub async fn analytics_drilldown(
        &self,
        query: &AnalyticsDrilldownQuery,
        dimension: &str,
    ) -> Result<AnalyticsDrilldownResponse, String> {
        let mut builder = QueryBuilder::<Postgres>::new(
            "SELECT t.id, t.region_id, COALESCE(r.name_ru, r.name_en, t.region_id) AS region_name, t.topic_id, COALESCE(tp.name_ru, tp.name_kk, t.topic_id) AS topic_label, COALESCE(t.priority, 'normal') AS priority, t.status, t.created_at FROM tickets t LEFT JOIN regions r ON r.id = t.region_id LEFT JOIN topics tp ON tp.id = t.topic_id",
        );
        push_drilldown_predicates(&mut builder, query, dimension)?;

        let limit = query.limit.unwrap_or(100).clamp(1, 100);
        let offset = query.offset.unwrap_or(0);
        builder
            .push(" ORDER BY t.created_at DESC, t.id DESC LIMIT ")
            .push_bind(limit as i64)
            .push(" OFFSET ")
            .push_bind(offset as i64);
        let rows: Vec<DbAnalyticsDrilldownTicket> = builder
            .build_query_as()
            .fetch_all(&self.pool)
            .await
            .map_err(|error| format!("analytics drilldown: {error}"))?;
        let mut count_builder =
            QueryBuilder::<Postgres>::new("SELECT COUNT(*)::bigint FROM tickets t");
        push_drilldown_predicates(&mut count_builder, query, dimension)?;
        let total = count_builder
            .build_query_scalar::<i64>()
            .fetch_one(&self.pool)
            .await
            .map_err(|error| format!("analytics drilldown count: {error}"))?
            .max(0) as usize;
        Ok(AnalyticsDrilldownResponse {
            items: rows
                .into_iter()
                .map(|row| AnalyticsDrilldownTicket {
                    id: row.id.to_string(),
                    region_id: row.region_id,
                    region_name: row.region_name,
                    topic_id: row.topic_id,
                    topic_label: row.topic_label,
                    priority: row.priority,
                    status: row.status.to_ascii_lowercase(),
                    created_at: row.created_at.to_rfc3339(),
                })
                .collect(),
            total,
            limit,
            offset,
        })
    }

    pub async fn forecast(&self, query: &ForecastQuery) -> Result<ForecastResponse, String> {
        let horizon_days = query
            .horizon
            .or(query.horizon_days)
            .unwrap_or(30)
            .clamp(1, 90);
        let filters = AnalyticsQuery {
            region_id: query.region_id.clone(),
            topic_id: query.topic_id.clone(),
            service_id: query.service_id.clone(),
            status: query.status.clone(),
            district: query.district.clone(),
            channel: query.channel.clone(),
            range: Some("366d".to_owned()),
        };
        let mut series_query = QueryBuilder::<Postgres>::new(
            "SELECT to_char(date_trunc('day', t.created_at), 'YYYY-MM-DD') AS date, COUNT(*)::float8 AS value FROM tickets t WHERE t.created_at >= now() - ",
        );
        series_query
            .push_bind(366_i32)
            .push(" * interval '1 day' AND t.created_at <= now()");
        push_analytics_filters(&mut series_query, &filters, "t");
        series_query.push(
            " GROUP BY date_trunc('day', t.created_at) ORDER BY date_trunc('day', t.created_at)",
        );
        let rows = series_query
            .build()
            .fetch_all(&self.pool)
            .await
            .map_err(|error| format!("forecast history: {error}"))?;
        let observed_days = rows.len();
        let mut dates = Vec::with_capacity(rows.len());
        let mut values = Vec::with_capacity(rows.len());
        let mut next_date: Option<NaiveDate> = None;
        for row in rows {
            let date = row
                .try_get::<String, _>("date")
                .map_err(|error| format!("forecast date: {error}"))?;
            let parsed = NaiveDate::parse_from_str(&date, "%Y-%m-%d")
                .map_err(|error| format!("forecast date format: {error}"))?;
            while next_date.is_some_and(|next| next < parsed) {
                let missing = next_date.expect("date is present");
                dates.push(missing.to_string());
                values.push(0.0);
                next_date = Some(missing + chrono::Duration::days(1));
            }
            dates.push(date);
            values.push(
                row.try_get::<f64, _>("value")
                    .map_err(|error| format!("forecast value: {error}"))?,
            );
            next_date = Some(parsed + chrono::Duration::days(1));
        }
        while next_date.is_some_and(|next| next <= Utc::now().date_naive()) {
            let missing = next_date.expect("date is present");
            dates.push(missing.to_string());
            values.push(0.0);
            next_date = Some(missing + chrono::Duration::days(1));
        }
        let history = dates
            .iter()
            .zip(values.iter())
            .map(|(date, value)| TimeSeriesPoint {
                date: date.clone(),
                tickets: value.max(0.0).round() as u32,
                resolved: 0,
            })
            .collect::<Vec<_>>();
        if observed_days < crate::FORECAST_SEASON_LENGTH_DAYS {
            return Ok(ForecastResponse {
                source: "postgres".to_owned(),
                model_version: self.forecast_model_version.clone(),
                model: "seasonal-naive-baseline".to_owned(),
                status: "INSUFFICIENT_HISTORY".to_owned(),
                insufficient_history: true,
                horizon_days,
                history,
                forecast_start: None,
                points: Vec::new(),
                expected_peaks: Vec::new(),
                backtest: json!({
                    "status": "INSUFFICIENT_HISTORY",
                    "sample_count": 0,
                    "observed_days": observed_days,
                    "required_days": crate::FORECAST_SEASON_LENGTH_DAYS,
                }),
            });
        }
        let payload = self
            .client
            .post(format!("{}/internal/v1/forecast", self.ml_service_url))
            .json(&json!({
                "values": values,
                "horizon": horizon_days,
                "season_length": 7,
                "model_version": &self.forecast_model_version,
            }))
            .send()
            .await
            .map_err(|error| format!("ML forecast request: {error}"))?
            .error_for_status()
            .map_err(|error| format!("ML forecast response: {error}"))?
            .json::<Value>()
            .await
            .map_err(|error| format!("ML forecast JSON: {error}"))?;
        let model_version = payload
            .get("model_version")
            .and_then(Value::as_str)
            .unwrap_or(&self.forecast_model_version)
            .to_owned();
        let model = payload
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("seasonal_naive")
            .to_owned();
        let status = payload
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("INSUFFICIENT_HISTORY")
            .to_owned();
        let insufficient_history = payload
            .get("insufficient_history")
            .and_then(Value::as_bool)
            .unwrap_or(values.len() < crate::FORECAST_SEASON_LENGTH_DAYS)
            || status == "INSUFFICIENT_HISTORY";
        let forecast_values = if insufficient_history {
            Vec::new()
        } else {
            let forecast_values = payload
                .get("forecast")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if forecast_values.len() != horizon_days as usize {
                return Err(format!(
                    "forecast service returned {} points for a {horizon_days}-day horizon",
                    forecast_values.len()
                ));
            }
            forecast_values
        };
        let start_date = dates
            .last()
            .and_then(|value| chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d").ok())
            .unwrap_or_else(|| Utc::now().date_naive());
        let points = forecast_values
            .iter()
            .enumerate()
            .map(|(index, value)| TimeSeriesPoint {
                date: (start_date + chrono::Duration::days(index as i64 + 1)).to_string(),
                tickets: value.as_f64().unwrap_or(0.0).max(0.0).round() as u32,
                resolved: 0,
            })
            .collect::<Vec<_>>();
        let expected_peaks = payload
            .get("expected_peaks")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_u64)
            .filter_map(|index| points.get(index as usize).map(|point| point.date.clone()))
            .collect::<Vec<_>>();
        let forecast_start = points.first().map(|point| point.date.clone());
        Ok(ForecastResponse {
            source: "postgres+ml".to_owned(),
            model_version,
            model,
            status: if insufficient_history {
                "INSUFFICIENT_HISTORY".to_owned()
            } else {
                status
            },
            insufficient_history,
            horizon_days,
            history,
            forecast_start,
            points,
            expected_peaks,
            backtest: payload
                .get("backtest")
                .cloned()
                .unwrap_or_else(|| json!({})),
        })
    }

    pub async fn analytics_query(&self, query: &QueryIntentRequest) -> Result<Value, String> {
        let intent = validate_query_intent(query)?;
        let filters = query_analytics_filters(query)?;
        let report = self.analytics(&filters).await?;
        let forecast = if intent == "forecast" {
            let horizon = query.horizon_days.unwrap_or(30);
            Some(
                self.forecast(&ForecastQuery {
                    horizon: Some(horizon),
                    horizon_days: None,
                    region_id: filters.region_id.clone(),
                    topic_id: filters.topic_id.clone(),
                    service_id: filters.service_id.clone(),
                    status: filters.status.clone(),
                    district: filters.district.clone(),
                    channel: filters.channel.clone(),
                })
                .await?,
            )
        } else {
            None
        };
        build_query_intent_result(query, &filters, &report, forecast.as_ref())
    }

    pub async fn list_alerts(&self, query: &AlertQuery) -> Result<Vec<Alert>, String> {
        let mut builder = QueryBuilder::<Postgres>::new("SELECT id FROM alerts WHERE TRUE");
        if let Some(status) = query.status.as_deref() {
            builder
                .push(" AND status = ")
                .push_bind(status.trim().to_ascii_uppercase());
        }
        if let Some(severity) = query.severity.as_deref() {
            builder
                .push(" AND severity = ")
                .push_bind(severity.trim().to_ascii_uppercase());
        }
        if let Some(region_id) = query.region_id.as_deref() {
            builder
                .push(" AND region_id = ")
                .push_bind(normalize_region_id(region_id));
        }
        builder.push(" ORDER BY created_at DESC, id DESC");
        let rows = builder
            .build()
            .fetch_all(&self.pool)
            .await
            .map_err(|error| format!("list alerts: {error}"))?;
        let mut alerts = Vec::with_capacity(rows.len());
        for row in rows {
            let id: i64 = row
                .try_get("id")
                .map_err(|error| format!("alert id: {error}"))?;
            alerts.push(self.alert_from_id(id).await?);
        }
        Ok(alerts)
    }

    pub async fn list_alerts_for_analytics(
        &self,
        query: &AnalyticsQuery,
    ) -> Result<Vec<Alert>, String> {
        let current_since = Utc::now() - chrono::Duration::days(analytics_days(query));
        let mut builder = QueryBuilder::<Postgres>::new(
            "SELECT DISTINCT a.id, a.created_at, t.id AS ticket_id FROM alerts a JOIN alert_ticket_links atl ON atl.alert_id = a.id JOIN tickets t ON t.id = atl.ticket_id WHERE upper(a.status) <> 'CLOSED' AND t.created_at >= ",
        );
        builder
            .push_bind(current_since)
            .push(" AND t.created_at <= now()");
        push_analytics_filters(&mut builder, query, "t");
        builder.push(" ORDER BY a.created_at DESC, a.id DESC, t.id");
        let rows = builder
            .build()
            .fetch_all(&self.pool)
            .await
            .map_err(|error| format!("list alerts for analytics: {error}"))?;

        let mut alert_ids = Vec::new();
        let mut linked_tickets = std::collections::BTreeMap::<i64, Vec<String>>::new();
        for row in rows {
            let alert_id: i64 = row
                .try_get("id")
                .map_err(|error| format!("analytics alert id: {error}"))?;
            let ticket_id: i64 = row
                .try_get("ticket_id")
                .map_err(|error| format!("analytics alert ticket id: {error}"))?;
            let tickets = linked_tickets.entry(alert_id).or_default();
            if tickets.is_empty() {
                alert_ids.push(alert_id);
            }
            tickets.push(ticket_id.to_string());
        }

        let mut alerts = Vec::with_capacity(alert_ids.len());
        for alert_id in alert_ids {
            let mut alert = self.alert_from_id(alert_id).await?;
            let ticket_ids = linked_tickets.remove(&alert_id).unwrap_or_default();
            alert.ticket_count = ticket_ids.len().min(u32::MAX as usize) as u32;
            alert.linked_ticket_ids = ticket_ids;
            alerts.push(alert);
        }
        Ok(alerts)
    }

    pub async fn get_alert(&self, alert_id: &str) -> Result<Alert, String> {
        let id = alert_id
            .parse::<i64>()
            .map_err(|_| format!("alert {alert_id} not found"))?;
        self.alert_from_id(id).await
    }

    pub async fn acknowledge_alert(&self, alert_id: &str, user_id: &str) -> Result<Alert, String> {
        let id = alert_id
            .parse::<i64>()
            .map_err(|_| format!("alert {alert_id} not found"))?;
        let updated = sqlx::query("UPDATE alerts SET status = CASE WHEN status = 'CLOSED' THEN status ELSE 'ACKNOWLEDGED' END, acknowledged_at = now(), acknowledged_by = $2 WHERE id = $1 RETURNING id")
            .bind(id)
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("acknowledge alert: {error}"))?;
        if updated.is_none() {
            return Err(format!("alert {alert_id} not found"));
        }
        self.alert_from_id(id).await
    }

    pub async fn close_alert(&self, alert_id: &str, user_id: &str) -> Result<Alert, String> {
        let id = alert_id
            .parse::<i64>()
            .map_err(|_| format!("alert {alert_id} not found"))?;
        let updated = sqlx::query("UPDATE alerts SET status = 'CLOSED', closed_at = now(), closed_by = $2 WHERE id = $1 RETURNING id")
            .bind(id)
            .bind(user_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("close alert: {error}"))?;
        if updated.is_none() {
            return Err(format!("alert {alert_id} not found"));
        }
        self.alert_from_id(id).await
    }

    pub async fn detect_alerts(
        &self,
        config: &AlertDetectorConfig,
    ) -> Result<AlertDetectionRun, String> {
        config.validate();
        let period_days = i32::try_from(config.period_days)
            .map_err(|_| "alert period_days exceeds PostgreSQL integer range".to_owned())?;
        let baseline_periods = i32::try_from(config.baseline_periods)
            .map_err(|_| "alert baseline_periods exceeds PostgreSQL integer range".to_owned())?;
        let rows = sqlx::query(
            r#"
            WITH detector_bounds AS (
                SELECT now() AS period_end, $1::integer AS period_days, $2::integer AS baseline_periods
            ), pairs AS (
                SELECT DISTINCT t.region_id, t.topic_id
                FROM tickets t CROSS JOIN detector_bounds b
                WHERE t.region_id IS NOT NULL AND t.topic_id IS NOT NULL
                  AND t.created_at >= b.period_end - ((b.period_days * (b.baseline_periods + 1)) * interval '1 day')
                  AND t.created_at < b.period_end
            ), coverage AS (
                SELECT t.region_id, min(t.created_at) AS coverage_start
                FROM tickets t CROSS JOIN detector_bounds b
                WHERE t.created_at < b.period_end AND t.region_id IS NOT NULL
                GROUP BY t.region_id
            ), periods AS (
                SELECT p.region_id, p.topic_id, bucket.period_index,
                    b.period_end - ((bucket.period_index + 1) * b.period_days) * interval '1 day' AS period_start,
                    b.period_end - (bucket.period_index * b.period_days) * interval '1 day' AS period_end
                FROM pairs p CROSS JOIN detector_bounds b
                CROSS JOIN LATERAL generate_series(0, b.baseline_periods) AS bucket(period_index)
            ), period_counts AS (
                SELECT p.region_id, p.topic_id, p.period_index, p.period_start, p.period_end,
                    count(t.id)::bigint AS ticket_count
                FROM periods p LEFT JOIN tickets t
                  ON t.region_id = p.region_id AND t.topic_id = p.topic_id
                 AND t.created_at >= p.period_start AND t.created_at < p.period_end
                GROUP BY p.region_id, p.topic_id, p.period_index, p.period_start, p.period_end
            )
            SELECT counts.region_id, counts.topic_id, coverage.coverage_start,
                array_agg(counts.ticket_count ORDER BY counts.period_index) AS period_counts,
                max(counts.period_end) FILTER (WHERE counts.period_index = 0) AS period_end,
                ARRAY(
                    SELECT t.id FROM tickets t CROSS JOIN detector_bounds b
                    WHERE t.region_id = counts.region_id AND t.topic_id = counts.topic_id
                      AND t.created_at >= b.period_end - (b.period_days * interval '1 day')
                      AND t.created_at < b.period_end
                    ORDER BY t.id
                ) AS current_ticket_ids
            FROM period_counts counts
            JOIN coverage ON coverage.region_id = counts.region_id
            GROUP BY counts.region_id, counts.topic_id, coverage.coverage_start
            ORDER BY counts.region_id, counts.topic_id
            "#,
        )
        .bind(period_days)
        .bind(baseline_periods)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| format!("detect alert series: {error}"))?;

        let mut evaluated_series = 0;
        let mut insufficient_history_series = 0;
        let mut new_alerts = 0;
        let mut alert_ids = Vec::new();
        for row in rows {
            let input = AlertSeriesInput {
                region_id: row
                    .try_get("region_id")
                    .map_err(|error| format!("alert region: {error}"))?,
                topic_id: row
                    .try_get("topic_id")
                    .map_err(|error| format!("alert topic: {error}"))?,
                period_end: row
                    .try_get("period_end")
                    .map_err(|error| format!("alert period end: {error}"))?,
                coverage_start: row
                    .try_get("coverage_start")
                    .map_err(|error| format!("alert history coverage: {error}"))?,
                counts: row
                    .try_get("period_counts")
                    .map_err(|error| format!("alert period counts: {error}"))?,
                current_ticket_ids: row
                    .try_get("current_ticket_ids")
                    .map_err(|error| format!("alert source ticket ids: {error}"))?,
            };
            let evidence = match evaluate_series(&input, config) {
                AlertEvaluation::InsufficientHistory => {
                    insufficient_history_series += 1;
                    continue;
                }
                AlertEvaluation::BelowThreshold => {
                    evaluated_series += 1;
                    continue;
                }
                AlertEvaluation::Anomaly(evidence) => {
                    evaluated_series += 1;
                    evidence
                }
            };
            if input.current_ticket_ids.len() as i64 != evidence.current_count {
                return Err(format!(
                    "alert source ticket count changed while evaluating {} × {}",
                    input.region_id, input.topic_id
                ));
            }

            let source_ticket_ids = input
                .current_ticket_ids
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            let required_start = evidence.period_start
                - chrono::Duration::days(
                    i64::from(config.period_days) * i64::from(config.baseline_periods),
                );
            let detail = json!({
                "schema_version": 1,
                "detector": "region_topic_median_mad",
                "detector_version": config.detector_version,
                "configuration": config,
                "message": "Зафиксирован необычный рост обращений. Требуется проверка.",
                "region_id": evidence.region_id,
                "topic_id": evidence.topic_id,
                "period_start": evidence.period_start,
                "period_end": evidence.period_end,
                "current_count": evidence.current_count,
                "source_ticket_ids": source_ticket_ids,
                "coverage_start": input.coverage_start,
                "required_history_start": required_start,
                "historical_counts": evidence.historical_counts,
                "baseline": evidence.baseline,
                "median_absolute_deviation": evidence.median_absolute_deviation,
                "robust_dispersion": evidence.robust_dispersion,
                "deviation": evidence.deviation,
                "robust_z": evidence.robust_z,
                "ratio": evidence.ratio,
                "severity": evidence.severity,
                "trigger_reasons": evidence.reasons,
            });
            let cooldown_start =
                Utc::now() - chrono::Duration::hours(i64::from(config.cooldown_hours));
            let lock_key = format!(
                "pulse109-alert:{}:{}",
                evidence.region_id, evidence.topic_id
            );
            let mut transaction = self
                .pool
                .begin()
                .await
                .map_err(|error| format!("begin alert persistence: {error}"))?;
            sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))")
                .bind(lock_key)
                .execute(&mut *transaction)
                .await
                .map_err(|error| format!("lock alert incident: {error}"))?;
            let recent = sqlx::query(
                "SELECT id, status FROM alerts WHERE alert_type = 'TOPIC_SPIKE' AND region_id = $1 AND topic_id = $2 AND created_at >= $3 ORDER BY created_at DESC, id DESC LIMIT 1",
            )
            .bind(&evidence.region_id)
            .bind(&evidence.topic_id)
            .bind(cooldown_start)
            .fetch_optional(&mut *transaction)
            .await
            .map_err(|error| format!("find alert cooldown: {error}"))?;
            if let Some(recent) = recent {
                let id: i64 = recent
                    .try_get("id")
                    .map_err(|error| format!("cooldown alert id: {error}"))?;
                let status: String = recent
                    .try_get("status")
                    .map_err(|error| format!("cooldown alert status: {error}"))?;
                transaction
                    .commit()
                    .await
                    .map_err(|error| format!("finish alert cooldown check: {error}"))?;
                if status != "CLOSED" {
                    alert_ids.push(id);
                }
                continue;
            }

            let incident_key = format!(
                "topic_spike:{}:{}:{}",
                evidence.region_id,
                evidence.topic_id,
                evidence.period_end.date_naive()
            );
            let current_count = i32::try_from(evidence.current_count)
                .map_err(|_| "alert current_count exceeds PostgreSQL integer range".to_owned())?;
            let inserted_id = sqlx::query_scalar::<_, i64>(
                "INSERT INTO alerts (incident_key, alert_type, severity, status, region_id, topic_id, period_start, period_end, current_count, baseline, deviation, detail) VALUES ($1, 'TOPIC_SPIKE', $2, 'OPEN', $3, $4, $5, $6, $7, $8, $9, $10) ON CONFLICT (incident_key) DO NOTHING RETURNING id",
            )
            .bind(&incident_key)
            .bind(evidence.severity)
            .bind(&evidence.region_id)
            .bind(&evidence.topic_id)
            .bind(evidence.period_start)
            .bind(evidence.period_end)
            .bind(current_count)
            .bind(evidence.baseline)
            .bind(evidence.deviation)
            .bind(detail)
            .fetch_optional(&mut *transaction)
            .await
            .map_err(|error| format!("persist alert: {error}"))?;
            let id = if let Some(id) = inserted_id {
                id
            } else {
                let existing = sqlx::query("SELECT id, status FROM alerts WHERE incident_key = $1")
                    .bind(&incident_key)
                    .fetch_one(&mut *transaction)
                    .await
                    .map_err(|error| format!("fetch existing alert: {error}"))?;
                let status: String = existing
                    .try_get("status")
                    .map_err(|error| format!("existing alert status: {error}"))?;
                let id: i64 = existing
                    .try_get("id")
                    .map_err(|error| format!("existing alert id: {error}"))?;
                transaction
                    .commit()
                    .await
                    .map_err(|error| format!("finish existing alert check: {error}"))?;
                if status != "CLOSED" {
                    alert_ids.push(id);
                }
                continue;
            };
            sqlx::query(
                "INSERT INTO alert_ticket_links (alert_id, ticket_id) SELECT $1, ticket_id FROM unnest($2::bigint[]) AS linked(ticket_id) ON CONFLICT DO NOTHING",
            )
            .bind(id)
            .bind(&input.current_ticket_ids)
            .execute(&mut *transaction)
            .await
            .map_err(|error| format!("link alert source tickets: {error}"))?;
            transaction
                .commit()
                .await
                .map_err(|error| format!("commit detected alert: {error}"))?;
            new_alerts += 1;
            alert_ids.push(id);
        }

        let mut items = Vec::with_capacity(alert_ids.len());
        for id in alert_ids {
            items.push(self.alert_from_id(id).await?);
        }
        Ok(AlertDetectionRun {
            status: if evaluated_series == 0 || insufficient_history_series > 0 {
                "INSUFFICIENT_HISTORY"
            } else {
                "OK"
            },
            source: "postgresql",
            evaluated_series,
            insufficient_history_series,
            new_alerts,
            items,
        })
    }

    async fn alert_from_id(&self, id: i64) -> Result<Alert, String> {
        let row = sqlx::query("SELECT a.id, a.incident_key, a.alert_type, a.severity, a.status, a.region_id, a.topic_id, a.period_start, a.period_end, a.current_count, a.baseline::double precision AS baseline, a.deviation::double precision AS deviation, a.created_at, a.acknowledged_at, a.acknowledged_by, a.closed_at, a.closed_by, COALESCE(a.detail, '{}'::jsonb) AS detail, COALESCE(tp.name_ru, tp.name_kk, a.topic_id, 'Все темы') AS topic_name FROM alerts a LEFT JOIN topics tp ON tp.id = a.topic_id WHERE a.id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("fetch alert: {error}"))?
            .ok_or_else(|| format!("alert {id} not found"))?;
        let link_rows = sqlx::query(
            "SELECT ticket_id::text FROM alert_ticket_links WHERE alert_id = $1 ORDER BY ticket_id",
        )
        .bind(id)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| format!("fetch alert links: {error}"))?;
        let mut links = Vec::with_capacity(link_rows.len());
        for item in link_rows {
            links.push(
                item.try_get::<String, _>("ticket_id")
                    .map_err(|error| format!("alert link ticket id: {error}"))?,
            );
        }
        let detail: Value = row
            .try_get("detail")
            .map_err(|error| format!("alert detail: {error}"))?;
        let topic_name: String = row
            .try_get("topic_name")
            .unwrap_or_else(|_| "Все темы".to_owned());
        let current_count: i32 = row
            .try_get("current_count")
            .map_err(|error| format!("alert current_count: {error}"))?;
        let period_start: DateTime<Utc> = row
            .try_get("period_start")
            .map_err(|error| format!("alert period_start: {error}"))?;
        let period_end: DateTime<Utc> = row
            .try_get("period_end")
            .map_err(|error| format!("alert period_end: {error}"))?;
        let baseline: Option<f64> = row
            .try_get("baseline")
            .map_err(|error| format!("alert baseline: {error}"))?;
        let deviation: Option<f64> = row
            .try_get("deviation")
            .map_err(|error| format!("alert deviation: {error}"))?;
        let title = detail
            .get("title")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("Всплеск обращений: {topic_name}"));
        let description = "Зафиксирован необычный рост обращений. Требуется проверка.".to_owned();
        let detected_at: DateTime<Utc> = row
            .try_get("created_at")
            .map_err(|error| format!("alert created_at: {error}"))?;
        let acknowledged_at: Option<DateTime<Utc>> = row.try_get("acknowledged_at").unwrap_or(None);
        let closed_at: Option<DateTime<Utc>> = row.try_get("closed_at").unwrap_or(None);
        Ok(Alert {
            id: id.to_string(),
            incident_key: row.try_get("incident_key").unwrap_or_default(),
            alert_type: row.try_get("alert_type").unwrap_or_default(),
            severity: row.try_get("severity").unwrap_or_default(),
            status: row.try_get("status").unwrap_or_default(),
            title,
            description,
            region_id: row
                .try_get("region_id")
                .unwrap_or_else(|_| "ALL".to_owned()),
            topic_id: row.try_get("topic_id").unwrap_or_else(|_| "ALL".to_owned()),
            ticket_count: current_count.max(0) as u32,
            period_start: Some(period_start.to_rfc3339()),
            period_end: Some(period_end.to_rfc3339()),
            current_count: current_count.max(0) as u32,
            baseline,
            deviation,
            robust_z: detail.get("robust_z").and_then(Value::as_f64),
            ratio: detail.get("ratio").and_then(Value::as_f64),
            detector_version: detail
                .get("detector_version")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            linked_ticket_ids: links,
            created_at: detected_at.to_rfc3339(),
            detected_at: detected_at.to_rfc3339(),
            acknowledged_by: row.try_get("acknowledged_by").unwrap_or(None),
            acknowledged_at: acknowledged_at.map(|value| value.to_rfc3339()),
            closed_by: row.try_get("closed_by").unwrap_or(None),
            closed_at: closed_at.map(|value| value.to_rfc3339()),
            detail,
        })
    }

    pub async fn learning_overview(&self) -> Result<LearningOverview, String> {
        let ids = sqlx::query("SELECT id FROM learning_cycles ORDER BY created_at DESC, id DESC")
            .fetch_all(&self.pool)
            .await
            .map_err(|error| format!("list learning cycles: {error}"))?;
        let mut items = Vec::with_capacity(ids.len());
        for row in ids {
            let id: i64 = row
                .try_get("id")
                .map_err(|error| format!("cycle id: {error}"))?;
            items.push(self.learning_cycle_from_id(id).await?);
        }
        let active_cycle = items
            .iter()
            .find(|cycle| {
                matches!(
                    cycle.state.as_str(),
                    "COLLECT" | "TRAINING" | "EVALUATE" | "DECISION"
                )
            })
            .cloned();
        let production_model = self
            .list_models(&ModelQuery {
                status: Some("PRODUCTION".to_owned()),
            })
            .await?
            .into_iter()
            .next();
        Ok(LearningOverview {
            items,
            active_cycle,
            production_model,
            controlled_loop: json!({
                "stages": ["COLLECT", "TRAINING", "EVALUATE", "DECISION", "PROMOTED", "REJECTED"],
                "production_auto_update": false,
                "trainer": "versioned Data/ML trainer; test fake is demo/test only",
            }),
        })
    }

    pub async fn create_learning_cycle(
        &self,
        request: &CreateLearningCycleRequest,
        config: &Config,
        actor_id: &str,
        request_id: &str,
    ) -> Result<LearningCycle, String> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin learning cycle: {error}"))?;
        sqlx::query("SELECT pg_advisory_xact_lock(hashtext('pulse109:classifier-learning-cycle'))")
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("lock classifier learning cycle: {error}"))?;
        let active_cycle_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS (SELECT 1 FROM learning_cycles WHERE state IN ('COLLECT', 'TRAINING', 'EVALUATE', 'DECISION'))",
        )
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("check active learning cycle: {error}"))?;
        if active_cycle_exists {
            return Err("an active classifier learning cycle already exists".to_owned());
        }
        let suffix = Utc::now().timestamp_nanos_opt().unwrap_or_default();
        let cycle_id = format!("cycle-{suffix}");
        let candidate_model_version = request
            .candidate_model_version
            .clone()
            .unwrap_or_else(|| format!("classifier-candidate-{suffix}"));
        let production_model_version: Option<String> = sqlx::query_scalar(
            "SELECT model_version FROM model_versions WHERE status = 'PRODUCTION' ORDER BY created_at DESC LIMIT 1",
        )
        .fetch_optional(&mut *tx)
        .await
        .map_err(|error| format!("read production model for learning cycle: {error}"))?;
        let evaluation_dataset_version = request
            .evaluation_dataset_version
            .clone()
            .or_else(|| config.learning_evaluation_dataset_version.clone());
        if let Some(dataset_version) = evaluation_dataset_version.as_deref() {
            let registered_ticket_count: Option<i64> = sqlx::query_scalar(
                "SELECT COUNT(dtl.ticket_id) FROM dataset_versions dv LEFT JOIN dataset_ticket_links dtl USING (dataset_version) WHERE dv.dataset_version = $1 GROUP BY dv.dataset_version",
            )
            .bind(dataset_version)
            .fetch_optional(&mut *tx)
            .await
            .map_err(|error| format!("check frozen evaluation dataset: {error}"))?;
            match registered_ticket_count {
                None => {
                    return Err(format!(
                        "frozen evaluation dataset {dataset_version} is not registered"
                    ));
                }
                Some(0) => {
                    return Err(format!(
                        "frozen evaluation dataset {dataset_version} has no ticket links"
                    ));
                }
                Some(_) => {}
            }
        }
        let id: i64 = sqlx::query_scalar("INSERT INTO learning_cycles (cycle_id, state, collect_started_at, collect_ends_at, production_model_version, candidate_model_version, min_feedback_count, promotion_policy_version, manual_close_enabled, frozen_evaluation_dataset_version) VALUES ($1, 'COLLECT', now(), now() + make_interval(hours => $2), $3, $4, $5, $6, $7, $8) RETURNING id")
            .bind(&cycle_id)
            .bind(config.learning_cycle_duration_hours)
            .bind(&production_model_version)
            .bind(&candidate_model_version)
            .bind(config.learning_min_feedback_count)
            .bind(&config.learning_promotion_policy_version)
            .bind(config.learning_manual_close_enabled)
            .bind(&evaluation_dataset_version)
            .fetch_one(&mut *tx)
            .await
            .map_err(|error| format!("create learning cycle: {error}"))?;
        if let Some(dataset_version) = evaluation_dataset_version.as_deref() {
            sqlx::query("INSERT INTO learning_cycle_evaluation_tickets (learning_cycle_id, dataset_version, ticket_id) SELECT $1, $2, ticket_id FROM dataset_ticket_links WHERE dataset_version = $2 ON CONFLICT DO NOTHING")
                .bind(id)
                .bind(dataset_version)
                .execute(&mut *tx)
                .await
                .map_err(|error| format!("freeze evaluation ticket IDs: {error}"))?;
        }
        sqlx::query("INSERT INTO audit_log (actor_id, action, entity_type, entity_id, request_id, metadata) VALUES ($1, 'CREATE_LEARNING_CYCLE', 'learning_cycle', $2, $3, $4)")
            .bind(actor_id)
            .bind(&cycle_id)
            .bind(request_id)
            .bind(json!({
                "candidate_model_version": &candidate_model_version,
                "frozen_evaluation_dataset_version": evaluation_dataset_version.as_deref(),
            }))
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("audit learning cycle creation: {error}"))?;
        tx.commit()
            .await
            .map_err(|error| format!("commit learning cycle: {error}"))?;
        self.learning_cycle_from_id(id).await
    }

    pub async fn get_learning_cycle(&self, cycle_id: &str) -> Result<LearningCycle, String> {
        self.learning_cycle_from_id_lookup(cycle_id).await
    }

    pub async fn add_learning_feedback(
        &self,
        cycle_id: &str,
        request: &LearningFeedbackRequest,
        user_id: &str,
    ) -> Result<LearningFeedback, String> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin learning feedback: {error}"))?;
        let cycle = sqlx::query("SELECT id, cycle_id, state, collect_ends_at, production_model_version FROM learning_cycles WHERE cycle_id = $1 OR id::text = $1 LIMIT 1 FOR UPDATE")
            .bind(cycle_id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(|error| format!("lock learning cycle for feedback: {error}"))?
            .ok_or_else(|| format!("learning cycle {cycle_id} not found"))?;
        let cycle_db_id: i64 = cycle
            .try_get("id")
            .map_err(|error| format!("learning cycle id: {error}"))?;
        let canonical_cycle_id: String = cycle
            .try_get("cycle_id")
            .map_err(|error| format!("learning cycle key: {error}"))?;
        let cycle_state: String = cycle
            .try_get("state")
            .map_err(|error| format!("learning cycle state: {error}"))?;
        if cycle_state != "COLLECT" {
            return Err(format!(
                "learning cycle {canonical_cycle_id} cannot accept feedback in state {cycle_state}"
            ));
        }
        let collect_ends_at: Option<DateTime<Utc>> = cycle
            .try_get("collect_ends_at")
            .map_err(|error| format!("learning collection end: {error}"))?;
        if collect_ends_at
            .map(|ends_at| ends_at <= Utc::now())
            .unwrap_or(true)
        {
            return Err(format!(
                "learning cycle {canonical_cycle_id} collection period has ended"
            ));
        }
        let production_model_version: Option<String> = cycle
            .try_get("production_model_version")
            .map_err(|error| format!("learning production model: {error}"))?;
        let ticket_id = request
            .ticket_id
            .parse::<i64>()
            .map_err(|_| format!("ticket {} not found", request.ticket_id))?;
        let ticket_exists: bool =
            sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM tickets WHERE id = $1)")
                .bind(ticket_id)
                .fetch_one(&mut *tx)
                .await
                .map_err(|error| format!("check feedback ticket: {error}"))?;
        if !ticket_exists {
            return Err(format!("ticket {} not found", request.ticket_id));
        }
        let feedback_type = request
            .feedback_type
            .as_deref()
            .or(request.decision.as_deref())
            .or(request.source.as_deref())
            .unwrap_or("accepted")
            .trim()
            .to_ascii_uppercase();
        let accepted_or_corrected =
            if feedback_type.contains("CORRECT") || feedback_type.contains("REJECT") {
                "CORRECTED"
            } else {
                "ACCEPTED"
            };
        let id: i64 = sqlx::query_scalar("INSERT INTO learning_feedback (cycle_id, ticket_id, production_model_version, production_prediction, operator_confirmed_decision, accepted_or_corrected) VALUES ($1, $2, $3, '{}'::jsonb, $4, $5) RETURNING id")
            .bind(cycle_db_id)
            .bind(ticket_id)
            .bind(production_model_version)
            .bind(json!({"feedback_type": feedback_type, "comment": request.comment, "user_id": user_id}))
            .bind(accepted_or_corrected)
            .fetch_one(&mut *tx)
            .await
            .map_err(|error| format!("insert learning feedback: {error}"))?;
        tx.commit()
            .await
            .map_err(|error| format!("commit learning feedback: {error}"))?;
        self.learning_feedback_from_id(id, &canonical_cycle_id)
            .await
    }

    pub async fn close_learning_cycle(
        &self,
        request: &CloseLearningCycleRequest,
        actor_id: &str,
        request_id: &str,
    ) -> Result<Value, String> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin learning cycle close: {error}"))?;
        let row = if let Some(cycle_id) = request.cycle_id.as_deref() {
            sqlx::query("SELECT id, cycle_id, state, collect_ends_at, evaluation_ends_at, manual_close_enabled, min_feedback_count, candidate_model_version, production_model_version, frozen_evaluation_dataset_version FROM learning_cycles WHERE cycle_id = $1 OR id::text = $1 LIMIT 1 FOR UPDATE")
                .bind(cycle_id)
                .fetch_optional(&mut *tx)
                .await
                .map_err(|error| format!("find learning cycle to close: {error}"))?
                .ok_or_else(|| format!("learning cycle {cycle_id} not found"))?
        } else {
            sqlx::query("SELECT id, cycle_id, state, collect_ends_at, evaluation_ends_at, manual_close_enabled, min_feedback_count, candidate_model_version, production_model_version, frozen_evaluation_dataset_version FROM learning_cycles WHERE (state = 'COLLECT' AND (manual_close_enabled OR collect_ends_at <= now())) OR (state = 'EVALUATE' AND (manual_close_enabled OR evaluation_ends_at <= now())) ORDER BY CASE state WHEN 'COLLECT' THEN collect_ends_at ELSE evaluation_ends_at END, id LIMIT 1 FOR UPDATE SKIP LOCKED")
                .fetch_optional(&mut *tx)
                .await
                .map_err(|error| format!("find eligible learning cycle: {error}"))?
                .ok_or_else(|| "no eligible COLLECT or EVALUATE cycle is available".to_owned())?
        };
        let cycle_db_id: i64 = row
            .try_get("id")
            .map_err(|error| format!("learning cycle id: {error}"))?;
        let cycle_id: String = row
            .try_get("cycle_id")
            .map_err(|error| format!("learning cycle key: {error}"))?;
        let cycle_state: String = row
            .try_get("state")
            .map_err(|error| format!("learning cycle state: {error}"))?;
        if !matches!(cycle_state.as_str(), "COLLECT" | "EVALUATE") {
            return Err(format!(
                "cycle {cycle_id} cannot close from state {cycle_state}"
            ));
        }
        let collect_ends_at: Option<DateTime<Utc>> = row
            .try_get("collect_ends_at")
            .map_err(|error| format!("learning collection end: {error}"))?;
        let manual_close_enabled: bool = row
            .try_get("manual_close_enabled")
            .map_err(|error| format!("learning manual close policy: {error}"))?;
        let evaluation_ends_at: Option<DateTime<Utc>> = row
            .try_get("evaluation_ends_at")
            .map_err(|error| format!("learning evaluation end: {error}"))?;
        let window_ends_at = if cycle_state == "COLLECT" {
            collect_ends_at
        } else {
            evaluation_ends_at
        };
        if !manual_close_enabled
            && window_ends_at
                .map(|ends_at| ends_at > Utc::now())
                .unwrap_or(true)
        {
            return Err(format!(
                "learning cycle {cycle_id} is not eligible to close before its window ends"
            ));
        }
        if cycle_state == "EVALUATE" {
            sqlx::query(
                "UPDATE learning_cycles SET state = 'DECISION', evaluation_ends_at = LEAST(COALESCE(evaluation_ends_at, now()), now()), updated_at = now(), decision_note = 'EVALUATION_WINDOW_CLOSED' WHERE id = $1 AND state = 'EVALUATE'",
            )
            .bind(cycle_db_id)
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("close candidate evaluation window: {error}"))?;
            let job_id: i64 = sqlx::query_scalar(
                "INSERT INTO background_jobs (job_type, payload, state) VALUES ('CANDIDATE_EVALUATION', $1, 'QUEUED') RETURNING id",
            )
            .bind(json!({"kind": "candidate_evaluation", "cycle_id": &cycle_id}))
            .fetch_one(&mut *tx)
                .await
                .map_err(|error| format!("queue candidate evaluation: {error}"))?;
            sqlx::query("INSERT INTO audit_log (actor_id, action, entity_type, entity_id, request_id, reason, metadata) VALUES ($1, 'CLOSE_LEARNING_CYCLE', 'learning_cycle', $2, $3, 'evaluation window closed', $4)")
                .bind(actor_id)
                .bind(&cycle_id)
                .bind(request_id)
                .bind(json!({"state": "DECISION", "job_id": job_id.to_string()}))
                .execute(&mut *tx)
                .await
                .map_err(|error| format!("audit evaluation window close: {error}"))?;
            tx.commit()
                .await
                .map_err(|error| format!("commit evaluation window close: {error}"))?;
            return Ok(json!({
                "job_id": job_id.to_string(),
                "state": "DECISION",
                "cycle": self.learning_cycle_from_id_lookup(&cycle_id).await?,
                "production_model_unchanged": true,
            }));
        }
        let min_feedback_count: i32 = row
            .try_get("min_feedback_count")
            .map_err(|error| format!("learning minimum feedback count: {error}"))?;
        let feedback_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM learning_feedback WHERE cycle_id = $1 AND validation_status = 'VALID'",
        )
        .bind(cycle_db_id)
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("count valid learning feedback: {error}"))?;
        if feedback_count < i64::from(min_feedback_count) {
            sqlx::query("UPDATE learning_cycles SET state = 'INSUFFICIENT_FEEDBACK', updated_at = now(), decision_note = 'INSUFFICIENT_FEEDBACK' WHERE id = $1 AND state = 'COLLECT'")
                .bind(cycle_db_id)
                .execute(&mut *tx)
                .await
                .map_err(|error| format!("close insufficient learning cycle: {error}"))?;
            sqlx::query("INSERT INTO audit_log (actor_id, action, entity_type, entity_id, request_id, reason, metadata) VALUES ($1, 'CLOSE_LEARNING_CYCLE', 'learning_cycle', $2, $3, 'insufficient feedback', $4)")
                .bind(actor_id)
                .bind(&cycle_id)
                .bind(request_id)
                .bind(json!({"state": "INSUFFICIENT_FEEDBACK"}))
                .execute(&mut *tx)
                .await
                .map_err(|error| format!("audit insufficient learning cycle close: {error}"))?;
            tx.commit()
                .await
                .map_err(|error| format!("commit insufficient learning cycle: {error}"))?;
            return Ok(json!({
                "job_id": Value::Null,
                "state": "INSUFFICIENT_FEEDBACK",
                "cycle": self.learning_cycle_from_id_lookup(&cycle_id).await?,
                "production_model_unchanged": true,
            }));
        }
        let candidate_model_version: String = row
            .try_get::<Option<String>, _>("candidate_model_version")
            .map_err(|error| format!("candidate model version: {error}"))?
            .unwrap_or_else(|| "pending".to_owned());
        let production_model_version: Option<String> = row
            .try_get("production_model_version")
            .map_err(|error| format!("production model version: {error}"))?;
        let frozen_evaluation_dataset_version: Option<String> = row
            .try_get("frozen_evaluation_dataset_version")
            .map_err(|error| format!("frozen evaluation dataset version: {error}"))?;
        let feedback_ids: Vec<i64> = sqlx::query_scalar(
            "SELECT COALESCE(array_agg(id ORDER BY id), ARRAY[]::bigint[]) FROM learning_feedback WHERE cycle_id = $1 AND validation_status = 'VALID'",
        )
        .bind(cycle_db_id)
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("list validated learning feedback: {error}"))?;
        let frozen_evaluation_ticket_ids: Vec<i64> = sqlx::query_scalar(
            "SELECT COALESCE(array_agg(ticket_id ORDER BY ticket_id), ARRAY[]::bigint[]) FROM learning_cycle_evaluation_tickets WHERE learning_cycle_id = $1",
        )
        .bind(cycle_db_id)
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("list frozen evaluation ticket IDs: {error}"))?;
        sqlx::query("UPDATE learning_cycles SET state = 'TRAINING', decision_note = 'BUILDING_CANDIDATE_DATASET', updated_at = now() WHERE id = $1 AND state = 'COLLECT'")
            .bind(cycle_db_id)
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("start candidate dataset build: {error}"))?;
        let payload = json!({
            "kind": "build_candidate_dataset",
            "cycle_id": cycle_id,
            "candidate_model_version": candidate_model_version,
            "production_model_version": production_model_version,
            "feedback_ids": feedback_ids.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "frozen_evaluation_dataset_version": frozen_evaluation_dataset_version,
            "frozen_evaluation_ticket_ids": frozen_evaluation_ticket_ids
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
        });
        let job_id: i64 = sqlx::query_scalar("INSERT INTO background_jobs (job_type, payload, state) VALUES ('BUILD_CANDIDATE_DATASET', $1, 'QUEUED') RETURNING id")
            .bind(payload)
            .fetch_one(&mut *tx)
            .await
            .map_err(|error| format!("queue candidate dataset build job: {error}"))?;
        sqlx::query("INSERT INTO audit_log (actor_id, action, entity_type, entity_id, request_id, reason, metadata) VALUES ($1, 'CLOSE_LEARNING_CYCLE', 'learning_cycle', $2, $3, 'candidate dataset build queued', $4)")
            .bind(actor_id)
            .bind(&cycle_id)
            .bind(request_id)
            .bind(json!({"state": "TRAINING", "job_id": job_id.to_string()}))
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("audit candidate dataset build: {error}"))?;
        tx.commit()
            .await
            .map_err(|error| format!("commit candidate dataset build job: {error}"))?;
        Ok(json!({
            "job_id": job_id.to_string(),
            "state": "TRAINING",
            "cycle": self.learning_cycle_from_id_lookup(&cycle_id).await?,
            "production_model_unchanged": true,
        }))
    }

    pub async fn candidate_evaluation(&self) -> Result<Value, String> {
        let cycle = self
            .learning_overview()
            .await?
            .active_cycle
            .ok_or_else(|| "no active learning cycle".to_owned())?;
        let cycle_db_id = cycle
            .id
            .parse::<i64>()
            .map_err(|_| "invalid database learning cycle id".to_owned())?;
        let evaluation_payload = sqlx::query_scalar::<_, Value>(
            "SELECT evaluation_payload FROM model_evaluations WHERE learning_cycle_id = $1 AND model_version = $2 LIMIT 1",
        )
            .bind(cycle_db_id)
            .bind(&cycle.candidate_model_version)
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("fetch candidate evaluation: {error}"))?;
        if let Some(payload) = evaluation_payload
            .filter(|value| value.as_object().is_some_and(|object| !object.is_empty()))
        {
            return Ok(payload);
        }

        let evaluation_job = sqlx::query(
            "SELECT state, error FROM background_jobs WHERE job_type = 'CANDIDATE_EVALUATION' AND payload->>'cycle_id' = $1 ORDER BY id DESC LIMIT 1",
        )
            .bind(&cycle.cycle_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("read candidate evaluation job: {error}"))?;
        let job_state = evaluation_job
            .as_ref()
            .and_then(|row| row.try_get::<String, _>("state").ok());
        let job_failed = job_state.as_deref() == Some("FAILED");
        let job_result_missing =
            cycle.state == "DECISION" && matches!(job_state.as_deref(), None | Some("COMPLETED"));
        let evaluation_failed = job_failed || job_result_missing;
        let status = if evaluation_failed {
            "FAILED"
        } else {
            "PENDING"
        };
        let reason = if cycle.state == "EVALUATE" {
            "SHADOW_WINDOW_OPEN".to_owned()
        } else if job_failed {
            evaluation_job
                .as_ref()
                .and_then(|row| row.try_get::<Option<String>, _>("error").ok().flatten())
                .unwrap_or_else(|| "CANDIDATE_EVALUATION_JOB_FAILED".to_owned())
        } else if job_result_missing && job_state.as_deref() == Some("COMPLETED") {
            "CANDIDATE_EVALUATION_RESULT_MISSING".to_owned()
        } else if job_result_missing {
            "CANDIDATE_EVALUATION_JOB_NOT_QUEUED".to_owned()
        } else if matches!(job_state.as_deref(), Some("QUEUED" | "RUNNING")) {
            "CANDIDATE_EVALUATION_RUNNING".to_owned()
        } else {
            "CANDIDATE_EVALUATION_PENDING".to_owned()
        };
        let now = Utc::now().to_rfc3339();
        let baseline_version = cycle
            .production_model_version
            .as_deref()
            .unwrap_or("unconfigured-production");
        let dataset_version = cycle
            .frozen_evaluation_dataset_version
            .as_deref()
            .unwrap_or("unconfigured-evaluation-dataset");
        let offline_sample_count: i32 = sqlx::query_scalar(
            "SELECT COUNT(*)::int FROM learning_cycle_evaluation_tickets evaluation_ticket JOIN tickets t ON t.id = evaluation_ticket.ticket_id JOIN dataset_ticket_links dataset_ticket ON dataset_ticket.ticket_id = t.id AND dataset_ticket.dataset_version = evaluation_ticket.dataset_version WHERE evaluation_ticket.learning_cycle_id = $1 AND evaluation_ticket.dataset_version = $2 AND NULLIF(BTRIM(t.original_text), '') IS NOT NULL AND NULLIF(BTRIM(t.topic_id), '') IS NOT NULL",
        )
        .bind(cycle_db_id)
        .bind(dataset_version)
        .fetch_one(&self.pool)
        .await
        .map_err(|error| format!("count candidate offline evidence: {error}"))?;
        let policy_thresholds = match cycle.promotion_policy_version.as_str() {
            "policy-v1" => json!({
                "minimum_offline_samples": 30,
                "minimum_shadow_samples": 20,
                "maximum_macro_f1_regression": 0.02,
                "maximum_class_f1_regression": 0.05,
                "maximum_shadow_correction_rate_delta": 0.05,
                "maximum_shadow_inference_failures": 0
            }),
            _ => json!({}),
        };
        let model_evaluation = |evaluation_id: String, model_version: &str, sample_count: i32| {
            json!({
                "schema_version": "model-evaluation.v1",
                "evaluation_id": evaluation_id,
                "model_type": "classifier",
                "model_version": model_version,
                "dataset_version": dataset_version,
                "evaluation_version": "candidate-evaluation.v1",
                "split_version": "frozen-evaluation.v1",
                "evaluation_type": "OFFLINE",
                "status": "INSUFFICIENT_DATA",
                "sample_count": sample_count,
                "metrics": {"reason": reason},
                "critical_regressions": [],
                "created_at": now,
                "evaluator": "pulse109.core.evaluation-status",
                "synthetic": false
            })
        };
        Ok(json!({
            "schema_version": "candidate-evaluation.v1",
            "status": status,
            "cycle_id": cycle.cycle_id,
            "candidate_model_version": cycle.candidate_model_version,
            "production_model_version": baseline_version,
            "candidate_dataset_version": cycle.dataset_version,
            "evaluation_version": "candidate-evaluation.v1",
            "policy_version": cycle.promotion_policy_version,
            "promotion_policy": {
                "version": cycle.promotion_policy_version,
                "thresholds": policy_thresholds
            },
            "offline_evaluation": model_evaluation(
                format!("candidate-offline-pending-{}", cycle_db_id),
                &cycle.candidate_model_version,
                offline_sample_count,
            ),
            "baseline_evaluation": model_evaluation(
                format!("production-offline-pending-{}", cycle_db_id),
                baseline_version,
                if cycle.production_model_version.is_some() { offline_sample_count } else { 0 },
            ),
            "shadow_evaluation": {
                "sample_count": 0,
                "agreement_with_confirmed": Value::Null,
                "correction_rate_delta": Value::Null,
                "critical_regressions": [],
                "metrics": {"linked_decision_count": cycle.shadow_operator_decision_count},
                "blind_ab": if cycle.blind_ab_enabled { "ENABLED" } else { "DISABLED" }
            },
            "gates": [{
                "key": "candidate_evaluation_job",
                "status": if evaluation_failed { "INSUFFICIENT_EVIDENCE" } else { "PENDING" },
                "reason": reason
            }, {
                "key": "offline_sample_count",
                "status": "PENDING",
                "observed": offline_sample_count,
                "threshold": policy_thresholds.get("minimum_offline_samples")
            }, {
                "key": "shadow_sample_count",
                "status": "PENDING",
                "observed": cycle.shadow_operator_decision_count,
                "threshold": policy_thresholds.get("minimum_shadow_samples")
            }],
            "decision": "INSUFFICIENT_EVIDENCE",
            "evaluated_at": now,
            "synthetic": false
        }))
    }

    pub async fn promote_learning_cycle(
        &self,
        cycle_id: &str,
        note: Option<&str>,
        user_id: &str,
    ) -> Result<LearningCycle, String> {
        let cycle = self.learning_cycle_from_id_lookup(cycle_id).await?;
        let cycle_db_id = cycle
            .id
            .parse::<i64>()
            .map_err(|_| "invalid database learning cycle id".to_owned())?;
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin promotion: {error}"))?;
        let locked_cycle = sqlx::query(
            "SELECT cycle_id, state, candidate_model_version, production_model_version, candidate_dataset_version, frozen_evaluation_dataset_version, promotion_policy_version FROM learning_cycles WHERE id = $1 FOR UPDATE",
        )
        .bind(cycle_db_id)
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("lock cycle for promotion: {error}"))?;
        let state: String = locked_cycle
            .try_get("state")
            .map_err(|error| format!("learning cycle state: {error}"))?;
        if state != "DECISION" {
            return Err(format!(
                "cycle {} cannot be promoted from state {}",
                cycle.id, state
            ));
        }
        let candidate_model_version: String = locked_cycle
            .try_get::<Option<String>, _>("candidate_model_version")
            .map_err(|error| format!("candidate model version: {error}"))?
            .ok_or_else(|| "candidate model artifact is not available".to_owned())?;
        let candidate_dataset_version: String = locked_cycle
            .try_get::<Option<String>, _>("candidate_dataset_version")
            .map_err(|error| format!("candidate dataset version: {error}"))?
            .ok_or_else(|| "candidate dataset lineage is not available".to_owned())?;
        let frozen_evaluation_dataset_version: String = locked_cycle
            .try_get::<Option<String>, _>("frozen_evaluation_dataset_version")
            .map_err(|error| format!("frozen evaluation dataset version: {error}"))?
            .ok_or_else(|| "frozen evaluation dataset lineage is not available".to_owned())?;
        let cycle_key: String = locked_cycle
            .try_get("cycle_id")
            .map_err(|error| format!("learning cycle key: {error}"))?;
        let production_model_version: String = locked_cycle
            .try_get::<Option<String>, _>("production_model_version")
            .map_err(|error| format!("production model version: {error}"))?
            .ok_or_else(|| "production model baseline is not available".to_owned())?;
        let policy_version: String = locked_cycle
            .try_get("promotion_policy_version")
            .map_err(|error| format!("promotion policy version: {error}"))?;
        let candidate_dataset_is_synthetic: bool = sqlx::query_scalar(
            "SELECT is_synthetic FROM dataset_versions WHERE dataset_version = $1",
        )
        .bind(&candidate_dataset_version)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|error| format!("read candidate dataset provenance: {error}"))?
        .ok_or_else(|| "candidate dataset lineage is not available".to_owned())?;
        let runtime_mode = env::var("PULSE_ENV").unwrap_or_else(|_| "demo".to_owned());
        if !synthetic_candidate_can_be_promoted(candidate_dataset_is_synthetic, &runtime_mode) {
            return Err(
                "synthetic candidate data cannot be promoted in production runtime".to_owned(),
            );
        }
        let evaluation = sqlx::query_scalar::<_, Value>(
            "SELECT evaluation_payload FROM model_evaluations WHERE learning_cycle_id = $1 AND model_version = $2 LIMIT 1",
        )
        .bind(cycle_db_id)
        .bind(&candidate_model_version)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|error| format!("check candidate evaluation decision: {error}"))?;
        if !evaluation.as_ref().is_some_and(|payload| {
            candidate_evaluation_is_promotable(
                payload,
                &cycle_key,
                &candidate_model_version,
                &production_model_version,
                &candidate_dataset_version,
                &frozen_evaluation_dataset_version,
                &policy_version,
            )
        }) {
            return Err("candidate evaluation is not ready for human promotion".to_owned());
        }
        let candidate_exists: bool = sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM model_versions WHERE model_version = $1 AND status IN ('CANDIDATE', 'SHADOW'))")
            .bind(&candidate_model_version)
            .fetch_one(&mut *tx)
            .await
            .map_err(|error| format!("check candidate model: {error}"))?;
        if !candidate_exists {
            return Err("candidate model artifact is not available".to_owned());
        }
        sqlx::query("UPDATE model_versions SET status = 'ARCHIVED' WHERE status = 'PRODUCTION'")
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("archive production model: {error}"))?;
        sqlx::query("UPDATE model_versions SET status = 'PRODUCTION', promoted_at = now() WHERE model_version = $1")
            .bind(&candidate_model_version)
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("promote candidate: {error}"))?;
        let transition = sqlx::query("UPDATE learning_cycles SET state = 'PROMOTED', decision_note = $2, updated_at = now() WHERE id = $1 AND state = 'DECISION'")
            .bind(cycle_db_id)
            .bind(note.or(Some("promoted")))
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("promote learning cycle: {error}"))?;
        if transition.rows_affected() != 1 {
            return Err("learning cycle changed before promotion".to_owned());
        }
        sqlx::query("INSERT INTO audit_log (actor_id, action, entity_type, entity_id, reason, metadata) VALUES ($1, 'PROMOTE_MODEL', 'learning_cycle', $2, $3, $4)")
            .bind(user_id)
            .bind(cycle.id.clone())
            .bind(note)
            .bind(json!({"candidate_model_version": candidate_model_version}))
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("audit promotion: {error}"))?;
        tx.commit()
            .await
            .map_err(|error| format!("commit promotion: {error}"))?;
        self.learning_cycle_from_id_lookup(cycle_id).await
    }

    pub async fn reject_learning_cycle(
        &self,
        cycle_id: &str,
        note: Option<&str>,
        user_id: &str,
    ) -> Result<LearningCycle, String> {
        let cycle = self.learning_cycle_from_id_lookup(cycle_id).await?;
        let cycle_db_id = cycle
            .id
            .parse::<i64>()
            .map_err(|_| "invalid database learning cycle id".to_owned())?;
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin rejection: {error}"))?;
        let locked_cycle = sqlx::query(
            "SELECT state, candidate_model_version FROM learning_cycles WHERE id = $1 FOR UPDATE",
        )
        .bind(cycle_db_id)
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("lock cycle for rejection: {error}"))?;
        let locked_state: String = locked_cycle
            .try_get("state")
            .map_err(|error| format!("learning cycle state: {error}"))?;
        if !matches!(locked_state.as_str(), "EVALUATE" | "DECISION") {
            return Err(format!(
                "cycle {} cannot be rejected from state {}",
                cycle.id, locked_state
            ));
        }
        let candidate_model_version: Option<String> = locked_cycle
            .try_get("candidate_model_version")
            .map_err(|error| format!("candidate model version: {error}"))?;
        if let Some(candidate_model_version) = candidate_model_version {
            let candidate_transition = sqlx::query(
                "UPDATE model_versions SET status = 'REJECTED' WHERE model_version = $1 AND status IN ('CANDIDATE', 'SHADOW')",
            )
            .bind(candidate_model_version)
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("reject candidate model: {error}"))?;
            if candidate_transition.rows_affected() != 1 {
                return Err("candidate model is no longer rejectable".to_owned());
            }
        }
        let decision_note = note
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or("rejected");
        let transition = sqlx::query("UPDATE learning_cycles SET state = 'REJECTED', decision_note = $2, updated_at = now() WHERE id = $1 AND state = $3")
            .bind(cycle_db_id)
            .bind(decision_note)
            .bind(&locked_state)
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("reject learning cycle: {error}"))?;
        if transition.rows_affected() != 1 {
            return Err("learning cycle changed before rejection".to_owned());
        }
        sqlx::query("INSERT INTO audit_log (actor_id, action, entity_type, entity_id, reason) VALUES ($1, 'REJECT_MODEL', 'learning_cycle', $2, $3)")
            .bind(user_id)
            .bind(cycle.id.clone())
            .bind(decision_note)
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("audit rejection: {error}"))?;
        tx.commit()
            .await
            .map_err(|error| format!("commit rejection: {error}"))?;
        self.learning_cycle_from_id_lookup(cycle_id).await
    }

    pub async fn list_models(&self, query: &ModelQuery) -> Result<Vec<ModelVersion>, String> {
        let mut builder =
            QueryBuilder::<Postgres>::new("SELECT model_version FROM model_versions WHERE TRUE");
        if let Some(status) = query.status.as_deref() {
            builder
                .push(" AND status = ")
                .push_bind(status.trim().to_ascii_uppercase());
        }
        builder.push(" ORDER BY created_at DESC, id DESC");
        let rows = builder
            .build()
            .fetch_all(&self.pool)
            .await
            .map_err(|error| format!("list models: {error}"))?;
        let mut models = Vec::with_capacity(rows.len());
        for row in rows {
            let model_version: String = row
                .try_get("model_version")
                .map_err(|error| format!("model version: {error}"))?;
            models.push(self.model_from_version(&model_version).await?);
        }
        Ok(models)
    }

    pub async fn get_model(&self, model_id: &str) -> Result<ModelVersion, String> {
        self.model_from_version(model_id).await
    }

    pub async fn promote_model(
        &self,
        model_id: &str,
        user_id: &str,
    ) -> Result<ModelVersion, String> {
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS (SELECT 1 FROM model_versions WHERE model_version = $1)",
        )
        .bind(model_id)
        .fetch_one(&self.pool)
        .await
        .map_err(|error| format!("check model: {error}"))?;
        if !exists {
            return Err(format!("model {model_id} not found"));
        }
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin model promotion: {error}"))?;
        let is_learning_candidate: bool = sqlx::query_scalar(
            "SELECT EXISTS (SELECT 1 FROM learning_cycles WHERE candidate_model_version = $1)",
        )
        .bind(model_id)
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("check controlled candidate lineage: {error}"))?;
        if is_learning_candidate {
            return Err(
                "candidate is managed by a learning cycle; use candidate promotion after all policy gates pass"
                    .to_owned(),
            );
        }
        sqlx::query("UPDATE model_versions SET status = 'ARCHIVED' WHERE status = 'PRODUCTION' AND model_version <> $1").bind(model_id).execute(&mut *tx).await.map_err(|error| format!("archive model: {error}"))?;
        sqlx::query("UPDATE model_versions SET status = 'PRODUCTION', promoted_at = now() WHERE model_version = $1").bind(model_id).execute(&mut *tx).await.map_err(|error| format!("promote model: {error}"))?;
        sqlx::query("INSERT INTO audit_log (actor_id, action, entity_type, entity_id, reason) VALUES ($1, 'PROMOTE_MODEL', 'model_version', $2, 'manual promotion')").bind(user_id).bind(model_id).execute(&mut *tx).await.map_err(|error| format!("audit model promotion: {error}"))?;
        tx.commit()
            .await
            .map_err(|error| format!("commit model promotion: {error}"))?;
        self.model_from_version(model_id).await
    }

    async fn learning_cycle_from_id_lookup(&self, value: &str) -> Result<LearningCycle, String> {
        let row = sqlx::query(
            "SELECT id FROM learning_cycles WHERE cycle_id = $1 OR id::text = $1 LIMIT 1",
        )
        .bind(value)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("find learning cycle: {error}"))?
        .ok_or_else(|| format!("learning cycle {value} not found"))?;
        self.learning_cycle_from_id(
            row.try_get("id")
                .map_err(|error| format!("cycle id: {error}"))?,
        )
        .await
    }

    async fn learning_cycle_from_id(&self, id: i64) -> Result<LearningCycle, String> {
        let row = sqlx::query(
            r#"
            SELECT lc.id, lc.cycle_id, lc.state,
                   COALESCE(lc.candidate_dataset_version, 'pending') AS dataset_version,
                   COALESCE(lc.candidate_model_version, 'pending') AS candidate_model_version,
                   lc.collect_started_at, lc.collect_ends_at,
                   lc.evaluation_started_at, lc.evaluation_ends_at,
                   lc.blind_ab_enabled, lc.production_model_version,
                   lc.frozen_evaluation_dataset_version, lc.candidate_dataset_checksum,
                   lc.min_feedback_count, lc.promotion_policy_version,
                   lc.manual_close_enabled, lc.created_at, lc.updated_at, lc.decision_note,
                   (SELECT COUNT(*)::int FROM learning_feedback lf
                    WHERE lf.cycle_id = lc.id AND lf.validation_status = 'VALID') AS feedback_count,
                   (SELECT COUNT(*)::int FROM learning_cycle_shadow_predictions sp
                    WHERE sp.learning_cycle_id = lc.id) AS shadow_prediction_count,
                   (SELECT COUNT(*)::int FROM learning_cycle_shadow_predictions sp
                    WHERE sp.learning_cycle_id = lc.id
                      AND sp.candidate_inference_status = 'FAILED') AS shadow_inference_failures,
                   (SELECT COUNT(*)::int FROM learning_cycle_shadow_predictions sp
                    WHERE sp.learning_cycle_id = lc.id
                      AND sp.operator_decision_id IS NOT NULL) AS shadow_operator_decision_count,
                   COALESCE((SELECT me.metrics_json FROM model_evaluations me
                             WHERE me.learning_cycle_id = lc.id
                               AND me.model_version = lc.candidate_model_version
                             ORDER BY me.created_at DESC LIMIT 1), '{}'::jsonb) AS metrics_json
            FROM learning_cycles lc
            WHERE lc.id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("fetch learning cycle: {error}"))?
        .ok_or_else(|| format!("learning cycle {id} not found"))?;
        let created_at: DateTime<Utc> = row
            .try_get("created_at")
            .map_err(|error| format!("cycle created_at: {error}"))?;
        let updated_at: DateTime<Utc> = row
            .try_get("updated_at")
            .map_err(|error| format!("cycle updated_at: {error}"))?;
        let collect_started_at: Option<DateTime<Utc>> = row
            .try_get("collect_started_at")
            .map_err(|error| format!("cycle collect_started_at: {error}"))?;
        let collect_ends_at: Option<DateTime<Utc>> = row
            .try_get("collect_ends_at")
            .map_err(|error| format!("cycle collect_ends_at: {error}"))?;
        let evaluation_started_at: Option<DateTime<Utc>> = row
            .try_get("evaluation_started_at")
            .map_err(|error| format!("cycle evaluation_started_at: {error}"))?;
        let evaluation_ends_at: Option<DateTime<Utc>> = row
            .try_get("evaluation_ends_at")
            .map_err(|error| format!("cycle evaluation_ends_at: {error}"))?;
        Ok(LearningCycle {
            id: row.try_get::<i64, _>("id").unwrap_or(id).to_string(),
            cycle_id: row.try_get("cycle_id").unwrap_or_default(),
            state: row.try_get("state").unwrap_or_default(),
            dataset_version: row
                .try_get("dataset_version")
                .unwrap_or_else(|_| "unknown".to_owned()),
            candidate_model_version: row
                .try_get("candidate_model_version")
                .unwrap_or_else(|_| "pending".to_owned()),
            collect_started_at: collect_started_at.unwrap_or(created_at).to_rfc3339(),
            collect_ends_at: collect_ends_at.unwrap_or(created_at).to_rfc3339(),
            evaluation_started_at: evaluation_started_at.map(|value| value.to_rfc3339()),
            evaluation_ends_at: evaluation_ends_at.map(|value| value.to_rfc3339()),
            shadow_prediction_count: row
                .try_get::<i32, _>("shadow_prediction_count")
                .unwrap_or(0)
                .max(0) as u32,
            shadow_inference_failures: row
                .try_get::<i32, _>("shadow_inference_failures")
                .unwrap_or(0)
                .max(0) as u32,
            shadow_operator_decision_count: row
                .try_get::<i32, _>("shadow_operator_decision_count")
                .unwrap_or(0)
                .max(0) as u32,
            blind_ab_enabled: row.try_get("blind_ab_enabled").unwrap_or(false),
            production_model_version: row.try_get("production_model_version").unwrap_or(None),
            frozen_evaluation_dataset_version: row
                .try_get("frozen_evaluation_dataset_version")
                .unwrap_or(None),
            candidate_dataset_checksum: row.try_get("candidate_dataset_checksum").unwrap_or(None),
            min_feedback_count: row
                .try_get::<i32, _>("min_feedback_count")
                .unwrap_or(1)
                .max(1) as u32,
            promotion_policy_version: row
                .try_get("promotion_policy_version")
                .unwrap_or_else(|_| "policy-v1".to_owned()),
            manual_close_enabled: row.try_get("manual_close_enabled").unwrap_or(false),
            metrics: learning_metrics_from_value(
                row.try_get("metrics_json").unwrap_or_else(|_| json!({})),
            ),
            feedback_count: row.try_get::<i32, _>("feedback_count").unwrap_or(0).max(0) as u32,
            decision_note: row.try_get("decision_note").unwrap_or(None),
            created_at: created_at.to_rfc3339(),
            updated_at: updated_at.to_rfc3339(),
        })
    }

    async fn learning_feedback_from_id(
        &self,
        id: i64,
        cycle_id: &str,
    ) -> Result<LearningFeedback, String> {
        let row = sqlx::query("SELECT id, ticket_id, accepted_or_corrected, feedback_created_at, operator_confirmed_decision->>'user_id' AS user_id, operator_confirmed_decision->>'comment' AS comment FROM learning_feedback WHERE id = $1")
            .bind(id)
            .fetch_one(&self.pool)
            .await
            .map_err(|error| format!("fetch learning feedback: {error}"))?;
        let created_at: DateTime<Utc> = row
            .try_get("feedback_created_at")
            .map_err(|error| format!("feedback created_at: {error}"))?;
        Ok(LearningFeedback {
            id: id.to_string(),
            ticket_id: row
                .try_get::<i64, _>("ticket_id")
                .unwrap_or_default()
                .to_string(),
            cycle_id: Some(cycle_id.to_owned()),
            feedback_type: row
                .try_get::<String, _>("accepted_or_corrected")
                .unwrap_or_default(),
            comment: row.try_get("comment").unwrap_or(None),
            suggestion: None,
            user_id: row
                .try_get::<Option<String>, _>("user_id")
                .unwrap_or(None)
                .unwrap_or_else(|| "unknown".to_owned()),
            created_at: created_at.to_rfc3339(),
        })
    }

    async fn model_from_version(&self, model_id: &str) -> Result<ModelVersion, String> {
        let row = sqlx::query("SELECT mv.model_version, mv.model_family, COALESCE(mv.dataset_version, 'unknown') AS dataset_version, mv.status, mv.created_at, mv.promoted_at, COALESCE(me.metrics_json, '{}'::jsonb) AS metrics_json FROM model_versions mv LEFT JOIN LATERAL (SELECT metrics_json FROM model_evaluations WHERE model_version = mv.model_version ORDER BY created_at DESC LIMIT 1) me ON TRUE WHERE mv.model_version = $1")
            .bind(model_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("fetch model: {error}"))?
            .ok_or_else(|| format!("model {model_id} not found"))?;
        let created_at: DateTime<Utc> = row
            .try_get("created_at")
            .map_err(|error| format!("model created_at: {error}"))?;
        let promoted_at: Option<DateTime<Utc>> = row.try_get("promoted_at").unwrap_or(None);
        let metrics_json: Value = row.try_get("metrics_json").unwrap_or_else(|_| json!({}));
        let labels = metrics_json
            .get("labels")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(ToOwned::to_owned)
            .collect();
        Ok(ModelVersion {
            id: row
                .try_get("model_version")
                .unwrap_or_else(|_| model_id.to_owned()),
            model_family: row
                .try_get("model_family")
                .unwrap_or_else(|_| "unknown".to_owned()),
            base_model: "not-configured".to_owned(),
            dataset_version: row
                .try_get("dataset_version")
                .unwrap_or_else(|_| "unknown".to_owned()),
            status: row.try_get("status").unwrap_or_default(),
            metrics: learning_metrics_from_value(metrics_json),
            languages: vec!["RU".to_owned(), "KZ".to_owned()],
            labels,
            created_at: created_at.to_rfc3339(),
            promoted_at: promoted_at.map(|value| value.to_rfc3339()),
        })
    }

    pub async fn list_audit_log(&self, limit: i64, offset: i64) -> Result<AuditLogPage, String> {
        let total = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM audit_log")
            .fetch_one(&self.pool)
            .await
            .map_err(|error| format!("count audit log: {error}"))?;
        let rows = sqlx::query(
            "SELECT id, actor_id, action, entity_type, entity_id, request_id, created_at FROM audit_log ORDER BY created_at DESC, id DESC LIMIT $1 OFFSET $2",
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| format!("list audit log: {error}"))?;

        let mut items = Vec::with_capacity(rows.len());
        for row in rows {
            items.push(AuditLogEvent {
                id: row
                    .try_get("id")
                    .map_err(|error| format!("audit id: {error}"))?,
                actor_id: row
                    .try_get("actor_id")
                    .map_err(|error| format!("audit actor: {error}"))?,
                action: row
                    .try_get("action")
                    .map_err(|error| format!("audit action: {error}"))?,
                entity_type: row
                    .try_get("entity_type")
                    .map_err(|error| format!("audit entity type: {error}"))?,
                entity_id: row
                    .try_get("entity_id")
                    .map_err(|error| format!("audit entity id: {error}"))?,
                request_id: safe_audit_request_id(
                    row.try_get("request_id")
                        .map_err(|error| format!("audit request id: {error}"))?,
                ),
                created_at: row
                    .try_get("created_at")
                    .map_err(|error| format!("audit created at: {error}"))?,
            });
        }

        Ok(AuditLogPage {
            items,
            total,
            limit,
            offset,
        })
    }

    pub async fn audit(
        &self,
        actor_id: &str,
        action: &str,
        entity_type: &str,
        entity_id: Option<&str>,
        request_id: Option<&str>,
        reason: Option<&str>,
        metadata: Value,
    ) -> Result<(), String> {
        sqlx::query("INSERT INTO audit_log (actor_id, action, entity_type, entity_id, request_id, reason, metadata) VALUES ($1, $2, $3, $4, $5, $6, $7)")
            .bind(actor_id)
            .bind(action)
            .bind(entity_type)
            .bind(entity_id)
            .bind(request_id)
            .bind(reason)
            .bind(metadata)
            .execute(&self.pool)
            .await
            .map_err(|error| format!("write audit log: {error}"))?;
        Ok(())
    }

    pub async fn taxonomy(&self) -> Result<Value, String> {
        let topics = sqlx::query("SELECT id, COALESCE(name_ru, name_kk, id) AS label FROM topics WHERE active ORDER BY name_ru")
            .fetch_all(&self.pool)
            .await
            .map_err(|error| format!("list topics: {error}"))?
            .into_iter()
            .map(|row| json!({"id": row.try_get::<String, _>("id").unwrap_or_default(), "label": row.try_get::<String, _>("label").unwrap_or_default()}))
            .collect::<Vec<_>>();
        let services = sqlx::query("SELECT id, COALESCE(name_ru, name_kk, id) AS label FROM services WHERE active ORDER BY name_ru")
            .fetch_all(&self.pool)
            .await
            .map_err(|error| format!("list services: {error}"))?
            .into_iter()
            .map(|row| json!({"id": row.try_get::<String, _>("id").unwrap_or_default(), "label": row.try_get::<String, _>("label").unwrap_or_default()}))
            .collect::<Vec<_>>();
        let regions = sqlx::query("SELECT id, COALESCE(name_ru, name_kk, name_en, id) AS label FROM regions WHERE active ORDER BY name_ru")
            .fetch_all(&self.pool)
            .await
            .map_err(|error| format!("list regions: {error}"))?
            .into_iter()
            .map(|row| json!({"id": row.try_get::<String, _>("id").unwrap_or_default(), "label": row.try_get::<String, _>("label").unwrap_or_default()}))
            .collect::<Vec<_>>();
        let statuses = sqlx::query_scalar::<_, String>(
            "SELECT DISTINCT status FROM tickets WHERE status IS NOT NULL AND btrim(status) <> '' ORDER BY status",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|error| format!("list ticket statuses: {error}"))?
        .into_iter()
        .map(|value| json!({"id": value, "label": value}))
        .collect::<Vec<_>>();
        let districts = sqlx::query_scalar::<_, String>(
            "SELECT DISTINCT district FROM tickets WHERE district IS NOT NULL AND btrim(district) <> '' ORDER BY district",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|error| format!("list ticket districts: {error}"))?
        .into_iter()
        .map(|value| json!({"id": value, "label": value}))
        .collect::<Vec<_>>();
        let channels = sqlx::query_scalar::<_, String>(
            "SELECT DISTINCT channel FROM tickets WHERE channel IS NOT NULL AND btrim(channel) <> '' ORDER BY channel",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|error| format!("list ticket channels: {error}"))?
        .into_iter()
        .map(|value| json!({"id": value, "label": value}))
        .collect::<Vec<_>>();
        Ok(
            json!({"topics": topics, "services": services, "regions": regions, "statuses": statuses, "districts": districts, "channels": channels, "source": "postgres"}),
        )
    }

    pub async fn assist_preview(
        &self,
        ticket_id: Option<&str>,
        text: Option<&str>,
        language: Option<&str>,
        region_id: Option<&str>,
        request_id: &str,
        trace_id: &str,
    ) -> Result<AssistPreviewResponse, String> {
        let started = Instant::now();
        let mut stages = Vec::new();
        let mut model_versions = std::collections::BTreeMap::new();
        let mut needs_review = false;
        let (ticket, source, exclude_id, latest_decision, prediction, language_state) =
            if let Some(ticket_id) = ticket_id {
                let detail = self.get_ticket(ticket_id).await?;
                let language_state = assist_language_state(Some(&detail.ticket.language));
                let ticket_id = detail.ticket.id.parse::<i64>().ok();
                let prediction = detail.prediction;
                let has_decision = detail.latest_decision.is_some();
                let classification_is_available = prediction.model_version != "unavailable"
                    && !prediction.topic_id.eq_ignore_ascii_case("unknown");
                let route_is_known = classification_is_available
                    && prediction.recommended_service != "UNKNOWN"
                    && prediction.predicted_priority != "UNKNOWN";
                if classification_is_available {
                    model_versions
                        .insert("classifier".to_owned(), prediction.model_version.clone());
                }
                stages.push(assist_stage(
                    "language",
                    if language_state == "UNKNOWN" {
                        "unknown"
                    } else {
                        "completed"
                    },
                    0.0,
                    None,
                    (language_state == "UNKNOWN").then_some("LANGUAGE_UNKNOWN"),
                ));
                stages.push(assist_stage(
                    "classification",
                    if classification_is_available {
                        "completed"
                    } else {
                        "unavailable"
                    },
                    0.0,
                    classification_is_available.then(|| prediction.model_version.clone()),
                    (!classification_is_available).then_some("ML_CLASSIFIER_UNAVAILABLE"),
                ));
                stages.push(assist_stage(
                    "routing",
                    if route_is_known {
                        "completed"
                    } else {
                        "unknown"
                    },
                    0.0,
                    None,
                    (!route_is_known).then_some("ROUTING_UNAVAILABLE"),
                ));
                stages.push(assist_stage(
                    "priority",
                    if route_is_known {
                        "completed"
                    } else {
                        "unknown"
                    },
                    0.0,
                    None,
                    (!route_is_known).then_some("ROUTING_UNAVAILABLE"),
                ));
                needs_review = !has_decision
                    || !classification_is_available
                    || !route_is_known
                    || prediction.confidence < 0.85
                    || matches!(language_state.as_str(), "MIXED" | "UNKNOWN");
                (
                    detail.ticket,
                    "postgres-ticket+ml+qdrant".to_owned(),
                    ticket_id,
                    detail.latest_decision,
                    prediction,
                    language_state,
                )
            } else {
                let text = text.ok_or_else(|| "text is required".to_owned())?.trim();
                if text.is_empty() {
                    return Err("text must not be empty".to_owned());
                }
                if text.chars().count() > 10_000 {
                    return Err("text must be at most 10000 characters".to_owned());
                }
                let normalized_region = normalize_region_id(region_id.unwrap_or("KZ-ASTANA"));
                let created_at = Utc::now().to_rfc3339();
                let mut ticket = Ticket {
                    id: "preview".to_owned(),
                    external_ref: "preview".to_owned(),
                    text: text.to_owned(),
                    language: "UNKNOWN".to_owned(),
                    region_id: normalized_region.clone(),
                    region_name: normalized_region,
                    topic_id: "unknown".to_owned(),
                    topic_label: "Не определено".to_owned(),
                    priority: "UNKNOWN".to_owned(),
                    status: "preview".to_owned(),
                    source: "preview".to_owned(),
                    created_at: created_at.clone(),
                    closed_at: None,
                    updated_at: created_at.clone(),
                };
                let classification_started = Instant::now();
                let classification = self.classify(text, language, request_id, trace_id).await;
                let classification_latency_ms =
                    classification_started.elapsed().as_secs_f64() * 1000.0;
                let mut language_state = assist_language_state(language);
                let prediction = match classification {
                    Ok(classification) => {
                        language_state =
                            assist_language_state(Some(&classification.prediction.language));
                        model_versions.insert(
                            "classifier".to_owned(),
                            classification.model_version.clone(),
                        );
                        stages.push(assist_stage(
                            "language",
                            if language_state == "UNKNOWN" {
                                "unknown"
                            } else {
                                "completed"
                            },
                            0.0,
                            None,
                            (language_state == "UNKNOWN").then_some("LANGUAGE_UNKNOWN"),
                        ));
                        stages.push(assist_stage(
                            "classification",
                            "completed",
                            classification_latency_ms,
                            Some(classification.model_version.clone()),
                            None,
                        ));
                        needs_review |= classification.prediction.needs_review
                            || matches!(language_state.as_str(), "MIXED" | "UNKNOWN");
                        ticket.language = language_state.clone();
                        ticket.topic_id = normalize_topic_id(&classification.prediction.topic_id);
                        ticket.topic_label = classification.prediction.topic.clone();
                        let routing_started = Instant::now();
                        let routing = self
                            .resolve_routing(
                                &ticket.topic_id,
                                &ticket.region_id,
                                None,
                                None,
                                RuleSource::Manual,
                            )
                            .await;
                        let routing_latency_ms = routing_started.elapsed().as_secs_f64() * 1000.0;
                        match routing {
                            Ok(routing) => {
                                ticket.priority = routing.priority.clone();
                                stages.push(assist_stage(
                                    "routing",
                                    "completed",
                                    routing_latency_ms,
                                    None,
                                    None,
                                ));
                                stages.push(assist_stage("priority", "completed", 0.0, None, None));
                                prediction_for_db(0, &classification, &routing)
                            }
                            Err(_) => {
                                needs_review = true;
                                ticket.priority = "UNKNOWN".to_owned();
                                stages.push(assist_stage(
                                    "routing",
                                    "unavailable",
                                    routing_latency_ms,
                                    None,
                                    Some("ROUTING_UNAVAILABLE"),
                                ));
                                stages.push(assist_stage(
                                    "priority",
                                    "skipped",
                                    0.0,
                                    None,
                                    Some("ROUTING_UNAVAILABLE"),
                                ));
                                let unavailable_routing = RoutingDecision {
                                    service_id: "service_other".to_owned(),
                                    service_name: "UNKNOWN".to_owned(),
                                    service_provenance: RuleProvenance::manual(
                                        "Маршрутизация недоступна; выберите службу вручную",
                                    ),
                                    priority: "UNKNOWN".to_owned(),
                                    priority_provenance: RuleProvenance::manual(
                                        "Приоритет недоступен; выберите его вручную",
                                    ),
                                    reason: "Ручная проверка маршрутизации".to_owned(),
                                };
                                prediction_for_db(0, &classification, &unavailable_routing)
                            }
                        }
                    }
                    Err(_) => {
                        ticket.language = language_state.clone();
                        let language_error = if language.is_none() {
                            Some("LANGUAGE_UNAVAILABLE")
                        } else if language_state == "UNKNOWN" {
                            Some("LANGUAGE_UNKNOWN")
                        } else {
                            None
                        };
                        stages.push(assist_stage(
                            "language",
                            if language_state == "UNKNOWN" {
                                "unknown"
                            } else {
                                "completed"
                            },
                            0.0,
                            None,
                            language_error,
                        ));
                        stages.push(assist_stage(
                            "classification",
                            "unavailable",
                            classification_latency_ms,
                            None,
                            Some("ML_CLASSIFIER_UNAVAILABLE"),
                        ));
                        stages.push(assist_stage(
                            "routing",
                            "skipped",
                            0.0,
                            None,
                            Some("CLASSIFICATION_UNAVAILABLE"),
                        ));
                        stages.push(assist_stage(
                            "priority",
                            "skipped",
                            0.0,
                            None,
                            Some("CLASSIFICATION_UNAVAILABLE"),
                        ));
                        needs_review = true;
                        unavailable_assist_prediction(&ticket)
                    }
                };
                (
                    ticket,
                    "ml+qdrant".to_owned(),
                    None,
                    None,
                    prediction,
                    language_state,
                )
            };
        let retrieval_started = Instant::now();
        let retrieval = self
            .search_assist_candidates(&ticket.text, exclude_id, request_id, trace_id)
            .await;
        let retrieval_latency_ms = retrieval_started.elapsed().as_secs_f64() * 1000.0;
        let relation_topic_id = latest_decision
            .as_ref()
            .map(|decision| decision.confirmed_topic_id.as_str())
            .unwrap_or(&prediction.topic_id);
        let current_created_at = DateTime::parse_from_rfc3339(&ticket.created_at)
            .ok()
            .map(|value| value.with_timezone(&Utc));
        let mut similar = Vec::new();
        match retrieval {
            Ok((active_index, hits)) => {
                model_versions.insert("embedder".to_owned(), active_index.embedder_version.clone());
                stages.push(assist_stage(
                    "retrieval",
                    "completed",
                    retrieval_latency_ms,
                    Some(active_index.embedder_version.clone()),
                    None,
                ));
                let relation_started = Instant::now();
                similar = hits
                    .into_iter()
                    .filter_map(|hit| {
                        let similarity =
                            qdrant_score_to_similarity(hit.score, &active_index.distance_metric)?;
                        related_ticket_candidate(
                            hit.id.to_string(),
                            similarity,
                            relation_topic_id,
                            &ticket.region_id,
                            current_created_at.as_ref(),
                            hit.topic_id.clone(),
                            hit.region_id.clone(),
                            hit.created_at.as_ref(),
                            &active_index.embedder_version,
                            &active_index.distance_metric,
                        )
                    })
                    .collect();
                stages.push(assist_stage(
                    "duplicate_repeat",
                    "completed",
                    relation_started.elapsed().as_secs_f64() * 1000.0,
                    None,
                    None,
                ));
            }
            Err(_) => {
                needs_review = true;
                stages.push(assist_stage(
                    "retrieval",
                    "unavailable",
                    retrieval_latency_ms,
                    None,
                    Some("RETRIEVAL_UNAVAILABLE"),
                ));
                stages.push(assist_stage(
                    "duplicate_repeat",
                    "skipped",
                    0.0,
                    None,
                    Some("RETRIEVAL_UNAVAILABLE"),
                ));
            }
        }
        let duplicate_candidates = similar
            .iter()
            .filter(|item| item.relation == "duplicate")
            .cloned()
            .collect();
        let repeat_candidates = similar
            .iter()
            .filter(|item| item.relation == "repeat")
            .cloned()
            .collect();
        let template_started = Instant::now();
        let response_template = if !response_template_language_supported(&language_state) {
            needs_review = true;
            stages.push(assist_stage(
                "response_template",
                "skipped",
                0.0,
                None,
                Some("LANGUAGE_UNSUPPORTED"),
            ));
            unavailable_response_template(&language_state)
        } else if let Some(decision) = latest_decision.as_ref() {
            if let Some(service_id) = decision.confirmed_service_id.as_deref() {
                match self
                    .response_template(
                        &ticket.language,
                        &decision.confirmed_topic_id,
                        service_id,
                        &ticket.region_name,
                    )
                    .await
                {
                    Ok(template) if template.approved => {
                        stages.push(assist_stage(
                            "response_template",
                            "completed",
                            template_started.elapsed().as_secs_f64() * 1000.0,
                            None,
                            None,
                        ));
                        template
                    }
                    Ok(template) => {
                        needs_review = true;
                        stages.push(assist_stage(
                            "response_template",
                            "manual",
                            template_started.elapsed().as_secs_f64() * 1000.0,
                            None,
                            Some("APPROVED_TEMPLATE_NOT_FOUND"),
                        ));
                        template
                    }
                    Err(_) => {
                        needs_review = true;
                        stages.push(assist_stage(
                            "response_template",
                            "unavailable",
                            template_started.elapsed().as_secs_f64() * 1000.0,
                            None,
                            Some("TEMPLATE_UNAVAILABLE"),
                        ));
                        unavailable_response_template(&language_state)
                    }
                }
            } else {
                needs_review = true;
                stages.push(assist_stage(
                    "response_template",
                    "manual",
                    template_started.elapsed().as_secs_f64() * 1000.0,
                    None,
                    Some("CONFIRMED_SERVICE_REQUIRED"),
                ));
                response_template_for(&ticket.language, &decision.confirmed_topic_label)
            }
        } else {
            needs_review = true;
            stages.push(assist_stage(
                "response_template",
                "manual",
                template_started.elapsed().as_secs_f64() * 1000.0,
                None,
                Some("CONFIRMED_DECISION_REQUIRED"),
            ));
            response_template_for(&ticket.language, &ticket.topic_label)
        };
        let partial = stages
            .iter()
            .any(|stage| matches!(stage.status.as_str(), "unavailable" | "skipped" | "unknown"));
        needs_review |= partial
            || matches!(language_state.as_str(), "MIXED" | "UNKNOWN")
            || latest_decision.is_none()
            || !response_template.approved;
        let actionable_context = if actionable_context_needs_candidates(&prediction) {
            let mut alternative_options = Vec::with_capacity(prediction.alternatives.len());
            let mut rules_available = true;
            for alternative in &prediction.alternatives {
                match self
                    .resolve_routing(
                        &alternative.topic_id,
                        &ticket.region_id,
                        None,
                        None,
                        RuleSource::Manual,
                    )
                    .await
                {
                    Ok(routing) => alternative_options.push(ActionableContextOption {
                        topic_id: alternative.topic_id.clone(),
                        topic_label: alternative.topic_label.clone(),
                        service: routing.service_name,
                        priority: routing.priority,
                    }),
                    Err(_) => {
                        rules_available = false;
                        break;
                    }
                }
            }
            if rules_available {
                actionable_context_for_candidates(&prediction, alternative_options)
            } else {
                actionable_context_manual_review(
                    "Не удалось проверить правила для альтернативных тем; проверьте тему, службу и приоритет вручную.",
                )
            }
        } else {
            actionable_context_for_candidates(&prediction, Vec::new())
        };
        Ok(AssistPreviewResponse {
            response_template,
            ticket,
            prediction,
            actionable_context,
            similar_tickets: similar,
            duplicate_candidates,
            repeat_candidates,
            source,
            orchestration: AssistOrchestration {
                request_id: request_id.to_owned(),
                trace_id: trace_id.to_owned(),
                status: if partial { "partial" } else { "complete" }.to_owned(),
                needs_review,
                language: language_state,
                latency_ms: started.elapsed().as_secs_f64() * 1000.0,
                model_versions,
                stages,
            },
        })
    }

    pub async fn apply_decision(
        &self,
        ticket_id: &str,
        action: &str,
        request: &DecisionRequest,
        user_id: &str,
    ) -> Result<DecisionResponse, String> {
        let ticket = self.fetch_ticket(ticket_id).await?;
        let prediction = self
            .fetch_prediction(
                ticket
                    .id
                    .parse::<i64>()
                    .map_err(|_| "invalid database ticket id".to_owned())?,
            )
            .await?;
        let confirmed_topic = if action == "correct" {
            request
                .topic_id
                .as_deref()
                .ok_or_else(|| "topic_id is required for a correction".to_owned())?
        } else {
            request.topic_id.as_deref().unwrap_or(&prediction.topic_id)
        };
        let confirmed_topic = normalize_topic_id(confirmed_topic);
        let confirmed_topic_label: String =
            sqlx::query_scalar("SELECT COALESCE(name_ru, name_kk, id) FROM topics WHERE id = $1")
                .bind(&confirmed_topic)
                .fetch_optional(&self.pool)
                .await
                .map_err(|error| format!("resolve confirmed topic: {error}"))?
                .ok_or_else(|| format!("unknown topic_id: {confirmed_topic}"))?;
        let routing = self
            .resolve_routing(
                &confirmed_topic,
                &ticket.region_id,
                None,
                None,
                RuleSource::Manual,
            )
            .await?;
        let (service_id, service) = if let Some(service_label) = request
            .service
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            let row = sqlx::query(
                "SELECT id, name_ru FROM services WHERE active AND (id = $1 OR lower(name_ru) = lower($1)) LIMIT 1",
            )
            .bind(service_label.trim())
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("resolve confirmed service: {error}"))?;
            if let Some(row) = row {
                (
                    row.try_get::<String, _>("id")
                        .map_err(|error| format!("confirmed service id: {error}"))?,
                    row.try_get::<String, _>("name_ru")
                        .map_err(|error| format!("confirmed service name: {error}"))?,
                )
            } else {
                ("service_other".to_owned(), service_label.trim().to_owned())
            }
        } else {
            (routing.service_id.clone(), routing.service_name.clone())
        };
        let priority = request
            .priority
            .as_deref()
            .map(normalize_priority)
            .unwrap_or(routing.priority);
        let service_overridden = request
            .service
            .as_deref()
            .is_some_and(|service| !service.trim().is_empty());
        let priority_overridden = request
            .priority
            .as_deref()
            .is_some_and(|priority| !priority.trim().is_empty());
        let service_provenance = RuleProvenance::manual(if service_overridden {
            "Служба переопределена оператором"
        } else {
            "Служба подтверждена оператором"
        });
        let priority_provenance = RuleProvenance::manual(if priority_overridden {
            "Приоритет переопределён оператором"
        } else {
            "Приоритет подтверждён оператором"
        });
        let decision_value = if action == "correct" {
            "CORRECTED"
        } else {
            "CONFIRMED"
        };
        let mut index_tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin Qdrant payload update: {error}"))?;
        let active_index = self.locked_vector_index(&mut index_tx).await?;
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin decision transaction: {error}"))?;
        let (decision_id, decision_created_at): (i64, DateTime<Utc>) = sqlx::query_as(
            "INSERT INTO operator_decisions (ticket_id, user_id, confirmed_topic_id, confirmed_service_id, confirmed_priority, decision, feedback) VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id, created_at",
        )
        .bind(ticket.id.parse::<i64>().map_err(|_| "invalid database ticket id".to_owned())?)
        .bind(user_id)
        .bind(&confirmed_topic)
        .bind(&service_id)
        .bind(&priority)
        .bind(decision_value)
        .bind(json!({
            "note": request.note,
            "service": service,
            "action": action,
            "predicted_topic_id": prediction.topic_id,
            "predicted_service": prediction.recommended_service,
            "predicted_priority": prediction.predicted_priority,
            "model_version": prediction.model_version,
            "confirmed_topic_id": confirmed_topic,
            "confirmed_service": service,
            "confirmed_priority": priority,
            "routing_reason": routing.reason,
            "service_provenance": service_provenance,
            "priority_provenance": priority_provenance,
        }))
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("insert operator decision: {error}"))?;
        let confirmed_payload = json!({
            "action": action,
            "topic_id": confirmed_topic,
            "confirmed_topic_label": confirmed_topic_label,
            "service": service,
            "priority": priority,
            "decision_id": decision_id.to_string(),
            "model_version": prediction.model_version,
            "user_id": user_id,
            "service_provenance": service_provenance,
            "priority_provenance": priority_provenance,
        });
        sqlx::query("UPDATE tickets SET operator_confirmed_decision = $2, needs_review = false, updated_in_pulse_at = now() WHERE id = $1")
            .bind(ticket.id.parse::<i64>().map_err(|_| "invalid database ticket id".to_owned())?)
            .bind(&confirmed_payload)
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("update ticket decision: {error}"))?;
        sqlx::query(
            "UPDATE learning_cycle_shadow_predictions AS shadow SET operator_decision_id = $2 FROM learning_cycles AS cycle WHERE shadow.learning_cycle_id = cycle.id AND cycle.state = 'EVALUATE' AND shadow.ticket_id = $1 AND shadow.operator_decision_id IS NULL",
        )
        .bind(ticket.id.parse::<i64>().map_err(|_| "invalid database ticket id".to_owned())?)
        .bind(decision_id)
        .execute(&mut *tx)
        .await
        .map_err(|error| format!("link shadow prediction to operator decision: {error}"))?;
        let active_collect_cycle = sqlx::query("SELECT id, cycle_id FROM learning_cycles WHERE state = 'COLLECT' AND collect_ends_at > now() ORDER BY collect_started_at DESC, id DESC LIMIT 1 FOR UPDATE")
            .fetch_optional(&mut *tx)
            .await
            .map_err(|error| format!("find active learning cycle for feedback: {error}"))?;
        let learning_feedback_cycle_id = if let Some(cycle) = active_collect_cycle {
            let cycle_db_id: i64 = cycle
                .try_get("id")
                .map_err(|error| format!("learning cycle id: {error}"))?;
            let cycle_id: String = cycle
                .try_get("cycle_id")
                .map_err(|error| format!("learning cycle key: {error}"))?;
            sqlx::query("INSERT INTO learning_feedback (cycle_id, ticket_id, production_model_version, production_prediction, operator_confirmed_decision, accepted_or_corrected) VALUES ($1, $2, $3, $4, $5, $6)")
                .bind(cycle_db_id)
                .bind(ticket.id.parse::<i64>().map_err(|_| "invalid database ticket id".to_owned())?)
                .bind(&prediction.model_version)
                .bind(json!({
                    "topic_id": prediction.topic_id,
                    "confidence": prediction.confidence,
                    "service": prediction.recommended_service,
                    "priority": prediction.predicted_priority,
                    "model_version": prediction.model_version,
                    "alternatives": prediction.alternatives,
                }))
                .bind(&confirmed_payload)
                .bind(if action == "correct" { "CORRECTED" } else { "ACCEPTED" })
                .execute(&mut *tx)
                .await
                .map_err(|error| format!("insert learning feedback: {error}"))?;
            Some(cycle_id)
        } else {
            None
        };
        tx.commit()
            .await
            .map_err(|error| format!("commit decision transaction: {error}"))?;
        // Operator corrections change the deterministic retrieval metadata.
        // Keep the vector itself stable, but update its Qdrant payload so the
        // next similarity decision sees the confirmed topic and region.
        let numeric_ticket_id = ticket
            .id
            .parse::<i64>()
            .map_err(|_| "invalid database ticket id".to_owned())?;
        self.qdrant_update_payload(
            &active_index,
            numeric_ticket_id,
            &confirmed_topic,
            &ticket.region_id,
        )
        .await?;
        index_tx
            .commit()
            .await
            .map_err(|error| format!("commit Qdrant payload update: {error}"))?;
        let decision = OperatorDecision {
            id: format!("decision-{decision_id}"),
            ticket_id: ticket.id.clone(),
            action: action.to_owned(),
            predicted_topic_id: prediction.topic_id.clone(),
            predicted_service: Some(prediction.recommended_service.clone()),
            predicted_priority: Some(prediction.predicted_priority.clone()),
            model_version: Some(prediction.model_version.clone()),
            confirmed_topic_id: confirmed_topic,
            confirmed_topic_label,
            confirmed_service_id: Some(service_id),
            service,
            priority,
            service_provenance,
            priority_provenance,
            note: request.note.clone(),
            user_id: user_id.to_owned(),
            created_at: decision_created_at.to_rfc3339(),
        };
        let ticket = self
            .fetch_ticket_by_id(ticket.id.parse::<i64>().unwrap_or_default())
            .await?;
        Ok(DecisionResponse {
            ticket,
            prediction,
            decision,
            learning_feedback_status: if learning_feedback_cycle_id.is_some() {
                "COLLECTED".to_owned()
            } else {
                "NO_ACTIVE_COLLECT_CYCLE".to_owned()
            },
            learning_feedback_cycle_id,
        })
    }

    pub async fn relation_feedback(
        &self,
        ticket_id: &str,
        related_ticket_id: Option<&str>,
        relation: &str,
        decision: &str,
        user_id: &str,
        suggestion: Option<&RelationSuggestionSnapshot>,
    ) -> Result<(), String> {
        let ticket = self.fetch_ticket(ticket_id).await?;
        let related = match related_ticket_id {
            Some(value) => Some(self.fetch_ticket(value).await?),
            None => None,
        };
        sqlx::query("INSERT INTO relation_feedback (ticket_id, related_ticket_id, relation, decision, user_id, suggestion_score, suggestion_threshold, suggestion_rule_version, suggestion_model_version, suggestion_distance_metric) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)")
            .bind(ticket.id.parse::<i64>().map_err(|_| "invalid ticket id".to_owned())?)
            .bind(related.map(|item| item.id.parse::<i64>().unwrap_or_default()))
            .bind(relation.to_ascii_uppercase())
            .bind(decision.to_ascii_uppercase())
            .bind(user_id)
            .bind(suggestion.map(|value| value.score as f64))
            .bind(suggestion.map(|value| value.threshold as f64))
            .bind(suggestion.map(|value| value.rule_version.as_str()))
            .bind(suggestion.map(|value| value.model_version.as_str()))
            .bind(suggestion.map(|value| value.distance_metric.as_str()))
            .execute(&self.pool)
            .await
            .map_err(|error| format!("insert relation feedback: {error}"))?;
        Ok(())
    }

    pub async fn list_routing_feedback(
        &self,
        ticket_id: &str,
    ) -> Result<Vec<RoutingFeedbackRecord>, String> {
        let ticket = self.fetch_ticket(ticket_id).await?;
        let numeric_ticket_id = ticket
            .id
            .parse::<i64>()
            .map_err(|_| "stored ticket has invalid database id".to_owned())?;
        let rows: Vec<DbRoutingFeedback> = sqlx::query_as(
            "SELECT id, ticket_id, operator_decision_id, original_route_recommendation, operator_confirmed_route, service_feedback, corrected_target_service, actor_user_id, source_system, evaluation_status, created_at FROM routing_feedback WHERE ticket_id = $1 ORDER BY created_at DESC, id DESC",
        )
        .bind(numeric_ticket_id)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| format!("list routing feedback: {error}"))?;
        Ok(rows.into_iter().map(Into::into).collect())
    }

    pub async fn create_routing_feedback(
        &self,
        ticket_id: &str,
        service_feedback: &str,
        corrected_target_service: Option<&str>,
        user_id: &str,
    ) -> Result<RoutingFeedbackRecord, String> {
        let ticket = self.fetch_ticket(ticket_id).await?;
        let numeric_ticket_id = ticket
            .id
            .parse::<i64>()
            .map_err(|_| "stored ticket has invalid database id".to_owned())?;
        let decision = self
            .fetch_latest_decision(numeric_ticket_id)
            .await?
            .ok_or_else(|| "an operator decision is required before service feedback".to_owned())?;
        let original_route_recommendation = decision
            .predicted_service
            .as_deref()
            .filter(|service| known_routing_value(service))
            .ok_or_else(|| {
                "original route recommendation is unavailable for this operator decision".to_owned()
            })?
            .to_owned();
        if !known_routing_value(&decision.service) {
            return Err("operator-confirmed route is unavailable for this ticket".to_owned());
        }

        let corrected_target = if let Some(target) = corrected_target_service {
            let row = sqlx::query(
                "SELECT id, name_ru FROM services WHERE active AND (id = $1 OR lower(name_ru) = lower($1) OR lower(name_kk) = lower($1)) LIMIT 1",
            )
            .bind(target.trim())
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("resolve corrected target service: {error}"))?
            .ok_or_else(|| "unknown corrected target service".to_owned())?;
            let service_id: String = row
                .try_get("id")
                .map_err(|error| format!("corrected target service id: {error}"))?;
            let service_name: String = row
                .try_get("name_ru")
                .map_err(|error| format!("corrected target service name: {error}"))?;
            if decision.confirmed_service_id.as_deref() == Some(service_id.as_str())
                || service_name.eq_ignore_ascii_case(&decision.service)
            {
                return Err(
                    "corrected target service must differ from the operator-confirmed route"
                        .to_owned(),
                );
            }
            Some((service_id, service_name))
        } else {
            None
        };
        let decision_id = decision
            .id
            .strip_prefix("decision-")
            .unwrap_or(&decision.id)
            .parse::<i64>()
            .map_err(|_| "stored operator decision has invalid database id".to_owned())?;
        let row: DbRoutingFeedback = sqlx::query_as(
            "INSERT INTO routing_feedback (ticket_id, operator_decision_id, original_route_recommendation, operator_confirmed_route, service_feedback, corrected_target_service_id, corrected_target_service, actor_user_id, source_system, evaluation_status) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10) RETURNING id, ticket_id, operator_decision_id, original_route_recommendation, operator_confirmed_route, service_feedback, corrected_target_service, actor_user_id, source_system, evaluation_status, created_at",
        )
        .bind(numeric_ticket_id)
        .bind(decision_id)
        .bind(&original_route_recommendation)
        .bind(&decision.service)
        .bind(service_feedback)
        .bind(corrected_target.as_ref().map(|(id, _)| id.as_str()))
        .bind(corrected_target.as_ref().map(|(_, name)| name.as_str()))
        .bind(user_id)
        .bind(ROUTING_FEEDBACK_DEMO_SOURCE_SYSTEM)
        .bind(ROUTING_FEEDBACK_PENDING_STATUS)
        .fetch_one(&self.pool)
        .await
        .map_err(|error| format!("insert routing feedback: {error}"))?;
        Ok(row.into())
    }

    async fn fetch_ticket(&self, ticket_id: &str) -> Result<Ticket, String> {
        if let Ok(id) = ticket_id.parse::<i64>() {
            return self.fetch_ticket_by_id(id).await;
        }
        let row: Option<DbTicket> =
            sqlx::query_as(&format!("{TICKET_SELECT} WHERE t.external_ticket_id = $1"))
                .bind(ticket_id)
                .fetch_optional(&self.pool)
                .await
                .map_err(|error| format!("fetch ticket: {error}"))?;
        row.map(ticket_from_db)
            .ok_or_else(|| format!("ticket {ticket_id} not found"))
    }

    async fn fetch_ticket_by_id(&self, id: i64) -> Result<Ticket, String> {
        let row: Option<DbTicket> = sqlx::query_as(&format!("{TICKET_SELECT} WHERE t.id = $1"))
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("fetch ticket: {error}"))?;
        row.map(ticket_from_db)
            .ok_or_else(|| format!("ticket {id} not found"))
    }

    async fn fetch_prediction(&self, ticket_id: i64) -> Result<Prediction, String> {
        let row: Option<DbPrediction> = sqlx::query_as(&format!(
            "{PREDICTION_SELECT} WHERE p.ticket_id = $1 ORDER BY p.created_at DESC LIMIT 1"
        ))
        .bind(ticket_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("fetch prediction: {error}"))?;
        row.map(prediction_from_db)
            .ok_or_else(|| format!("prediction for {ticket_id} not found"))
    }

    async fn fetch_latest_decision(
        &self,
        ticket_id: i64,
    ) -> Result<Option<OperatorDecision>, String> {
        let row: Option<DbDecision> = sqlx::query_as(
            "SELECT d.id, d.ticket_id, d.decision, COALESCE(d.feedback->>'predicted_topic_id', d.confirmed_topic_id) AS predicted_topic_id, d.confirmed_topic_id, COALESCE(tp.name_ru, tp.name_kk, d.confirmed_topic_id) AS confirmed_topic_label, d.confirmed_service_id, COALESCE(d.confirmed_priority, 'normal') AS confirmed_priority, d.feedback->>'service' AS service, d.feedback, COALESCE(d.user_id, 'unknown') AS user_id, d.feedback->>'note' AS note, d.created_at FROM operator_decisions d LEFT JOIN topics tp ON tp.id = d.confirmed_topic_id WHERE d.ticket_id = $1 ORDER BY d.created_at DESC, d.id DESC LIMIT 1",
        )
        .bind(ticket_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("fetch decision: {error}"))?;
        Ok(row.map(decision_from_db))
    }
}

struct MlClassificationWithModel {
    model_version: String,
    prediction: MlClassification,
}

fn ticket_from_db(row: DbTicket) -> Ticket {
    Ticket {
        id: row.id.to_string(),
        external_ref: row.external_ticket_id,
        text: row.original_text,
        language: row.language.to_ascii_lowercase(),
        region_id: row.region_id,
        region_name: row.region_name,
        topic_id: row.topic_id,
        topic_label: row.topic_label,
        priority: row.priority,
        status: row.status.to_ascii_lowercase(),
        source: row.source_system,
        created_at: row.created_at.to_rfc3339(),
        closed_at: row.closed_at.map(|value| value.to_rfc3339()),
        updated_at: row.updated_at.to_rfc3339(),
    }
}

fn prediction_from_db(row: DbPrediction) -> Prediction {
    let alternatives = row
        .alternatives
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|item| {
            Some(AlternativePrediction {
                topic_id: item["topic_id"].as_str()?.to_owned(),
                topic_label: item["topic"].as_str().unwrap_or("Другое").to_owned(),
                confidence: item["confidence"].as_f64().unwrap_or(0.0) as f32,
            })
        })
        .collect();
    let confidence = row.confidence.unwrap_or(0.0) as f32;
    let routing_reason = row
        .prediction
        .get("routing_reason")
        .and_then(Value::as_str)
        .unwrap_or("Источник маршрутизации не сохранён; требуется ручная проверка")
        .to_owned();
    let service_provenance =
        rule_provenance_from_db(&row.prediction, "service_provenance", &routing_reason);
    let priority_provenance = rule_provenance_from_db(
        &row.prediction,
        "priority_provenance",
        "Источник приоритета не сохранён; требуется ручная проверка",
    );
    Prediction {
        ticket_id: row.ticket_id.to_string(),
        model_version: row.model_version,
        topic_id: row.topic_id.unwrap_or_else(|| "unknown".to_owned()),
        topic_label: row.topic_label,
        confidence,
        confidence_state: persisted_confidence_state(&row.prediction, row.needs_review, confidence),
        recommended_service: row.service_name,
        predicted_priority: row.priority,
        routing_reason,
        service_provenance,
        priority_provenance,
        alternatives,
        created_at: row.created_at.to_rfc3339(),
    }
}

fn routing_lookup_facts(topic_id: &str, region_id: &str) -> Vec<crate::ExplainabilityFact> {
    vec![
        crate::ExplainabilityFact {
            field: crate::ExplainabilityFactField::TopicId,
            value: topic_id.to_owned(),
        },
        crate::ExplainabilityFact {
            field: crate::ExplainabilityFactField::RegionId,
            value: region_id.to_owned(),
        },
    ]
}

fn explainability_facts_from_db(value: &Value) -> Vec<crate::ExplainabilityFact> {
    value
        .get("facts_used")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|fact| {
            let field = match fact.get("field").and_then(Value::as_str)? {
                "topic_id" => crate::ExplainabilityFactField::TopicId,
                "region_id" => crate::ExplainabilityFactField::RegionId,
                _ => return None,
            };
            let value = fact.get("value").and_then(Value::as_str)?.trim();
            if value.is_empty() {
                return None;
            }
            Some(crate::ExplainabilityFact {
                field,
                value: value.to_owned(),
            })
        })
        .collect()
}

fn rule_provenance_from_db(prediction: &Value, key: &str, fallback_reason: &str) -> RuleProvenance {
    let stored = prediction.get(key);
    let source = stored
        .and_then(|value| value.get("source"))
        .and_then(Value::as_str)
        .map(RuleSource::from_db)
        .unwrap_or(RuleSource::Manual);
    let version = stored
        .and_then(|value| value.get("version"))
        .and_then(Value::as_i64)
        .and_then(|value| i32::try_from(value).ok())
        .filter(|value| *value > 0);
    let reason = stored
        .and_then(|value| value.get("reason"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(fallback_reason)
        .to_owned();
    RuleProvenance {
        source,
        version,
        reason,
        facts_used: stored.map(explainability_facts_from_db).unwrap_or_default(),
    }
}

fn persisted_confidence_state(prediction: &Value, needs_review: bool, confidence: f32) -> String {
    const UNCERTAIN_CONFIDENCE_THRESHOLD: f32 = 0.58;

    match prediction
        .get("confidence_state")
        .and_then(Value::as_str)
        .map(str::trim)
        .map(str::to_ascii_uppercase)
        .as_deref()
    {
        Some("CONFIDENT" | "HIGH") => "confident".to_owned(),
        Some("UNCERTAIN" | "MEDIUM") => "uncertain".to_owned(),
        Some("LOW_CONFIDENCE" | "LOW") => "low_confidence".to_owned(),
        _ if !needs_review => "confident".to_owned(),
        _ if confidence >= UNCERTAIN_CONFIDENCE_THRESHOLD => "uncertain".to_owned(),
        _ => "low_confidence".to_owned(),
    }
}

fn prediction_for_db(
    ticket_id: i64,
    classification: &MlClassificationWithModel,
    routing: &RoutingDecision,
) -> Prediction {
    let topic_id = normalize_topic_id(&classification.prediction.topic_id);
    let confidence = classification.prediction.confidence;
    Prediction {
        ticket_id: ticket_id.to_string(),
        model_version: classification.model_version.clone(),
        topic_id,
        topic_label: classification.prediction.topic.clone(),
        confidence,
        confidence_state: classification
            .prediction
            .confidence_state
            .to_ascii_lowercase(),
        recommended_service: routing.service_name.clone(),
        predicted_priority: routing.priority.clone(),
        routing_reason: routing.reason.clone(),
        service_provenance: routing.service_provenance.clone(),
        priority_provenance: routing.priority_provenance.clone(),
        alternatives: classification
            .prediction
            .alternatives
            .iter()
            .map(|item| AlternativePrediction {
                topic_id: normalize_topic_id(&item.topic_id),
                topic_label: item.topic.clone(),
                confidence: item.confidence.max(item.score),
            })
            .collect(),
        created_at: Utc::now().to_rfc3339(),
    }
}

fn alternatives_json(classification: &MlClassification) -> Value {
    Value::Array(
        classification
            .alternatives
            .iter()
            .map(|item| {
                json!({
                    "topic_id": normalize_topic_id(&item.topic_id),
                    "topic": item.topic,
                    "confidence": item.confidence.max(item.score),
                })
            })
            .collect(),
    )
}

fn decision_from_db(row: DbDecision) -> OperatorDecision {
    let confirmed_topic_id = row
        .confirmed_topic_id
        .unwrap_or_else(|| "unknown".to_owned());
    let confirmed_topic_label = row
        .confirmed_topic_label
        .unwrap_or_else(|| confirmed_topic_id.clone());
    OperatorDecision {
        id: format!("decision-{}", row.id),
        ticket_id: row.ticket_id.to_string(),
        action: if row.decision == "CORRECTED" {
            "correct"
        } else {
            "confirm"
        }
        .to_owned(),
        predicted_topic_id: row
            .predicted_topic_id
            .clone()
            .unwrap_or_else(|| "unknown".to_owned()),
        predicted_service: row
            .feedback
            .get("predicted_service")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        predicted_priority: row
            .feedback
            .get("predicted_priority")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        model_version: row
            .feedback
            .get("model_version")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        confirmed_topic_id,
        confirmed_topic_label,
        confirmed_service_id: row.confirmed_service_id,
        service: row.service.unwrap_or_else(|| "Другая служба".to_owned()),
        priority: row.confirmed_priority,
        service_provenance: rule_provenance_from_db(
            &row.feedback,
            "service_provenance",
            "Источник решения не сохранён; значение подтверждено оператором",
        ),
        priority_provenance: rule_provenance_from_db(
            &row.feedback,
            "priority_provenance",
            "Источник решения не сохранён; значение подтверждено оператором",
        ),
        note: row.note,
        user_id: row.user_id,
        created_at: row.created_at.to_rfc3339(),
    }
}

fn response_template_database_id(template_id: &str) -> Result<i64, String> {
    let value = template_id.strip_prefix("template-").unwrap_or(template_id);
    value
        .parse::<i64>()
        .map_err(|_| format!("not found: response template {template_id} not found"))
}

fn response_template_for(language: &str, _topic: &str) -> ResponseTemplate {
    manual_response_template(language)
}

fn unavailable_response_template(language: &str) -> ResponseTemplate {
    let kazakh = language.eq_ignore_ascii_case("KZ") || language.eq_ignore_ascii_case("kk");
    ResponseTemplate {
        id: "unavailable".to_owned(),
        title: if kazakh {
            "Үлгі жоқ"
        } else {
            "Нет шаблона"
        }
        .to_owned(),
        body: if kazakh {
            "Расталған шешімге сәйкес жауап үлгісі жоқ. Жауапты қолмен жазыңыз."
        } else {
            "Для подтверждённого решения нет подходящего шаблона. Составьте ответ вручную."
        }
        .to_owned(),
        language: if kazakh { "kz" } else { "ru" }.to_owned(),
        approved: false,
        source: "UNAVAILABLE".to_owned(),
        template_key: None,
        topic_id: None,
        service_id: None,
        version: None,
    }
}

fn normalize_language(value: &str) -> String {
    match value.trim().to_ascii_uppercase().as_str() {
        "KZ" | "KK" | "KAZ" => "KZ".to_owned(),
        "RU" | "RUS" => "RU".to_owned(),
        "MIXED" => "UNKNOWN".to_owned(),
        _ => "UNKNOWN".to_owned(),
    }
}

fn normalize_priority(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "critical" | "критический" | "критично" => "critical".to_owned(),
        "high" | "высокий" | "высокая" => "high".to_owned(),
        "low" | "низкий" | "низкая" => "low".to_owned(),
        "medium" | "normal" | "средний" | "обычный" => "medium".to_owned(),
        _ => "medium".to_owned(),
    }
}

fn normalize_status(value: &str) -> String {
    match value.trim().to_ascii_uppercase().as_str() {
        "OPEN" | "NEW" => "OPEN".to_owned(),
        "IN_PROGRESS" | "IN PROGRESS" | "WORKING" => "IN_PROGRESS".to_owned(),
        "RESOLVED" => "RESOLVED".to_owned(),
        "CLOSED" => "CLOSED".to_owned(),
        "CANCELLED" | "CANCELED" => "CANCELLED".to_owned(),
        _ => "UNKNOWN".to_owned(),
    }
}

fn json_text(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(|item| {
        item.as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| item.as_i64().map(|number| number.to_string()))
            .or_else(|| item.as_f64().map(|number| number.to_string()))
    })
}

fn json_datetime(value: &Value, key: &str) -> Result<DateTime<Utc>, String> {
    let text = json_text(value, key).ok_or_else(|| format!("{key} is required"))?;
    DateTime::parse_from_rfc3339(&text)
        .map(|parsed| parsed.with_timezone(&Utc))
        .map_err(|error| format!("invalid {key}: {error}"))
}

fn json_datetime_optional(value: &Value, key: &str) -> Result<Option<DateTime<Utc>>, String> {
    match json_text(value, key) {
        Some(text) => DateTime::parse_from_rfc3339(&text)
            .map(|parsed| Some(parsed.with_timezone(&Utc)))
            .map_err(|error| format!("invalid {key}: {error}")),
        None => Ok(None),
    }
}

fn analytics_days(query: &AnalyticsQuery) -> i64 {
    query
        .range
        .as_deref()
        .and_then(|value| {
            let digits = value.trim().trim_end_matches('d');
            digits.parse::<i64>().ok()
        })
        .unwrap_or(30)
        .clamp(1, 366)
}

fn percent_change(current: i64, previous: i64) -> Option<f64> {
    if previous == 0 {
        None
    } else {
        Some(((current - previous) as f64 / previous as f64) * 100.0)
    }
}

fn push_analytics_filters(
    builder: &mut QueryBuilder<'_, Postgres>,
    query: &AnalyticsQuery,
    alias: &str,
) {
    if let Some(region_id) = query.region_id.as_deref() {
        builder
            .push(format!(" AND {alias}.region_id = "))
            .push_bind(normalize_region_id(region_id));
    }
    if let Some(topic_id) = query.topic_id.as_deref() {
        builder
            .push(format!(" AND {alias}.topic_id = "))
            .push_bind(normalize_topic_id(topic_id));
    }
    if let Some(service_id) = query.service_id.as_deref() {
        builder
            .push(format!(" AND {alias}.service_id = "))
            .push_bind(service_id.trim().to_owned());
    }
    if let Some(status) = query.status.as_deref() {
        builder
            .push(format!(" AND {alias}.status = "))
            .push_bind(status.trim().to_ascii_uppercase());
    }
    if let Some(district) = query.district.as_deref() {
        builder
            .push(format!(" AND {alias}.district = "))
            .push_bind(district.trim().to_owned());
    }
    if let Some(channel) = query.channel.as_deref() {
        builder
            .push(format!(" AND {alias}.channel = "))
            .push_bind(channel.trim().to_owned());
    }
}

fn push_drilldown_predicates(
    builder: &mut QueryBuilder<'_, Postgres>,
    query: &AnalyticsDrilldownQuery,
    dimension: &str,
) -> Result<(), String> {
    let days = analytics_days(&query.filters);
    let current_since = Utc::now() - chrono::Duration::days(days);
    builder
        .push(" WHERE t.created_at >= ")
        .push_bind(current_since)
        .push(" AND t.created_at <= now()");
    push_analytics_filters(builder, &query.filters, "t");

    let value = query.value.as_deref().unwrap_or_default().trim();
    match dimension {
        "region" => {
            builder
                .push(" AND t.region_id = ")
                .push_bind(normalize_region_id(value));
        }
        "topic" => {
            builder
                .push(" AND t.topic_id = ")
                .push_bind(normalize_topic_id(value));
        }
        "date" => {
            let date = chrono::NaiveDate::parse_from_str(value, "%Y-%m-%d")
                .map_err(|error| format!("invalid drilldown date: {error}"))?;
            let start = DateTime::<Utc>::from_naive_utc_and_offset(
                date.and_hms_opt(0, 0, 0)
                    .ok_or_else(|| "invalid drilldown date".to_owned())?,
                Utc,
            );
            let end = start + chrono::Duration::days(1);
            builder
                .push(" AND t.created_at >= ")
                .push_bind(start)
                .push(" AND t.created_at < ")
                .push_bind(end);
        }
        "alert" => {
            let alert_id = value
                .parse::<i64>()
                .map_err(|error| format!("invalid alert id: {error}"))?;
            builder
                .push(" AND EXISTS (SELECT 1 FROM alert_ticket_links atl WHERE atl.alert_id = ")
                .push_bind(alert_id)
                .push(" AND atl.ticket_id = t.id)");
        }
        "overview" => match value.to_ascii_lowercase().as_str() {
            "high_priority" => {
                builder.push(" AND lower(COALESCE(t.priority, '')) IN ('high', 'critical')");
            }
            "open" => {
                builder.push(" AND t.status IN ('OPEN', 'IN_PROGRESS', 'TRIAGED')");
            }
            "resolved" => {
                builder.push(" AND t.status IN ('RESOLVED', 'CLOSED')");
            }
            "" | "all" => {}
            other => {
                return Err(format!("unsupported overview drilldown: {other}"));
            }
        },
        _ => return Err(format!("unsupported drilldown dimension: {dimension}")),
    }
    Ok(())
}

fn learning_metrics_from_value(value: Value) -> LearningMetrics {
    LearningMetrics {
        macro_f1: value
            .get("macro_f1")
            .or_else(|| value.get("f1"))
            .and_then(Value::as_f64)
            .unwrap_or(0.0) as f32,
        accuracy: value.get("accuracy").and_then(Value::as_f64).unwrap_or(0.0) as f32,
        evaluated_samples: value
            .get("evaluated_samples")
            .or_else(|| value.get("sample_count"))
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32,
    }
}

fn normalize_topic_id(value: &str) -> String {
    let normalized = value.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "topic-water" | "water" => "water_supply".to_owned(),
        "topic-roads" | "roads" => "roads".to_owned(),
        "topic-health" | "health" => "healthcare".to_owned(),
        "topic-education" | "education" => "education".to_owned(),
        "topic-transport" | "transport" => "public_transport".to_owned(),
        "topic-environment" | "environment" => "environment".to_owned(),
        "topic-safety" | "street_lighting" | "street-lighting" => "street_lighting".to_owned(),
        "topic-utilities" | "utilities" => "electricity".to_owned(),
        "topic-digital" | "digital" => "telecom".to_owned(),
        "topic-housing" | "housing" => "buildings".to_owned(),
        "other" | "topic-other" | "другая тема" => "unknown".to_owned(),
        _ => normalized,
    }
}

fn normalize_region_id(value: &str) -> String {
    let value = value.trim().to_ascii_uppercase();
    let mapped = match value.as_str() {
        "R01" | "KZ-01" | "KZ-ASTANA" => "KZ-ASTANA",
        "R02" | "KZ-02" | "KZ-ALMATY" => "KZ-ALMATY",
        "R03" | "KZ-03" | "KZ-SHYMKENT" => "KZ-SHYMKENT",
        "R04" | "KZ-04" | "KZ-AKMOLA" => "KZ-AKMOLA",
        "R05" | "KZ-05" | "KZ-AKTOBE" => "KZ-AKTOBE",
        "R06" | "KZ-06" | "KZ-ALMATY-REGION" => "KZ-ALMATY-REGION",
        "R07" | "KZ-07" | "KZ-ATYRAU" => "KZ-ATYRAU",
        "R08" | "KZ-08" | "KZ-EAST-KAZAKHSTAN" => "KZ-EAST-KAZAKHSTAN",
        "R09" | "KZ-09" | "KZ-ZHAMBYL" => "KZ-ZHAMBYL",
        "R10" | "KZ-10" | "KZ-WEST-KAZAKHSTAN" => "KZ-WEST-KAZAKHSTAN",
        "R11" | "KZ-11" | "KZ-KARAGANDA" => "KZ-KARAGANDA",
        "R12" | "KZ-12" | "KZ-KOSTANAY" => "KZ-KOSTANAY",
        "R13" | "KZ-13" | "KZ-KYZYLORDA" => "KZ-KYZYLORDA",
        "R14" | "KZ-14" | "KZ-MANGYSTAU" => "KZ-MANGYSTAU",
        "R15" | "KZ-15" | "KZ-PAVLODAR" => "KZ-PAVLODAR",
        "R16" | "KZ-16" | "KZ-NORTH-KAZAKHSTAN" => "KZ-NORTH-KAZAKHSTAN",
        "R17" | "KZ-17" | "KZ-TURKESTAN" => "KZ-TURKESTAN",
        "R18" | "KZ-18" | "KZ-ULYTAU" => "KZ-ULYTAU",
        "R19" | "KZ-19" | "KZ-ABAY" => "KZ-ABAY",
        "R20" | "KZ-20" | "KZ-ZHETISU" => "KZ-ZHETISU",
        _ => value.as_str(),
    };
    mapped.to_owned()
}

impl From<DbTicket> for Ticket {
    fn from(value: DbTicket) -> Self {
        ticket_from_db(value)
    }
}

#[allow(dead_code)]
fn _topic_contract_is_kept_for_docs(_topic: &Topic) {}

#[cfg(test)]
mod assist_preview_tests {
    use super::*;

    #[test]
    fn response_templates_require_a_known_ru_or_kz_language() {
        assert!(response_template_language_supported("RU"));
        assert!(response_template_language_supported("KZ"));
        assert!(!response_template_language_supported("MIXED"));
        assert!(!response_template_language_supported("UNKNOWN"));
        for language in ["MIXED", "UNKNOWN"] {
            let template = unavailable_response_template(language);
            assert_eq!(template.source, "UNAVAILABLE");
            assert!(!template.approved);
        }
    }

    #[tokio::test]
    async fn ml_failure_returns_unknown_manual_preview_with_request_context() {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .acquire_timeout(std::time::Duration::from_millis(200))
            .connect_lazy("postgres://pulse:pulse@127.0.0.1:1/pulse")
            .unwrap();
        let repository = PgRepository {
            pool,
            qdrant_url: "http://127.0.0.1:1".to_owned(),
            ml_service_url: "http://127.0.0.1:1".to_owned(),
            qdrant_collection: "pulse109_test".to_owned(),
            embedding_dimension: DEFAULT_EMBEDDING_DIMENSION,
            embedding_distance: "Cosine".to_owned(),
            embedder_version: DEFAULT_EMBEDDER_VERSION.to_owned(),
            forecast_model_version: DEFAULT_FORECAST_MODEL_VERSION.to_owned(),
            client: Client::new(),
        };

        let preview = repository
            .assist_preview(
                None,
                Some("Проверить освещение на улице"),
                Some("RU"),
                Some("KZ-ASTANA"),
                "assist-request-test",
                "assist-trace-test",
            )
            .await
            .unwrap();

        assert_eq!(preview.orchestration.status, "partial");
        assert!(preview.orchestration.needs_review);
        assert_eq!(preview.orchestration.request_id, "assist-request-test");
        assert_eq!(preview.orchestration.trace_id, "assist-trace-test");
        assert_eq!(preview.orchestration.language, "RU");
        assert_eq!(preview.prediction.topic_id, "unknown");
        assert_eq!(preview.prediction.confidence, 0.0);
        assert_eq!(preview.prediction.recommended_service, "UNKNOWN");
        assert_eq!(preview.prediction.predicted_priority, "UNKNOWN");
        assert_eq!(
            preview.prediction.service_provenance.source,
            RuleSource::Manual
        );
        assert_eq!(
            preview.prediction.priority_provenance.source,
            RuleSource::Manual
        );
        assert_eq!(preview.response_template.source, "MANUAL_REQUIRED");
        assert!(preview.similar_tickets.is_empty());
        assert!(preview.duplicate_candidates.is_empty());
        assert!(preview.repeat_candidates.is_empty());
        assert!(preview.orchestration.stages.iter().any(|stage| {
            stage.name == "classification"
                && stage.error_code.as_deref() == Some("ML_CLASSIFIER_UNAVAILABLE")
        }));
        assert!(preview.orchestration.stages.iter().any(|stage| {
            stage.name == "retrieval"
                && stage.error_code.as_deref() == Some("RETRIEVAL_UNAVAILABLE")
        }));
    }
}

#[cfg(test)]
mod routing_provenance_tests {
    use super::*;

    #[test]
    fn stored_rule_source_and_version_are_preserved() {
        let provenance = rule_provenance_from_db(
            &json!({
                "service_provenance": {
                    "source": "OFFICIAL",
                    "version": 3,
                    "reason": "Утверждённое правило",
                    "facts_used": [
                        {"field": "topic_id", "value": "TOPIC-WATER"},
                        {"field": "region_id", "value": "KZ-ASTANA"}
                    ]
                }
            }),
            "service_provenance",
            "fallback",
        );

        assert_eq!(provenance.source, RuleSource::Official);
        assert_eq!(provenance.version, Some(3));
        assert_eq!(provenance.reason, "Утверждённое правило");
        assert_eq!(
            provenance.facts_used,
            routing_lookup_facts("TOPIC-WATER", "KZ-ASTANA")
        );

        let history = rule_provenance_from_db(
            &json!({
                "priority_provenance": {
                    "source": "LABEL_HISTORY",
                    "version": 2,
                    "reason": "Историческое значение"
                }
            }),
            "priority_provenance",
            "fallback",
        );
        assert_eq!(history.source, RuleSource::LabelHistory);
        assert_eq!(history.version, Some(2));
        assert_eq!(history.reason, "Историческое значение");

        let manual = rule_provenance_from_db(
            &json!({
                "service_provenance": {
                    "source": "MANUAL",
                    "version": 1,
                    "reason": "Ручное правило"
                }
            }),
            "service_provenance",
            "fallback",
        );
        assert_eq!(manual.source, RuleSource::Manual);
        assert_eq!(manual.version, Some(1));
        assert_eq!(manual.reason, "Ручное правило");
    }

    #[test]
    fn legacy_or_unknown_provenance_defaults_to_manual() {
        let legacy = rule_provenance_from_db(&json!({}), "service_provenance", "Проверьте вручную");
        let unknown = rule_provenance_from_db(
            &json!({
                "service_provenance": {
                    "source": "UNVERIFIED_MODEL",
                    "version": 0,
                    "reason": " "
                }
            }),
            "service_provenance",
            "Проверьте вручную",
        );

        assert_eq!(legacy.source, RuleSource::Manual);
        assert_eq!(legacy.version, None);
        assert_eq!(legacy.reason, "Проверьте вручную");
        assert!(legacy.facts_used.is_empty());
        assert_eq!(unknown.source, RuleSource::Manual);
        assert_eq!(unknown.version, None);
        assert_eq!(unknown.reason, "Проверьте вручную");
    }

    #[test]
    fn routing_lookup_facts_are_limited_to_actual_query_inputs() {
        assert_eq!(
            routing_lookup_facts("TOPIC-WATER", "KZ-ASTANA"),
            vec![
                crate::ExplainabilityFact {
                    field: crate::ExplainabilityFactField::TopicId,
                    value: "TOPIC-WATER".to_owned(),
                },
                crate::ExplainabilityFact {
                    field: crate::ExplainabilityFactField::RegionId,
                    value: "KZ-ASTANA".to_owned(),
                },
            ]
        );
    }
}

#[cfg(test)]
mod vector_index_tests {
    use super::*;

    #[test]
    fn qdrant_config_matches_expected_dimension_and_distance() {
        let info = json!({
            "result": {
                "config": {
                    "params": {
                        "vectors": {"size": 768, "distance": "Cosine"}
                    }
                }
            }
        });
        let index = VectorIndexConfig {
            embedder_version: "embedder-v2".to_owned(),
            embedding_dimension: 768,
            distance_metric: "Cosine".to_owned(),
            collection_name: "pulse109_embedder_v2_d768".to_owned(),
            generation: 2,
        };

        assert!(validate_qdrant_vector_config(&info, &index).is_ok());
    }

    #[test]
    fn qdrant_config_rejects_a_mismatched_dimension() {
        let info = json!({
            "result": {
                "config": {
                    "params": {
                        "vectors": {"size": 32, "distance": "Cosine"}
                    }
                }
            }
        });
        let index = VectorIndexConfig {
            embedder_version: "embedder-v2".to_owned(),
            embedding_dimension: 768,
            distance_metric: "Cosine".to_owned(),
            collection_name: "pulse109_embedder_v2_d768".to_owned(),
            generation: 2,
        };

        assert!(validate_qdrant_vector_config(&info, &index).is_err());
    }

    #[test]
    fn qdrant_scores_use_one_similarity_scale_for_supported_metrics() {
        assert_eq!(qdrant_score_to_similarity(0.82, "Cosine"), Some(0.82));
        assert_eq!(qdrant_score_to_similarity(0.82, "Dot"), Some(0.82));
        assert!((qdrant_score_to_similarity(0.2, "Euclid").unwrap() - 0.98).abs() < 0.001);
        assert_eq!(qdrant_score_to_similarity(f32::NAN, "Cosine"), None);
        assert_eq!(qdrant_score_to_similarity(0.5, "Unknown"), None);
    }

    #[test]
    fn ticket_mapping_preserves_the_official_closed_time() {
        let created_at = DateTime::parse_from_rfc3339("2026-09-20T08:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let closed_at = DateTime::parse_from_rfc3339("2026-09-21T18:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let row = DbTicket {
            id: 17,
            external_ticket_id: "CRM-17".to_owned(),
            original_text: "Synthetic road repair request".to_owned(),
            language: "RU".to_owned(),
            region_id: "R01".to_owned(),
            region_name: "Region 1".to_owned(),
            topic_id: "roads".to_owned(),
            topic_label: "Roads".to_owned(),
            priority: "normal".to_owned(),
            status: "CLOSED".to_owned(),
            source_system: "crm".to_owned(),
            created_at,
            closed_at: Some(closed_at.clone()),
            updated_at: closed_at,
        };

        let ticket = ticket_from_db(row);
        assert!(TICKET_SELECT.contains("t.closed_at"));
        assert_eq!(
            ticket.closed_at.as_deref(),
            Some("2026-09-21T18:00:00+00:00")
        );
    }

    #[test]
    fn row_diagnostic_does_not_include_rejected_content() {
        let ImportError::Invalid(diagnostic) = import_row_error(4, "INVALID_VALUE", "region_id")
        else {
            panic!("row validation should produce an invalid import error");
        };

        assert_eq!(diagnostic, "row 4: INVALID_VALUE: region_id");
        assert!(!diagnostic.contains("sensitive-ticket-text"));
    }
}

#[cfg(test)]
mod persisted_confidence_state_tests {
    use super::*;

    #[test]
    fn preserves_persisted_classifier_states_and_reads_legacy_rows() {
        assert_eq!(
            persisted_confidence_state(&json!({"confidence_state": "UNCERTAIN"}), true, 0.71),
            "uncertain"
        );
        assert_eq!(
            persisted_confidence_state(&json!({"confidence_state": "LOW_CONFIDENCE"}), true, 0.42),
            "low_confidence"
        );
        assert_eq!(
            persisted_confidence_state(&json!({"confidence_state": "CONFIDENT"}), false, 0.79),
            "confident"
        );
        assert_eq!(
            persisted_confidence_state(&json!({}), true, 0.65),
            "uncertain"
        );
        assert_eq!(
            persisted_confidence_state(&json!({}), true, 0.42),
            "low_confidence"
        );
        assert_eq!(
            persisted_confidence_state(&json!({}), false, 0.42),
            "confident"
        );
    }
}
