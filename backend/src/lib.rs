//! Pulse 109 Core API.
//!
//! The explicit `PULSE_STORAGE=memory` mode is an in-memory deterministic
//! fixture for tests and local UI work.  Compose and production use the
//! PostgreSQL repository in `pg.rs`, which calls the ML and Qdrant services.

#![recursion_limit = "512"]

use axum::{
    body::Body,
    extract::{Path, Query, Request, State},
    http::{header, HeaderMap, HeaderValue, Method, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{delete, get, post, put},
    Json, Router,
};
use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    convert::Infallible,
    env,
    io::Write,
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, RwLock,
    },
    time::{Duration as StdDuration, Instant},
};
use thiserror::Error;
use tokio::sync::broadcast;
use tokio_stream::{wrappers::BroadcastStream, StreamExt};
use tower_http::cors::CorsLayer;
use tracing::info;

mod anomaly;
mod pg;
pub use anomaly::AlertDetectorConfig;
use pg::{default_qdrant_collection, safe_trace_id, PgRepository};

const SERVICE_NAME: &str = "pulse109-core";
const API_VERSION: &str = "0.1.0";
const DEMO_TIMESTAMP: &str = "2026-09-21T08:00:00Z";
const FORECAST_HISTORY_DAYS: i64 = 366;
const FORECAST_SEASON_LENGTH_DAYS: usize = 7;
pub(crate) const RELATED_CANDIDATE_THRESHOLD: f32 = 0.78;
pub(crate) const DUPLICATE_CANDIDATE_THRESHOLD: f32 = 0.90;
const RELATED_CANDIDATE_RULE_VERSION: &str = "related-ticket-rules.v1";
const MAX_RESPONSE_TEMPLATE_BODY_CHARS: usize = 4_000;
const MAX_RESPONSE_TEMPLATE_IMPORT_ITEMS: usize = 200;
const DEFAULT_LEARNING_CYCLE_DURATION_HOURS: i32 = 168;
const DEFAULT_LEARNING_MIN_FEEDBACK_COUNT: i32 = 1;
const DEFAULT_LEARNING_PROMOTION_POLICY_VERSION: &str = "policy-v1";
const MAX_LEARNING_CYCLE_DURATION_HOURS: i32 = 87_600;
const AUDIT_LOG_DEFAULT_LIMIT: i64 = 50;
const AUDIT_LOG_MAX_LIMIT: i64 = 100;
const AUDIT_LOG_MAX_OFFSET: i64 = 1_000_000;
static LOG_REQUEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn is_active_learning_cycle_state(state: &str) -> bool {
    matches!(state, "COLLECT" | "TRAINING" | "EVALUATE" | "DECISION")
}

fn learning_collect_end_is_due(collect_ends_at: &str, now: DateTime<Utc>) -> bool {
    DateTime::parse_from_rfc3339(collect_ends_at)
        .map(|ends_at| ends_at.with_timezone(&Utc) <= now)
        .unwrap_or(true)
}

/// Runtime configuration.  `dev_auth` is enabled by default for the local
/// deterministic demo: a request without `x-pulse-role` acts as ADMIN, while
/// an explicitly supplied role is always checked.
#[derive(Clone, Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub dev_auth: bool,
    pub storage: String,
    pub learning_cycle_duration_hours: i32,
    pub learning_min_feedback_count: i32,
    pub learning_manual_close_enabled: bool,
    pub learning_promotion_policy_version: String,
    pub learning_evaluation_dataset_version: Option<String>,
    pub alert_detector: AlertDetectorConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_owned(),
            port: 8080,
            dev_auth: true,
            storage: "memory".to_owned(),
            learning_cycle_duration_hours: DEFAULT_LEARNING_CYCLE_DURATION_HOURS,
            learning_min_feedback_count: DEFAULT_LEARNING_MIN_FEEDBACK_COUNT,
            learning_manual_close_enabled: true,
            learning_promotion_policy_version: DEFAULT_LEARNING_PROMOTION_POLICY_VERSION.to_owned(),
            learning_evaluation_dataset_version: None,
            alert_detector: AlertDetectorConfig::default(),
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        let defaults = Self::default();
        let pulse_env = env::var("PULSE_ENV").ok();
        let storage = env::var("PULSE_STORAGE").unwrap_or_else(|_| match pulse_env.as_deref() {
            Some(value) if !matches!(value.to_ascii_lowercase().as_str(), "test" | "unit") => {
                "postgres".to_owned()
            }
            _ => defaults.storage.clone(),
        });
        let default_dev_auth = pulse_env
            .as_deref()
            .map(|value| {
                matches!(
                    value.to_ascii_lowercase().as_str(),
                    "demo" | "test" | "unit"
                )
            })
            .unwrap_or_else(|| !storage.eq_ignore_ascii_case("postgres"));
        let is_demo_environment = pulse_env
            .as_deref()
            .map(|value| {
                matches!(
                    value.to_ascii_lowercase().as_str(),
                    "demo" | "test" | "unit"
                )
            })
            .unwrap_or_else(|| !storage.eq_ignore_ascii_case("postgres"));
        let dev_auth = resolve_dev_auth(
            parse_optional_bool_env("PULSE_DEV_AUTH"),
            default_dev_auth && defaults.dev_auth,
            is_demo_environment,
        )
        .unwrap_or_else(|error| panic!("{error}"));
        Self {
            host: env::var("PULSE_HOST").unwrap_or(defaults.host),
            port: env::var("PULSE_PORT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(defaults.port),
            dev_auth,
            storage,
            learning_cycle_duration_hours: parse_learning_cycle_duration_hours(
                defaults.learning_cycle_duration_hours,
            ),
            learning_min_feedback_count: parse_positive_env(
                "PULSE_LEARNING_MIN_FEEDBACK_COUNT",
                defaults.learning_min_feedback_count,
            ),
            learning_manual_close_enabled: parse_optional_bool_env("PULSE_LEARNING_MANUAL_CLOSE")
                .unwrap_or(is_demo_environment),
            learning_promotion_policy_version: env::var("PULSE_LEARNING_PROMOTION_POLICY_VERSION")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or(defaults.learning_promotion_policy_version),
            learning_evaluation_dataset_version: env::var(
                "PULSE_LEARNING_EVALUATION_DATASET_VERSION",
            )
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty()),
            alert_detector: AlertDetectorConfig::from_env(),
        }
    }
}

fn parse_positive_env<T>(name: &str, default: T) -> T
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

fn parse_learning_cycle_duration_hours(default: i32) -> i32 {
    let value = parse_positive_env("PULSE_LEARNING_CYCLE_DURATION_HOURS", default);
    assert!(
        value <= MAX_LEARNING_CYCLE_DURATION_HOURS,
        "PULSE_LEARNING_CYCLE_DURATION_HOURS must not exceed {MAX_LEARNING_CYCLE_DURATION_HOURS}"
    );
    value
}

fn parse_optional_bool_env(name: &str) -> Option<bool> {
    let value = env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())?;
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" => Some(true),
        "0" | "false" | "no" => Some(false),
        _ => panic!("{name} must be true or false"),
    }
}

fn resolve_dev_auth(
    requested: Option<bool>,
    default_enabled: bool,
    is_demo_environment: bool,
) -> Result<bool, &'static str> {
    let enabled = requested.unwrap_or(default_enabled);
    if enabled && !is_demo_environment {
        return Err("PULSE_DEV_AUTH=true is only allowed in demo, test, or unit environments");
    }
    Ok(enabled)
}

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    store: Arc<RwLock<Store>>,
    pub config: Config,
    repository: Option<Arc<PgRepository>>,
    alert_events: broadcast::Sender<String>,
}

impl AppState {
    pub fn demo() -> Self {
        let (alert_events, _) = broadcast::channel(128);
        Self {
            store: Arc::new(RwLock::new(Store::demo())),
            config: Config::default(),
            repository: None,
            alert_events,
        }
    }

    pub fn from_env() -> Self {
        let config = Config::from_env();
        let (alert_events, _) = broadcast::channel(128);
        let repository = if config.storage.eq_ignore_ascii_case("postgres") {
            let database_url = env::var("DATABASE_URL").unwrap_or_else(|_| {
                "postgres://pulse:pulse_demo_only@127.0.0.1:5432/pulse".to_owned()
            });
            let qdrant_url =
                env::var("QDRANT_URL").unwrap_or_else(|_| "http://127.0.0.1:6333".to_owned());
            let ml_service_url =
                env::var("ML_SERVICE_URL").unwrap_or_else(|_| "http://127.0.0.1:8000".to_owned());
            let embedding_dimension = env::var("EMBEDDING_DIMENSION")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(32);
            let embedder_version = env::var("EMBEDDER_VERSION")
                .unwrap_or_else(|_| "embedder-demo-2026-09-21-001".to_owned());
            let qdrant_collection = env::var("QDRANT_COLLECTION")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| {
                    default_qdrant_collection(&embedder_version, embedding_dimension)
                });
            Some(Arc::new(
                PgRepository::connect_lazy_with_options(
                    &database_url,
                    qdrant_url,
                    ml_service_url,
                    qdrant_collection,
                    embedding_dimension,
                    embedder_version,
                )
                .unwrap_or_else(|error| panic!("invalid PostgreSQL configuration: {error}")),
            ))
        } else {
            None
        };
        Self {
            store: Arc::new(RwLock::new(Store::demo())),
            config,
            repository,
            alert_events,
        }
    }

    pub async fn initialize(&self) -> Result<(), String> {
        if let Some(repository) = &self.repository {
            repository.initialize().await?;
        }
        Ok(())
    }

    fn repository(&self) -> Option<Arc<PgRepository>> {
        self.repository.clone()
    }

    fn publish_alerts_changed(&self) {
        // A missing subscriber is normal; the next subscriber receives a fresh snapshot.
        if self.alert_events.receiver_count() > 0 {
            let _ = self
                .alert_events
                .send(r#"{"type":"alerts.changed"}"#.to_owned());
        }
    }

    fn read_store(&self) -> Result<std::sync::RwLockReadGuard<'_, Store>, ApiError> {
        self.store
            .read()
            .map_err(|_| ApiError::Internal("store lock poisoned".to_owned()))
    }

    fn write_store(&self) -> Result<std::sync::RwLockWriteGuard<'_, Store>, ApiError> {
        self.store
            .write()
            .map_err(|_| ApiError::Internal("store lock poisoned".to_owned()))
    }
}

#[derive(Clone, Debug)]
struct Store {
    regions: Vec<Region>,
    topics: Vec<Topic>,
    response_templates: BTreeMap<String, ResponseTemplateRecord>,
    tickets: BTreeMap<String, Ticket>,
    predictions: BTreeMap<String, Prediction>,
    decisions: Vec<OperatorDecision>,
    alerts: BTreeMap<String, Alert>,
    learning_cycles: BTreeMap<String, LearningCycle>,
    learning_feedback: Vec<LearningFeedback>,
    models: BTreeMap<String, ModelVersion>,
    next_ticket_number: u64,
    next_decision_number: u64,
    next_cycle_number: u64,
    next_feedback_number: u64,
    next_response_template_number: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Region {
    pub id: String,
    pub name: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Topic {
    pub id: String,
    pub label: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Ticket {
    pub id: String,
    pub external_ref: String,
    pub text: String,
    pub language: String,
    pub region_id: String,
    pub region_name: String,
    pub topic_id: String,
    pub topic_label: String,
    pub priority: String,
    pub status: String,
    pub source: String,
    pub created_at: String,
    pub closed_at: Option<String>,
    pub updated_at: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct AlternativePrediction {
    pub topic_id: String,
    pub topic_label: String,
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RuleSource {
    Official,
    LabelHistory,
    Manual,
}

impl RuleSource {
    pub fn from_db(value: &str) -> Self {
        match value.trim().to_ascii_uppercase().as_str() {
            "OFFICIAL" => Self::Official,
            "LABEL_HISTORY" => Self::LabelHistory,
            _ => Self::Manual,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RuleProvenance {
    pub source: RuleSource,
    pub version: Option<i32>,
    pub reason: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub facts_used: Vec<ExplainabilityFact>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExplainabilityFactField {
    TopicId,
    RegionId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExplainabilityFact {
    pub field: ExplainabilityFactField,
    pub value: String,
}

impl RuleProvenance {
    pub fn manual(reason: impl Into<String>) -> Self {
        Self {
            source: RuleSource::Manual,
            version: None,
            reason: reason.into(),
            facts_used: Vec::new(),
        }
    }

    pub fn with_fact(mut self, field: ExplainabilityFactField, value: impl Into<String>) -> Self {
        let value = value.into();
        let value = value.trim();
        if !value.is_empty() {
            self.facts_used.push(ExplainabilityFact {
                field,
                value: value.to_owned(),
            });
        }
        self
    }

    pub fn with_facts(mut self, facts_used: Vec<ExplainabilityFact>) -> Self {
        self.facts_used = facts_used
            .into_iter()
            .filter(|fact| !fact.value.trim().is_empty())
            .collect();
        self
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Prediction {
    pub ticket_id: String,
    pub model_version: String,
    pub topic_id: String,
    pub topic_label: String,
    pub confidence: f32,
    pub confidence_state: String,
    pub recommended_service: String,
    pub predicted_priority: String,
    pub routing_reason: String,
    pub service_provenance: RuleProvenance,
    pub priority_provenance: RuleProvenance,
    pub alternatives: Vec<AlternativePrediction>,
    pub created_at: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct OperatorDecision {
    pub id: String,
    pub ticket_id: String,
    pub action: String,
    pub predicted_topic_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted_service: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted_priority: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    pub confirmed_topic_id: String,
    pub confirmed_topic_label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirmed_service_id: Option<String>,
    pub service: String,
    pub priority: String,
    pub service_provenance: RuleProvenance,
    pub priority_provenance: RuleProvenance,
    pub note: Option<String>,
    pub user_id: String,
    pub created_at: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct SimilarTicket {
    pub ticket_id: String,
    pub score: f32,
    pub relation: String,
    pub topic_id: String,
    pub region_id: String,
    pub created_at: Option<String>,
    pub matched_factors: Vec<String>,
    pub suggestion: RelationSuggestionSnapshot,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RelationSuggestionSnapshot {
    pub score: f32,
    pub threshold: f32,
    pub rule_version: String,
    pub model_version: String,
    pub distance_metric: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseTemplate {
    pub id: String,
    pub title: String,
    pub body: String,
    pub language: String,
    pub approved: bool,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topic_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<i32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseTemplateInput {
    pub template_key: String,
    pub language: String,
    pub topic_id: String,
    pub service_id: String,
    pub body: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportResponseTemplatesRequest {
    pub items: Vec<ResponseTemplateInput>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UpdateResponseTemplateRequest {
    pub body: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseTemplateRecord {
    pub id: String,
    pub template_key: String,
    pub language: String,
    pub topic_id: Option<String>,
    pub service_id: Option<String>,
    pub body: String,
    pub approved: bool,
    pub version: i32,
    pub created_by: Option<String>,
    pub updated_by: Option<String>,
    pub approved_by: Option<String>,
    pub approved_at: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseTemplatesResponse {
    pub items: Vec<ResponseTemplateRecord>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Alert {
    pub id: String,
    pub incident_key: String,
    pub alert_type: String,
    pub severity: String,
    pub status: String,
    pub title: String,
    pub description: String,
    pub region_id: String,
    pub topic_id: String,
    pub ticket_count: u32,
    pub period_start: Option<String>,
    pub period_end: Option<String>,
    pub current_count: u32,
    pub baseline: Option<f64>,
    pub deviation: Option<f64>,
    pub robust_z: Option<f64>,
    pub ratio: Option<f64>,
    pub detector_version: Option<String>,
    pub linked_ticket_ids: Vec<String>,
    pub created_at: String,
    pub detected_at: String,
    pub acknowledged_by: Option<String>,
    pub acknowledged_at: Option<String>,
    pub closed_by: Option<String>,
    pub closed_at: Option<String>,
    pub detail: Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct LearningMetrics {
    pub macro_f1: f32,
    pub accuracy: f32,
    pub evaluated_samples: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct LearningCycle {
    pub id: String,
    pub cycle_id: String,
    pub state: String,
    pub dataset_version: String,
    pub candidate_model_version: String,
    pub collect_started_at: String,
    pub collect_ends_at: String,
    pub evaluation_started_at: Option<String>,
    pub evaluation_ends_at: Option<String>,
    pub shadow_prediction_count: u32,
    pub shadow_inference_failures: u32,
    pub shadow_operator_decision_count: u32,
    pub blind_ab_enabled: bool,
    pub production_model_version: Option<String>,
    pub frozen_evaluation_dataset_version: Option<String>,
    pub candidate_dataset_checksum: Option<String>,
    pub min_feedback_count: u32,
    pub promotion_policy_version: String,
    pub manual_close_enabled: bool,
    pub metrics: LearningMetrics,
    pub feedback_count: u32,
    pub decision_note: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct LearningFeedback {
    pub id: String,
    pub ticket_id: String,
    pub cycle_id: Option<String>,
    pub feedback_type: String,
    pub comment: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<RelationSuggestionSnapshot>,
    pub user_id: String,
    pub created_at: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelVersion {
    pub id: String,
    pub model_family: String,
    pub base_model: String,
    pub dataset_version: String,
    pub status: String,
    pub metrics: LearningMetrics,
    pub languages: Vec<String>,
    pub labels: Vec<String>,
    pub created_at: String,
    pub promoted_at: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Role {
    Operator,
    Manager,
    Admin,
    MlReviewer,
}

impl Role {
    fn parse(value: &str) -> Option<Self> {
        match value.to_ascii_uppercase().as_str() {
            "OPERATOR" => Some(Self::Operator),
            "MANAGER" => Some(Self::Manager),
            "ADMIN" => Some(Self::Admin),
            "ML_REVIEWER" | "ML-REVIEWER" => Some(Self::MlReviewer),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Operator => "OPERATOR",
            Self::Manager => "MANAGER",
            Self::Admin => "ADMIN",
            Self::MlReviewer => "ML_REVIEWER",
        }
    }
}

#[derive(Clone, Debug)]
struct Actor {
    user_id: String,
    role: Role,
}

fn actor(headers: &HeaderMap, config: &Config) -> Result<Actor, ApiError> {
    let role_header_name = if config.dev_auth {
        "x-pulse-role"
    } else {
        "x-authenticated-role"
    };
    let user_header_name = if config.dev_auth {
        "x-user-id"
    } else {
        "x-authenticated-user"
    };
    let role_header = headers
        .get(role_header_name)
        .and_then(|value| value.to_str().ok());
    let user_id = headers
        .get(user_header_name)
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(if config.dev_auth {
            "demo-admin"
        } else {
            "unknown"
        })
        .to_owned();

    match role_header {
        Some(value) => Role::parse(value)
            .map(|role| Actor { user_id, role })
            .ok_or_else(|| ApiError::Unauthorized("invalid x-pulse-role".to_owned())),
        None if config.dev_auth => Ok(Actor {
            user_id,
            role: Role::Admin,
        }),
        None => Err(ApiError::Unauthorized(
            "trusted auth gateway headers are required outside development mode".to_owned(),
        )),
    }
}

fn require_role(headers: &HeaderMap, config: &Config, allowed: &[Role]) -> Result<Actor, ApiError> {
    let current = actor(headers, config)?;
    if allowed.contains(&current.role) {
        Ok(current)
    } else {
        Err(ApiError::Forbidden(format!(
            "role {} is not allowed for this operation",
            current.role.as_str()
        )))
    }
}

fn request_id_from_headers(headers: &HeaderMap) -> String {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("core-request")
        .to_owned()
}

fn log_request_id_from_headers(headers: &HeaderMap) -> String {
    headers
        .get("x-pulse-log-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("n/a")
        .to_owned()
}

fn trace_id_from_headers(headers: &HeaderMap) -> String {
    headers
        .get("x-trace-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| request_id_from_headers(headers))
}

fn log_trace_id_from_headers(headers: &HeaderMap) -> String {
    safe_trace_id(&trace_id_from_headers(headers))
}

fn response_error_code(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 => "BAD_REQUEST",
        401 => "UNAUTHORIZED",
        403 => "FORBIDDEN",
        404 => "NOT_FOUND",
        409 => "CONFLICT",
        422 => "VALIDATION_ERROR",
        429 => "RATE_LIMITED",
        503 => "NOT_READY",
        400..=499 => "CLIENT_ERROR",
        500..=599 => "SERVER_ERROR",
        _ => "none",
    }
}

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("unauthorized: {0}")]
    Unauthorized(String),
    #[error("forbidden: {0}")]
    Forbidden(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("internal server error: {0}")]
    Internal(String),
    #[error("service unavailable: {0}")]
    Unavailable(String),
}

impl ApiError {
    fn public_message(&self) -> &'static str {
        match self {
            Self::BadRequest(_) => "invalid request",
            Self::Unauthorized(_) => "authentication required",
            Self::Forbidden(_) => "role does not allow this action",
            Self::NotFound(_) => "resource not found",
            Self::Conflict(_) => "request conflicts with current resource state",
            Self::Internal(_) => "internal server error",
            Self::Unavailable(_) => "service is not ready",
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::BadRequest(_) => "BAD_REQUEST",
            Self::Unauthorized(_) => "UNAUTHORIZED",
            Self::Forbidden(_) => "FORBIDDEN",
            Self::NotFound(_) => "NOT_FOUND",
            Self::Conflict(_) => "CONFLICT",
            Self::Internal(_) => "INTERNAL_ERROR",
            Self::Unavailable(_) => "NOT_READY",
        }
    }

    fn status(&self) -> StatusCode {
        match self {
            Self::BadRequest(_) => StatusCode::BAD_REQUEST,
            Self::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            Self::Forbidden(_) => StatusCode::FORBIDDEN,
            Self::NotFound(_) => StatusCode::NOT_FOUND,
            Self::Conflict(_) => StatusCode::CONFLICT,
            Self::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
            Self::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status();
        let body = Json(json!({
            "error": {
                "code": self.code(),
                "message": self.public_message(),
            }
        }));
        (status, body).into_response()
    }
}

fn demo_regions() -> Vec<Region> {
    [
        ("R01", "Астана"),
        ("R02", "Алматы"),
        ("R03", "Шымкент"),
        ("R04", "Акмолинская область"),
        ("R05", "Актюбинская область"),
        ("R06", "Алматинская область"),
        ("R07", "Атырауская область"),
        ("R08", "Восточно-Казахстанская область"),
        ("R09", "Жамбылская область"),
        ("R10", "Западно-Казахстанская область"),
        ("R11", "Карагандинская область"),
        ("R12", "Костанайская область"),
        ("R13", "Кызылординская область"),
        ("R14", "Мангистауская область"),
        ("R15", "Павлодарская область"),
        ("R16", "Северо-Казахстанская область"),
        ("R17", "Туркестанская область"),
        ("R18", "Улытауская область"),
        ("R19", "Абайская область"),
        ("R20", "Жетысуская область"),
    ]
    .into_iter()
    .map(|(id, name)| Region {
        id: id.to_owned(),
        name: name.to_owned(),
    })
    .collect()
}

fn demo_topics() -> Vec<Topic> {
    [
        ("TOPIC-WATER", "Водоснабжение"),
        ("TOPIC-ROADS", "Дороги и благоустройство"),
        ("TOPIC-HEALTH", "Здравоохранение"),
        ("TOPIC-SOCIAL", "Социальная поддержка"),
        ("TOPIC-EDUCATION", "Образование"),
        ("TOPIC-UTILITIES", "Коммунальные услуги"),
        ("TOPIC-SAFETY", "Безопасность"),
        ("TOPIC-TRANSPORT", "Общественный транспорт"),
        ("TOPIC-ENVIRONMENT", "Экология"),
        ("TOPIC-DIGITAL", "Государственные сервисы"),
        ("TOPIC-HOUSING", "Жильё"),
        ("TOPIC-OTHER", "Другое"),
    ]
    .into_iter()
    .map(|(id, label)| Topic {
        id: id.to_owned(),
        label: label.to_owned(),
    })
    .collect()
}

// Explicit offline fixture behavior for `PULSE_STORAGE=memory` tests only.
// The production PostgreSQL path delegates classification to ML Service.
fn demo_topic_for_text(text: &str) -> (&'static str, f32, &'static str, &'static str) {
    let value = text.to_ascii_lowercase();
    if value.contains("вод") || value.contains("су ") || value.contains("суару") {
        ("TOPIC-WATER", 0.96, "Водоканал", "high")
    } else if value.contains("фонар")
        || value.contains("освещ")
        || value.contains("жарық")
        || value.contains("шамы")
    {
        ("TOPIC-SAFETY", 0.92, "Служба городского освещения", "high")
    } else if value.contains("электр")
        || value.contains("свет")
        || value.contains("тоқ")
        || value.contains("электр қуаты")
    {
        ("TOPIC-UTILITIES", 0.91, "Электросети", "high")
    } else if value.contains("дорог") || value.contains("шұңқыр") || value.contains("жол")
    {
        ("TOPIC-ROADS", 0.91, "Городская инфраструктура", "high")
    } else if value.contains("врач") || value.contains("емхана") || value.contains("больниц")
    {
        ("TOPIC-HEALTH", 0.89, "Управление здравоохранения", "medium")
    } else if value.contains("школ") || value.contains("мектеп") {
        ("TOPIC-EDUCATION", 0.88, "Управление образования", "medium")
    } else if value.contains("автобус") || value.contains("көлік") || value.contains("маршрут")
    {
        ("TOPIC-TRANSPORT", 0.87, "Управление транспорта", "medium")
    } else if value.contains("қоқыс") || value.contains("мусор") || value.contains("контейнер")
    {
        ("TOPIC-ENVIRONMENT", 0.84, "Управление экологии", "low")
    } else if value.contains("эколог") || value.contains("выброс") || value.contains("загряз")
    {
        ("TOPIC-ENVIRONMENT", 0.84, "Управление экологии", "medium")
    } else if value.contains("пособ")
        || value.contains("выплат")
        || value.contains("әлеуметтік көмек")
    {
        ("TOPIC-SOCIAL", 0.86, "Центр социальной защиты", "medium")
    } else if value.contains("портал")
        || value.contains("приложен")
        || value.contains("онлайн")
        || value.contains("интернет")
    {
        ("TOPIC-DIGITAL", 0.82, "Цифровой акимат", "medium")
    } else if value.contains("отоплен") || value.contains("батаре") || value.contains("жылу")
    {
        (
            "TOPIC-UTILITIES",
            0.88,
            "Городские коммунальные службы",
            "high",
        )
    } else if value.contains("газ") {
        ("TOPIC-UTILITIES", 0.88, "Газовая служба", "high")
    } else {
        ("TOPIC-OTHER", 0.61, "Единый контакт-центр", "low")
    }
}

fn normalize_region_id(value: String) -> String {
    let trimmed = value.trim().to_owned();
    if let Some(suffix) = trimmed.strip_prefix("KZ-") {
        if let Ok(number) = suffix.parse::<u8>() {
            if (1..=20).contains(&number) {
                return format!("R{number:02}");
            }
        }
    }
    trimmed
}

fn detect_language(text: &str) -> &'static str {
    if text
        .chars()
        .any(|ch| matches!(ch, 'ә' | 'ғ' | 'қ' | 'ң' | 'ө' | 'ұ' | 'ү' | 'һ' | 'і'))
    {
        "kk"
    } else {
        "ru"
    }
}

fn detect_preview_language(text: &str) -> &'static str {
    let mut has_cyrillic = false;
    let mut has_kazakh = false;
    let mut has_russian_specific = false;
    for character in text.chars() {
        let lower = character.to_lowercase().next().unwrap_or(character);
        if ('а'..='я').contains(&lower) || lower == 'ё' {
            has_cyrillic = true;
        }
        if matches!(lower, 'ә' | 'ғ' | 'қ' | 'ң' | 'ө' | 'ұ' | 'ү' | 'һ' | 'і') {
            has_cyrillic = true;
            has_kazakh = true;
        }
        if matches!(lower, 'ё' | 'э' | 'ъ') {
            has_russian_specific = true;
        }
    }
    if !has_cyrillic {
        "UNKNOWN"
    } else if has_kazakh && has_russian_specific {
        "MIXED"
    } else if has_kazakh {
        "KZ"
    } else {
        "RU"
    }
}

fn normalize_preview_language(language: Option<&str>, text: &str) -> Result<String, ApiError> {
    let Some(language) = language.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(detect_preview_language(text).to_owned());
    };
    match language.to_ascii_uppercase().as_str() {
        "RU" | "RUS" => Ok("RU".to_owned()),
        "KZ" | "KK" | "KAZ" => Ok("KZ".to_owned()),
        "MIXED" => Ok("MIXED".to_owned()),
        "UNKNOWN" => Ok("UNKNOWN".to_owned()),
        _ => Err(ApiError::BadRequest(
            "language must be RU, KZ, MIXED, or UNKNOWN".to_owned(),
        )),
    }
}

fn unavailable_demo_prediction(ticket: &Ticket) -> Prediction {
    Prediction {
        ticket_id: ticket.id.clone(),
        model_version: "unavailable".to_owned(),
        topic_id: "UNKNOWN".to_owned(),
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

fn unavailable_demo_response_template(language: &str) -> ResponseTemplate {
    let kazakh = language == "KZ";
    ResponseTemplate {
        id: "unavailable".to_owned(),
        title: if kazakh {
            "Жауап үлгісі қолжетімсіз".to_owned()
        } else {
            "Шаблон ответа недоступен".to_owned()
        },
        body: if kazakh {
            "Тілді және шешімді қолмен тексеріп, жауапты өзіңіз құрастырыңыз.".to_owned()
        } else {
            "Проверьте язык и решение вручную, затем составьте ответ самостоятельно.".to_owned()
        },
        language: if kazakh { "kz" } else { "ru" }.to_owned(),
        approved: false,
        source: "UNAVAILABLE".to_owned(),
        template_key: None,
        topic_id: None,
        service_id: None,
        version: None,
    }
}

fn manual_response_template(language: &str) -> ResponseTemplate {
    let kazakh = language.eq_ignore_ascii_case("KZ") || language.eq_ignore_ascii_case("kk");
    ResponseTemplate {
        id: "manual".to_owned(),
        title: if kazakh {
            "Жауапты қолмен жазыңыз"
        } else {
            "Напишите ответ вручную"
        }
        .to_owned(),
        body: String::new(),
        language: if kazakh { "kz" } else { "ru" }.to_owned(),
        approved: false,
        source: "MANUAL_REQUIRED".to_owned(),
        template_key: None,
        topic_id: None,
        service_id: None,
        version: None,
    }
}

fn confidence_state(confidence: f32) -> &'static str {
    if confidence >= 0.85 {
        "high"
    } else if confidence >= 0.7 {
        "medium"
    } else {
        "low"
    }
}

fn priority_for_topic(topic_id: &str) -> &'static str {
    match topic_id {
        "TOPIC-WATER" | "TOPIC-SAFETY" | "TOPIC-HEALTH" => "high",
        "TOPIC-ROADS" | "TOPIC-UTILITIES" | "TOPIC-TRANSPORT" => "medium",
        _ => "normal",
    }
}

fn service_for_topic(topic_id: &str) -> &'static str {
    match topic_id {
        "TOPIC-WATER" => "Водоканал",
        "TOPIC-ROADS" => "Городская инфраструктура",
        "TOPIC-HEALTH" => "Управление здравоохранения",
        "TOPIC-SOCIAL" => "Центр социальной защиты",
        "TOPIC-EDUCATION" => "Управление образования",
        "TOPIC-UTILITIES" => "Городские коммунальные службы",
        "TOPIC-SAFETY" => "Служба безопасности",
        "TOPIC-TRANSPORT" => "Управление транспорта",
        "TOPIC-ENVIRONMENT" => "Управление экологии",
        "TOPIC-DIGITAL" => "Цифровой акимат",
        "TOPIC-HOUSING" => "Жилищная инспекция",
        _ => "Единый контакт-центр",
    }
}

fn topic_label(topics: &[Topic], id: &str) -> String {
    topics
        .iter()
        .find(|topic| topic.id == id)
        .map(|topic| topic.label.clone())
        .unwrap_or_else(|| "Другое".to_owned())
}

fn prediction_for_ticket(ticket: &Ticket, topics: &[Topic]) -> Prediction {
    let (topic_id, confidence, service, priority_state) = demo_topic_for_text(&ticket.text);
    let topic_id = if ticket.topic_id.is_empty() {
        topic_id.to_owned()
    } else {
        ticket.topic_id.clone()
    };
    let confidence = if ticket.topic_id.is_empty() {
        confidence
    } else {
        match ticket.id.as_bytes().first().copied().unwrap_or_default() % 3 {
            0 => 0.94,
            1 => 0.86,
            _ => 0.78,
        }
    };
    let service = if ticket.topic_id.is_empty() {
        service.to_owned()
    } else {
        service_for_topic(&topic_id).to_owned()
    };
    let priority = if ticket.topic_id.is_empty() {
        priority_state.to_owned()
    } else {
        ticket.priority.clone()
    };
    let alternatives = topics
        .iter()
        .filter(|topic| topic.id != topic_id)
        .take(2)
        .enumerate()
        .map(|(index, topic)| AlternativePrediction {
            topic_id: topic.id.clone(),
            topic_label: topic.label.clone(),
            confidence: (0.17 - index as f32 * 0.05).max(0.05),
        })
        .collect();
    let service_provenance = RuleProvenance::manual(
        "Демонстрационное сопоставление; официальные правила 109 не предоставлены",
    );
    let service_provenance = if ticket.topic_id.is_empty() {
        service_provenance
    } else {
        service_provenance.with_fact(ExplainabilityFactField::TopicId, &topic_id)
    };
    Prediction {
        ticket_id: ticket.id.clone(),
        model_version: "classifier-demo-2026-09-001".to_owned(),
        topic_id: topic_id.clone(),
        topic_label: topic_label(topics, &topic_id),
        confidence,
        confidence_state: confidence_state(confidence).to_owned(),
        recommended_service: service,
        predicted_priority: priority,
        routing_reason: "Демонстрационное сопоставление; официальные правила 109 не предоставлены"
            .to_owned(),
        service_provenance,
        priority_provenance: RuleProvenance::manual(
            "Демонстрационный приоритет; официальные правила 109 не предоставлены",
        ),
        alternatives,
        created_at: ticket.created_at.clone(),
    }
}

fn validate_template_body(body: &str) -> Result<(), String> {
    let mut cursor = 0;
    loop {
        let remaining = &body[cursor..];
        let opening = remaining.find("{{");
        let closing = remaining.find("}}");
        let Some(start) = opening else {
            if closing.is_some() {
                return Err("template variable close marker has no opening marker".to_owned());
            }
            return Ok(());
        };
        if closing.is_some_and(|end| end < start) {
            return Err("template variable close marker has no opening marker".to_owned());
        }
        let after_start = &remaining[start + 2..];
        let end = after_start
            .find("}}")
            .ok_or_else(|| "template variable is not closed".to_owned())?;
        let variable = &after_start[..end];
        if !matches!(variable, "topic" | "service" | "region") {
            return Err(format!("unsupported template variable: {variable}"));
        }
        cursor += start + 2 + end + 2;
    }
}

fn render_template_body(body: &str, topic: &str, service: &str, region: &str) -> Option<String> {
    validate_template_body(body).ok()?;
    let mut output = String::with_capacity(body.len());
    let mut cursor = 0;
    while let Some(relative_start) = body[cursor..].find("{{") {
        let start = cursor + relative_start;
        output.push_str(&body[cursor..start]);
        let variable_start = start + 2;
        let relative_end = body[variable_start..].find("}}")?;
        let variable_end = variable_start + relative_end;
        let value = match &body[variable_start..variable_end] {
            "topic" => topic,
            "service" => service,
            "region" => region,
            _ => return None,
        };
        output.push_str(value);
        cursor = variable_end + 2;
    }
    output.push_str(&body[cursor..]);
    Some(output)
}

fn normalize_response_template_input(
    mut input: ResponseTemplateInput,
) -> Result<ResponseTemplateInput, ApiError> {
    input.template_key = input.template_key.trim().to_ascii_lowercase();
    if input.template_key.is_empty()
        || input.template_key.len() > 100
        || !input.template_key.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
        })
    {
        return Err(ApiError::BadRequest(
            "template_key must be 1 to 100 ASCII letters, digits, dots, dashes, or underscores"
                .to_owned(),
        ));
    }
    input.language = match input.language.trim().to_ascii_uppercase().as_str() {
        "RU" | "RUS" => "RU".to_owned(),
        "KZ" | "KK" | "KAZ" => "KZ".to_owned(),
        _ => return Err(ApiError::BadRequest("language must be RU or KZ".to_owned())),
    };
    input.topic_id = input.topic_id.trim().to_owned();
    input.service_id = input.service_id.trim().to_owned();
    if input.topic_id.is_empty()
        || input.topic_id.len() > 100
        || input.service_id.is_empty()
        || input.service_id.len() > 100
    {
        return Err(ApiError::BadRequest(
            "topic_id and service_id must be between 1 and 100 characters".to_owned(),
        ));
    }
    input.body = input.body.trim().to_owned();
    let body_length = input.body.chars().count();
    if body_length == 0 || body_length > MAX_RESPONSE_TEMPLATE_BODY_CHARS {
        return Err(ApiError::BadRequest(format!(
            "body must be between 1 and {MAX_RESPONSE_TEMPLATE_BODY_CHARS} characters"
        )));
    }
    validate_template_body(&input.body).map_err(|_| {
        ApiError::BadRequest("body contains an unsupported template variable".to_owned())
    })?;
    Ok(input)
}

fn response_template(
    store: &Store,
    language: &str,
    topic_id: &str,
    service_id: &str,
    service_name: &str,
    region_name: &str,
) -> ResponseTemplate {
    let normalized_language = match language.trim().to_ascii_uppercase().as_str() {
        "RU" | "RUS" => "RU",
        "KZ" | "KK" | "KAZ" => "KZ",
        _ => return manual_response_template(language),
    };
    let Some(template) = store
        .response_templates
        .values()
        .filter(|template| {
            template.approved
                && template.language == normalized_language
                && template.topic_id.as_deref() == Some(topic_id)
                && template.service_id.as_deref() == Some(service_id)
        })
        .max_by_key(|template| (template.version, template.id.as_str()))
    else {
        return manual_response_template(language);
    };
    let topic = topic_label(&store.topics, topic_id);
    let Some(body) = render_template_body(&template.body, &topic, service_name, region_name) else {
        return manual_response_template(language);
    };
    ResponseTemplate {
        id: template.id.clone(),
        title: format!("Ответ: {topic} · {service_name}"),
        body,
        language: normalized_language.to_ascii_lowercase(),
        approved: true,
        source: "APPROVED_TEMPLATE".to_owned(),
        template_key: Some(template.template_key.clone()),
        topic_id: template.topic_id.clone(),
        service_id: template.service_id.clone(),
        version: Some(template.version),
    }
}

impl Store {
    fn demo() -> Self {
        let regions = demo_regions();
        let topics = demo_topics();
        let examples = [
            (
                "Подскажите, когда восстановят воду на нашей улице?",
                "ru",
                "TOPIC-WATER",
                "high",
            ),
            (
                "Жолдағы шұңқырды жөндеуді сұраймын",
                "kk",
                "TOPIC-ROADS",
                "high",
            ),
            (
                "Записаться к врачу в поликлинику невозможно",
                "ru",
                "TOPIC-HEALTH",
                "medium",
            ),
            (
                "Мектепке қосымша сынып керек",
                "kk",
                "TOPIC-EDUCATION",
                "medium",
            ),
            (
                "Автобус 12 не соблюдает расписание",
                "ru",
                "TOPIC-TRANSPORT",
                "medium",
            ),
            (
                "Қоқыс контейнерлері уақытында жиналмайды",
                "kk",
                "TOPIC-ENVIRONMENT",
                "normal",
            ),
            (
                "Прошу назначить адресную социальную помощь",
                "ru",
                "TOPIC-SOCIAL",
                "normal",
            ),
            (
                "В подъезде не работает освещение",
                "ru",
                "TOPIC-UTILITIES",
                "medium",
            ),
            (
                "Қауіпсіздік камерасы істемейді",
                "kk",
                "TOPIC-SAFETY",
                "high",
            ),
            (
                "Не удаётся получить справку через портал",
                "ru",
                "TOPIC-DIGITAL",
                "normal",
            ),
            (
                "Нужна консультация по очереди на жильё",
                "ru",
                "TOPIC-HOUSING",
                "normal",
            ),
            (
                "Пожалуйста, улучшите освещение дороги",
                "ru",
                "TOPIC-ROADS",
                "medium",
            ),
        ];
        let mut tickets = BTreeMap::new();
        let mut predictions = BTreeMap::new();
        for (index, (text, language, topic_id, priority)) in examples.into_iter().enumerate() {
            let region = &regions[index % regions.len()];
            let created_at = format!("2026-09-{:02}T{:02}:00:00Z", 10 + index / 2, 8 + index % 8);
            let ticket = Ticket {
                id: format!("ticket-{:03}", index + 1),
                external_ref: format!("DEMO-{:04}", index + 1),
                text: text.to_owned(),
                language: language.to_owned(),
                region_id: region.id.clone(),
                region_name: region.name.clone(),
                topic_id: topic_id.to_owned(),
                topic_label: topic_label(&topics, topic_id),
                priority: priority.to_owned(),
                status: if index < 3 { "open" } else { "triaged" }.to_owned(),
                source: "demo".to_owned(),
                created_at,
                closed_at: None,
                updated_at: DEMO_TIMESTAMP.to_owned(),
            };
            let prediction = prediction_for_ticket(&ticket, &topics);
            predictions.insert(ticket.id.clone(), prediction);
            tickets.insert(ticket.id.clone(), ticket);
        }

        let alerts = [
            Alert {
                id: "alert-001".to_owned(),
                incident_key: "demo:topic_spike:R01:TOPIC-WATER".to_owned(),
                alert_type: "topic_spike".to_owned(),
                severity: "high".to_owned(),
                status: "open".to_owned(),
                title: "Всплеск обращений по водоснабжению".to_owned(),
                description: "Зафиксирован необычный рост обращений. Требуется проверка."
                    .to_owned(),
                region_id: "R01".to_owned(),
                topic_id: "TOPIC-WATER".to_owned(),
                ticket_count: 1,
                period_start: None,
                period_end: None,
                current_count: 1,
                baseline: None,
                deviation: None,
                robust_z: None,
                ratio: None,
                detector_version: None,
                linked_ticket_ids: vec!["ticket-001".to_owned()],
                created_at: DEMO_TIMESTAMP.to_owned(),
                detected_at: DEMO_TIMESTAMP.to_owned(),
                acknowledged_by: None,
                acknowledged_at: None,
                closed_by: None,
                closed_at: None,
                detail: json!({"source": "DEMO_ONLY"}),
            },
            Alert {
                id: "alert-002".to_owned(),
                incident_key: "demo:anomaly:R02:TOPIC-TRANSPORT".to_owned(),
                alert_type: "anomaly".to_owned(),
                severity: "medium".to_owned(),
                status: "acknowledged".to_owned(),
                title: "Аномальная динамика по транспорту".to_owned(),
                description: "Зафиксирован необычный рост обращений. Требуется проверка."
                    .to_owned(),
                region_id: "R02".to_owned(),
                topic_id: "TOPIC-TRANSPORT".to_owned(),
                ticket_count: 1,
                period_start: None,
                period_end: None,
                current_count: 1,
                baseline: None,
                deviation: None,
                robust_z: None,
                ratio: None,
                detector_version: None,
                linked_ticket_ids: vec!["ticket-005".to_owned()],
                created_at: "2026-09-20T11:00:00Z".to_owned(),
                detected_at: "2026-09-20T11:00:00Z".to_owned(),
                acknowledged_by: Some("demo-manager".to_owned()),
                acknowledged_at: Some("2026-09-20T12:00:00Z".to_owned()),
                closed_by: None,
                closed_at: None,
                detail: json!({"source": "DEMO_ONLY"}),
            },
        ]
        .into_iter()
        .map(|alert| (alert.id.clone(), alert))
        .collect();

        let metrics = LearningMetrics {
            macro_f1: 0.84,
            accuracy: 0.88,
            evaluated_samples: 120,
        };
        let learning_cycle = LearningCycle {
            id: "cycle-001".to_owned(),
            cycle_id: "cycle-001".to_owned(),
            state: "EVALUATE".to_owned(),
            dataset_version: "dataset-demo-2026-09-001".to_owned(),
            candidate_model_version: "classifier-candidate-2026-09-001".to_owned(),
            collect_started_at: "2026-09-18T10:00:00Z".to_owned(),
            collect_ends_at: "2026-09-25T10:00:00Z".to_owned(),
            evaluation_started_at: Some("2026-09-25T10:00:00Z".to_owned()),
            evaluation_ends_at: Some("2026-10-02T10:00:00Z".to_owned()),
            shadow_prediction_count: 0,
            shadow_inference_failures: 0,
            shadow_operator_decision_count: 0,
            blind_ab_enabled: false,
            production_model_version: Some("classifier-demo-2026-09-001".to_owned()),
            frozen_evaluation_dataset_version: None,
            candidate_dataset_checksum: None,
            min_feedback_count: DEFAULT_LEARNING_MIN_FEEDBACK_COUNT as u32,
            promotion_policy_version: DEFAULT_LEARNING_PROMOTION_POLICY_VERSION.to_owned(),
            manual_close_enabled: true,
            metrics: metrics.clone(),
            feedback_count: 3,
            decision_note: None,
            created_at: "2026-09-18T10:00:00Z".to_owned(),
            updated_at: DEMO_TIMESTAMP.to_owned(),
        };
        let mut learning_cycles = BTreeMap::new();
        learning_cycles.insert(learning_cycle.id.clone(), learning_cycle);

        let labels: Vec<String> = topics.iter().map(|topic| topic.id.clone()).collect();
        let models = [
            ModelVersion {
                id: "classifier-demo-2026-09-001".to_owned(),
                model_family: "deterministic-demo".to_owned(),
                base_model: "rule-based-fixture".to_owned(),
                dataset_version: "dataset-demo-2026-09-000".to_owned(),
                status: "production".to_owned(),
                metrics: LearningMetrics {
                    macro_f1: 0.80,
                    accuracy: 0.83,
                    evaluated_samples: 120,
                },
                languages: vec!["ru".to_owned(), "kk".to_owned()],
                labels: labels.clone(),
                created_at: "2026-09-15T09:00:00Z".to_owned(),
                promoted_at: Some("2026-09-16T09:00:00Z".to_owned()),
            },
            ModelVersion {
                id: "classifier-candidate-2026-09-001".to_owned(),
                model_family: "tfidf-linear-demo".to_owned(),
                base_model: "multilingual baseline".to_owned(),
                dataset_version: "dataset-demo-2026-09-001".to_owned(),
                status: "candidate".to_owned(),
                metrics,
                languages: vec!["ru".to_owned(), "kk".to_owned()],
                labels,
                created_at: "2026-09-18T10:00:00Z".to_owned(),
                promoted_at: None,
            },
        ]
        .into_iter()
        .map(|model| (model.id.clone(), model))
        .collect();

        let demo_confirmed_topic_label = topic_label(&topics, "TOPIC-ROADS");
        let demo_decision_prediction = predictions.get("ticket-002");
        let demo_predicted_service =
            demo_decision_prediction.map(|prediction| prediction.recommended_service.clone());
        let demo_predicted_priority =
            demo_decision_prediction.map(|prediction| prediction.predicted_priority.clone());
        let demo_model_version =
            demo_decision_prediction.map(|prediction| prediction.model_version.clone());
        Self {
            regions,
            topics,
            response_templates: BTreeMap::new(),
            tickets,
            predictions,
            decisions: vec![OperatorDecision {
                id: "decision-001".to_owned(),
                ticket_id: "ticket-002".to_owned(),
                action: "correct".to_owned(),
                predicted_topic_id: "TOPIC-ROADS".to_owned(),
                predicted_service: demo_predicted_service,
                predicted_priority: demo_predicted_priority,
                model_version: demo_model_version,
                confirmed_topic_id: "TOPIC-ROADS".to_owned(),
                confirmed_topic_label: demo_confirmed_topic_label,
                confirmed_service_id: Some("Городская инфраструктура".to_owned()),
                service: "Городская инфраструктура".to_owned(),
                priority: "high".to_owned(),
                service_provenance: RuleProvenance::manual("Демо-решение оператора"),
                priority_provenance: RuleProvenance::manual("Демо-решение оператора"),
                note: Some("Подтверждено оператором для demo flow".to_owned()),
                user_id: "demo-operator".to_owned(),
                created_at: "2026-09-20T08:30:00Z".to_owned(),
            }],
            alerts,
            learning_cycles,
            learning_feedback: Vec::new(),
            models,
            next_ticket_number: 13,
            next_decision_number: 2,
            next_cycle_number: 2,
            next_feedback_number: 1,
            next_response_template_number: 1,
        }
    }
}

/// Build the application router.  This function is public so integration
/// tests and local tooling can exercise the API without binding a TCP port.
pub fn app(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/api/v1/openapi.json", get(openapi))
        .route("/api/v1/docs", get(openapi))
        .route("/api/v1/tickets", get(list_tickets).post(create_ticket))
        .route("/api/v1/audit", get(audit_log))
        .route("/api/v1/import", post(import_tickets))
        .route("/api/v1/datasets/provenance", get(dataset_provenance))
        .route("/api/v1/tickets/{ticket_id}", get(get_ticket))
        .route(
            "/api/v1/tickets/{ticket_id}/vector",
            delete(delete_ticket_vector),
        )
        .route(
            "/api/v1/tickets/{ticket_id}/pulse-state",
            put(store_pulse_state),
        )
        .route(
            "/api/v1/tickets/{ticket_id}/prediction",
            get(get_prediction),
        )
        .route("/api/v1/assist/preview", post(assist_preview))
        .route(
            "/api/v1/response-templates",
            get(list_response_templates).post(create_response_template),
        )
        .route(
            "/api/v1/response-templates/import",
            post(import_response_templates),
        )
        .route(
            "/api/v1/response-templates/{template_id}",
            get(get_response_template)
                .put(update_response_template)
                .delete(delete_response_template),
        )
        .route(
            "/api/v1/response-templates/{template_id}/approve",
            post(approve_response_template),
        )
        .route("/api/v1/assist/{ticket_id}/confirm", post(confirm_ticket))
        .route("/api/v1/assist/{ticket_id}/correct", post(correct_ticket))
        .route("/api/v1/assist/confirm", post(confirm_ticket_from_body))
        .route("/api/v1/assist/correct", post(correct_ticket_from_body))
        .route("/api/v1/assist/confirm/{ticket_id}", post(confirm_ticket))
        .route("/api/v1/assist/correct/{ticket_id}", post(correct_ticket))
        .route("/api/v1/analytics", get(analytics))
        .route("/api/v1/analytics/overview", get(analytics))
        .route("/api/v1/analytics/drilldown", get(analytics_drilldown))
        .route("/api/v1/taxonomy", get(taxonomy))
        .route("/api/v1/analytics/query", post(analytics_query))
        .route("/api/v1/retrieval/reindex", post(reindex_vectors))
        .route("/api/v1/analytics/export.pdf", get(export_pdf))
        .route("/api/v1/analytics/export.xlsx", get(export_xlsx))
        .route("/api/v1/reports", get(reports))
        .route("/api/v1/events", get(events))
        .route("/api/v1/forecast", get(forecast))
        .route("/api/v1/alerts", get(list_alerts))
        .route("/api/v1/alerts/detect", post(detect_alerts))
        .route("/api/v1/alerts/{alert_id}", get(get_alert))
        .route("/api/v1/alerts/{alert_id}/ack", post(ack_alert))
        .route("/api/v1/alerts/{alert_id}/acknowledge", post(ack_alert))
        .route("/api/v1/alerts/{alert_id}/close", post(close_alert))
        .route(
            "/api/v1/learning",
            get(learning_overview).post(create_learning_cycle),
        )
        .route("/api/v1/learning/{cycle_id}", get(get_learning_cycle))
        .route(
            "/api/v1/learning/{cycle_id}/feedback",
            post(add_learning_feedback),
        )
        .route(
            "/api/v1/learning/{cycle_id}/promote",
            post(promote_learning_cycle),
        )
        .route(
            "/api/v1/learning/{cycle_id}/reject",
            post(reject_learning_cycle),
        )
        .route("/api/v1/learning/cycle", get(learning_overview))
        .route("/api/v1/learning/cycle/close", post(close_learning_cycle))
        .route(
            "/api/v1/learning/candidate/evaluation",
            get(candidate_evaluation),
        )
        .route(
            "/api/v1/learning/candidate/promote",
            post(promote_active_learning_cycle),
        )
        .route(
            "/api/v1/learning/candidate/reject",
            post(reject_active_learning_cycle),
        )
        .route(
            "/api/v1/tickets/{ticket_id}/relation-feedback",
            post(relation_feedback),
        )
        .route("/api/v1/models", get(list_models))
        .route("/api/v1/models/{model_id}", get(get_model))
        .route("/api/v1/models/{model_id}/promote", post(promote_model))
        .layer(middleware::from_fn(request_context))
        .layer(
            CorsLayer::new()
                .allow_origin(tower_http::cors::Any)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers(tower_http::cors::Any)
                .expose_headers([
                    header::HeaderName::from_static("x-request-id"),
                    header::HeaderName::from_static("x-trace-id"),
                ]),
        )
        .with_state(state)
}

async fn request_context(mut request: Request, next: Next) -> Response {
    let request_id = request
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| {
            format!(
                "demo-{}",
                Utc::now().timestamp_nanos_opt().unwrap_or_default()
            )
        });
    let incoming_trace_id = request
        .headers()
        .get("x-trace-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(request_id.as_str());
    let trace_id = safe_trace_id(incoming_trace_id);
    let started = Instant::now();
    let method = request.method().clone();
    let path = request
        .uri()
        .path()
        .split('/')
        .filter(|segment| !segment.is_empty())
        .take(3)
        .collect::<Vec<_>>()
        .join("/");
    let path = format!("/{path}");
    let log_request_id = format!(
        "core-{}-{}",
        Utc::now().timestamp_nanos_opt().unwrap_or_default(),
        LOG_REQUEST_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    );
    request.extensions_mut().insert(request_id.clone());
    if let Ok(value) = HeaderValue::from_str(&log_request_id) {
        request.headers_mut().insert("x-pulse-log-id", value);
    }
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        request.headers_mut().insert("x-request-id", value);
    }
    if let Ok(value) = HeaderValue::from_str(&trace_id) {
        request.headers_mut().insert("x-trace-id", value);
    }
    let mut response = next.run(request).await;
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", value);
    }
    if let Ok(value) = HeaderValue::from_str(&trace_id) {
        response.headers_mut().insert("x-trace-id", value);
    }
    let error_code = response_error_code(response.status());
    info!(
        service = SERVICE_NAME,
        request_id = %log_request_id,
        trace_id = %trace_id,
        endpoint = %path,
        method = %method,
        latency_ms,
        status = response.status().as_u16(),
        model_version = "n/a",
        error_code,
        "request_completed"
    );
    response
}

async fn healthz() -> Json<Value> {
    Json(json!({
        "status": "ok",
        "service": SERVICE_NAME,
        "version": API_VERSION,
    }))
}

async fn readyz(State(state): State<AppState>) -> Response {
    let (storage, demo, report) = if let Some(repository) = state.repository() {
        ("postgres", false, repository.readiness().await)
    } else {
        let store = match state.read_store() {
            Ok(store) => store,
            Err(error) => return error.into_response(),
        };
        (
            "in_memory_demo",
            true,
            json!({
                "status": "ready",
                "checks": {
                    "store": { "status": "ready" },
                    "postgres": { "status": "not_applicable" },
                    "database_migrations": { "status": "not_applicable" },
                    "qdrant": { "status": "not_applicable" },
                    "qdrant_collection": { "status": "not_applicable" },
                    "ml_service": { "status": "not_applicable" },
                    "model_artifact": { "status": "not_applicable" },
                    "ml_embedder": { "status": "not_applicable" }
                },
                "demo_ticket_count": store.tickets.len(),
                "demo_model_count": store.models.len()
            }),
        )
    };
    let ready = report.get("status").and_then(Value::as_str) == Some("ready");
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    let body = json!({
        "status": if ready { "ready" } else { "not_ready" },
        "service": SERVICE_NAME,
        "storage": storage,
        "demo": demo,
        "checks": report.get("checks").cloned().unwrap_or_else(|| json!({})),
    });
    (status, Json(body)).into_response()
}

#[derive(Debug, Deserialize)]
pub struct TicketQuery {
    pub region_id: Option<String>,
    pub topic_id: Option<String>,
    pub language: Option<String>,
    pub status: Option<String>,
    pub q: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct TicketListResponse {
    pub items: Vec<Ticket>,
    pub total: usize,
    pub limit: usize,
    pub offset: usize,
}

async fn list_tickets(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<TicketQuery>,
) -> Result<Json<TicketListResponse>, ApiError> {
    let actor = require_role(
        &headers,
        &state.config,
        &[Role::Operator, Role::Manager, Role::Admin],
    )?;
    if let Some(repository) = state.repository() {
        let response = repository
            .list_tickets(&query)
            .await
            .map_err(ApiError::Internal)?;
        repository
            .audit(
                &actor.user_id,
                "READ_TICKET_LIST",
                "ticket_collection",
                None,
                Some(&log_request_id_from_headers(&headers)),
                Some("ticket list viewed"),
                ticket_list_audit_metadata(&response),
            )
            .await
            .map_err(ApiError::Internal)?;
        return Ok(Json(response));
    }
    let store = state.read_store()?;
    let limit = query.limit.unwrap_or(50).clamp(1, 100);
    let offset = query.offset.unwrap_or(0);
    let search = query.q.as_deref().map(str::to_lowercase);
    let mut items: Vec<Ticket> = store
        .tickets
        .values()
        .filter(|ticket| {
            query
                .region_id
                .as_deref()
                .is_none_or(|value| ticket.region_id == value)
                && query
                    .topic_id
                    .as_deref()
                    .is_none_or(|value| ticket.topic_id == value)
                && query
                    .language
                    .as_deref()
                    .is_none_or(|value| ticket.language == value)
                && query
                    .status
                    .as_deref()
                    .is_none_or(|value| ticket.status == value)
                && search
                    .as_deref()
                    .is_none_or(|value| ticket.text.to_lowercase().contains(value))
        })
        .cloned()
        .collect();
    let total = items.len();
    items = items.into_iter().skip(offset).take(limit).collect();
    Ok(Json(TicketListResponse {
        items,
        total,
        limit,
        offset,
    }))
}

fn ticket_list_audit_metadata(response: &TicketListResponse) -> Value {
    json!({
        "ticket_ids": response.items.iter().map(|item| &item.id).collect::<Vec<_>>(),
        "limit": response.limit,
        "offset": response.offset,
    })
}

#[derive(Debug, Deserialize)]
pub struct CreateTicketRequest {
    pub text: String,
    pub language: Option<String>,
    pub region_id: Option<String>,
    pub priority: Option<String>,
    pub source: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ImportRequest {
    pub source_system: String,
    pub source_uri: Option<String>,
    pub dataset_version: Option<String>,
    pub manifest_uri: Option<String>,
    pub manifest_sha256: Option<String>,
    pub is_synthetic: bool,
    #[serde(default)]
    pub tickets: Vec<Value>,
    #[serde(default)]
    pub quarantine: Vec<Value>,
}

#[derive(Debug, Serialize)]
pub struct ImportResponse {
    pub import_run_id: String,
    pub source_system: String,
    pub dataset_version: String,
    pub total_rows: usize,
    pub imported_rows: usize,
    pub duplicate_rows: usize,
    pub quarantined_rows: usize,
    pub indexed_rows: usize,
    pub is_synthetic: bool,
    pub source: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct DatasetProvenance {
    pub synthetic_ticket_count: i64,
    pub real_ticket_count: i64,
    pub unassigned_ticket_count: i64,
    pub quarantined_row_count: i64,
    pub dataset_version_count: i64,
}

#[derive(Debug, Serialize)]
pub struct TicketDetailResponse {
    pub ticket: Ticket,
    pub prediction: Prediction,
    pub latest_decision: Option<OperatorDecision>,
}

async fn create_ticket(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CreateTicketRequest>,
) -> Result<(StatusCode, Json<TicketDetailResponse>), ApiError> {
    let actor = require_role(
        &headers,
        &state.config,
        &[Role::Operator, Role::Manager, Role::Admin],
    )?;
    let text = request.text.trim();
    if text.is_empty() {
        return Err(ApiError::BadRequest("text must not be empty".to_owned()));
    }
    if text.chars().count() > 10_000 {
        return Err(ApiError::BadRequest(
            "text must be at most 10000 characters".to_owned(),
        ));
    }
    if let Some(repository) = state.repository() {
        let response = repository
            .create_ticket(
                text,
                request.language.as_deref(),
                request.region_id.as_deref(),
                request.priority.as_deref(),
                request.source.as_deref(),
                &request_id_from_headers(&headers),
            )
            .await
            .map_err(ApiError::Internal)?;
        repository
            .audit(
                &actor.user_id,
                "CREATE_TICKET",
                "ticket",
                Some(&response.ticket.id),
                Some(&request_id_from_headers(&headers)),
                None,
                json!({"source": &response.ticket.source}),
            )
            .await
            .map_err(ApiError::Internal)?;
        info!(
            service = SERVICE_NAME,
            ticket_id = %response.ticket.id,
            model_version = %response.prediction.model_version,
            request_id = %log_request_id_from_headers(&headers),
            trace_id = %log_trace_id_from_headers(&headers),
            endpoint = "/api/v1/tickets",
            latency_ms = 0.0_f64,
            status = 201_u16,
            error_code = "none",
            storage = "postgres",
            result_state = "created",
            "ticket_created"
        );
        return Ok((StatusCode::CREATED, Json(response)));
    }
    let mut store = state.write_store()?;
    let region_id = normalize_region_id(request.region_id.unwrap_or_else(|| "R01".to_owned()));
    let region = store
        .regions
        .iter()
        .find(|region| region.id == region_id)
        .cloned()
        .ok_or_else(|| ApiError::BadRequest(format!("unknown region_id: {region_id}")))?;
    let language = request
        .language
        .unwrap_or_else(|| detect_language(text).to_owned());
    if language != "ru" && language != "kk" {
        return Err(ApiError::BadRequest("language must be ru or kk".to_owned()));
    }
    let (topic_id, _, _, _) = demo_topic_for_text(text);
    let priority = request
        .priority
        .unwrap_or_else(|| priority_for_topic(topic_id).to_owned());
    let id = format!("ticket-{:03}", store.next_ticket_number);
    store.next_ticket_number += 1;
    let ticket = Ticket {
        id: id.clone(),
        external_ref: format!("DEMO-{:04}", store.next_ticket_number),
        text: text.to_owned(),
        language,
        region_id: region.id,
        region_name: region.name,
        topic_id: topic_id.to_owned(),
        topic_label: topic_label(&store.topics, topic_id),
        priority,
        status: "open".to_owned(),
        source: request.source.unwrap_or_else(|| "operator".to_owned()),
        created_at: DEMO_TIMESTAMP.to_owned(),
        closed_at: None,
        updated_at: DEMO_TIMESTAMP.to_owned(),
    };
    let prediction = prediction_for_ticket(&ticket, &store.topics);
    store.predictions.insert(id.clone(), prediction.clone());
    store.tickets.insert(id, ticket.clone());
    info!(
        service = SERVICE_NAME,
        ticket_id = %ticket.id,
        model_version = %prediction.model_version,
        request_id = "n/a",
        trace_id = "n/a",
        endpoint = "/api/v1/tickets",
        latency_ms = 0.0_f64,
        status = 201_u16,
        error_code = "none",
        result_state = "created",
        "ticket_created"
    );
    Ok((
        StatusCode::CREATED,
        Json(TicketDetailResponse {
            ticket,
            prediction,
            latest_decision: None,
        }),
    ))
}

async fn dataset_provenance(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<DatasetProvenance>, ApiError> {
    require_role(
        &headers,
        &state.config,
        &[Role::Operator, Role::Manager, Role::Admin],
    )?;
    if let Some(repository) = state.repository() {
        return repository
            .dataset_provenance()
            .await
            .map(Json)
            .map_err(ApiError::Internal);
    }
    let store = state.read_store()?;
    Ok(Json(DatasetProvenance {
        synthetic_ticket_count: store.tickets.len() as i64,
        real_ticket_count: 0,
        unassigned_ticket_count: 0,
        quarantined_row_count: 0,
        dataset_version_count: 1,
    }))
}

async fn import_tickets(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ImportRequest>,
) -> Result<(StatusCode, Json<ImportResponse>), ApiError> {
    let actor = require_role(&headers, &state.config, &[Role::Admin, Role::Manager])?;
    let Some(repository) = state.repository() else {
        return Err(ApiError::Conflict(
            "imports require PULSE_STORAGE=postgres".to_owned(),
        ));
    };
    let response = repository
        .import_tickets(&request, &request_id_from_headers(&headers))
        .await
        .map_err(|error| match error {
            pg::ImportError::Invalid(message) => ApiError::BadRequest(message),
            pg::ImportError::Conflict(message) => ApiError::Conflict(message),
            pg::ImportError::Internal(message) => ApiError::Internal(message),
        })?;
    repository
        .audit(
            &actor.user_id,
            "IMPORT_DATASET",
            "data_import_run",
            Some(&response.import_run_id),
            Some(&request_id_from_headers(&headers)),
            None,
            json!({"source_system": &response.source_system, "dataset_version": &response.dataset_version}),
        )
        .await
        .map_err(ApiError::Internal)?;
    info!(
        service = SERVICE_NAME,
        source_system = %response.source_system,
        import_run_id = %response.import_run_id,
        imported_rows = response.imported_rows,
        quarantined_rows = response.quarantined_rows,
        request_id = %log_request_id_from_headers(&headers),
        trace_id = %log_trace_id_from_headers(&headers),
        endpoint = "/api/v1/import",
        latency_ms = 0.0_f64,
        status = 201_u16,
        error_code = "none",
        "data_import_completed"
    );
    Ok((StatusCode::CREATED, Json(response)))
}

async fn reindex_vectors(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let actor = require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    let Some(repository) = state.repository() else {
        return Err(ApiError::Conflict(
            "Qdrant reindex requires PULSE_STORAGE=postgres".to_owned(),
        ));
    };
    let response = repository
        .queue_reindex()
        .await
        .map_err(ApiError::Internal)?;
    repository
        .audit(
            &actor.user_id,
            "QUEUE_QDRANT_REINDEX",
            "background_job",
            response.get("job_id").and_then(Value::as_str),
            Some(&request_id_from_headers(&headers)),
            None,
            response.clone(),
        )
        .await
        .map_err(ApiError::Internal)?;
    Ok((StatusCode::ACCEPTED, Json(response)))
}

async fn delete_ticket_vector(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(ticket_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let actor = require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    let Some(repository) = state.repository() else {
        return Err(ApiError::Conflict(
            "Qdrant vector deletion requires PULSE_STORAGE=postgres".to_owned(),
        ));
    };
    let response = repository
        .delete_vector(&ticket_id)
        .await
        .map_err(|error| {
            if error.contains("not found") {
                ApiError::NotFound(error)
            } else {
                ApiError::Internal(error)
            }
        })?;
    repository
        .audit(
            &actor.user_id,
            "DELETE_QDRANT_VECTOR",
            "ticket",
            Some(&ticket_id),
            Some(&request_id_from_headers(&headers)),
            None,
            response.clone(),
        )
        .await
        .map_err(ApiError::Internal)?;
    Ok(Json(response))
}

#[derive(Debug, Deserialize, Default)]
struct AuditLogQuery {
    limit: Option<i64>,
    offset: Option<i64>,
}

async fn audit_log(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<AuditLogQuery>,
) -> Result<Json<pg::AuditLogPage>, ApiError> {
    require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    let limit = query.limit.unwrap_or(AUDIT_LOG_DEFAULT_LIMIT);
    if !(1..=AUDIT_LOG_MAX_LIMIT).contains(&limit) {
        return Err(ApiError::BadRequest(
            "limit must be between 1 and 100".to_owned(),
        ));
    }
    let offset = query.offset.unwrap_or(0);
    if !(0..=AUDIT_LOG_MAX_OFFSET).contains(&offset) {
        return Err(ApiError::BadRequest(
            "offset must be between 0 and 1000000".to_owned(),
        ));
    }
    let repository = state
        .repository()
        .ok_or_else(|| ApiError::Conflict("audit log requires PostgreSQL storage".to_owned()))?;
    repository
        .list_audit_log(limit, offset)
        .await
        .map(Json)
        .map_err(ApiError::Internal)
}

async fn get_ticket(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(ticket_id): Path<String>,
) -> Result<Json<TicketDetailResponse>, ApiError> {
    let actor = require_role(
        &headers,
        &state.config,
        &[Role::Operator, Role::Manager, Role::Admin],
    )?;
    if let Some(repository) = state.repository() {
        let response = repository.get_ticket(&ticket_id).await.map_err(|error| {
            if error.contains("not found") {
                ApiError::NotFound(error)
            } else {
                ApiError::Internal(error)
            }
        })?;
        repository
            .audit(
                &actor.user_id,
                "READ_TICKET_DETAIL",
                "ticket",
                Some(&ticket_id),
                Some(&log_request_id_from_headers(&headers)),
                Some("ticket detail viewed"),
                json!({}),
            )
            .await
            .map_err(ApiError::Internal)?;
        return Ok(Json(response));
    }
    let store = state.read_store()?;
    let ticket = store
        .tickets
        .get(&ticket_id)
        .cloned()
        .ok_or_else(|| ApiError::NotFound(format!("ticket {ticket_id} not found")))?;
    let prediction = store
        .predictions
        .get(&ticket_id)
        .cloned()
        .unwrap_or_else(|| prediction_for_ticket(&ticket, &store.topics));
    let latest_decision = store
        .decisions
        .iter()
        .rev()
        .find(|decision| decision.ticket_id == ticket_id)
        .cloned();
    Ok(Json(TicketDetailResponse {
        ticket,
        prediction,
        latest_decision,
    }))
}

async fn get_prediction(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(ticket_id): Path<String>,
) -> Result<Json<Prediction>, ApiError> {
    require_role(
        &headers,
        &state.config,
        &[Role::Operator, Role::Manager, Role::Admin],
    )?;
    if let Some(repository) = state.repository() {
        let response = repository
            .get_prediction(&ticket_id)
            .await
            .map_err(|error| {
                if error.contains("not found") {
                    ApiError::NotFound(error)
                } else {
                    ApiError::Internal(error)
                }
            })?;
        return Ok(Json(response));
    }
    let store = state.read_store()?;
    store
        .predictions
        .get(&ticket_id)
        .cloned()
        .ok_or_else(|| ApiError::NotFound(format!("prediction for {ticket_id} not found")))
        .map(Json)
}

#[derive(Debug, Deserialize)]
pub struct PreviewRequest {
    #[serde(alias = "ticketId")]
    pub ticket_id: Option<String>,
    pub text: Option<String>,
    pub language: Option<String>,
    #[serde(alias = "regionId")]
    pub region_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AssistPreviewResponse {
    pub ticket: Ticket,
    pub prediction: Prediction,
    pub similar_tickets: Vec<SimilarTicket>,
    pub duplicate_candidates: Vec<SimilarTicket>,
    pub repeat_candidates: Vec<SimilarTicket>,
    pub response_template: ResponseTemplate,
    pub source: String,
    pub orchestration: AssistOrchestration,
}

#[derive(Debug, Serialize)]
pub struct AssistOrchestration {
    pub request_id: String,
    pub trace_id: String,
    pub status: String,
    pub needs_review: bool,
    pub language: String,
    pub latency_ms: f64,
    pub model_versions: std::collections::BTreeMap<String, String>,
    pub stages: Vec<AssistStage>,
}

#[derive(Debug, Serialize)]
pub struct AssistStage {
    pub name: String,
    pub status: String,
    pub latency_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
}

fn known_relation_identifier(value: &str) -> bool {
    let normalized = value.trim();
    !normalized.is_empty()
        && !normalized.eq_ignore_ascii_case("unknown")
        && !normalized.eq_ignore_ascii_case("unavailable")
}

pub(crate) fn related_ticket_candidate(
    ticket_id: String,
    score: f32,
    current_topic_id: &str,
    current_region_id: &str,
    current_created_at: Option<&DateTime<Utc>>,
    candidate_topic_id: String,
    candidate_region_id: String,
    candidate_created_at: Option<&DateTime<Utc>>,
    model_version: &str,
    distance_metric: &str,
) -> Option<SimilarTicket> {
    if !score.is_finite() || score < RELATED_CANDIDATE_THRESHOLD {
        return None;
    }

    let same_topic = known_relation_identifier(current_topic_id)
        && known_relation_identifier(&candidate_topic_id)
        && current_topic_id
            .trim()
            .eq_ignore_ascii_case(candidate_topic_id.trim());
    let same_region = known_relation_identifier(current_region_id)
        && known_relation_identifier(&candidate_region_id)
        && current_region_id
            .trim()
            .eq_ignore_ascii_case(candidate_region_id.trim());
    let within_time_window = match (current_created_at, candidate_created_at) {
        (Some(current), Some(candidate)) => {
            current
                .signed_duration_since(candidate.to_owned())
                .num_days()
                .unsigned_abs()
                <= 30
        }
        _ => false,
    };

    let (relation, threshold) = if score >= DUPLICATE_CANDIDATE_THRESHOLD
        && same_topic
        && same_region
        && within_time_window
    {
        ("duplicate", DUPLICATE_CANDIDATE_THRESHOLD)
    } else if same_topic && within_time_window {
        ("repeat", RELATED_CANDIDATE_THRESHOLD)
    } else {
        ("similar", RELATED_CANDIDATE_THRESHOLD)
    };
    let mut matched_factors = Vec::new();
    if same_topic {
        matched_factors.push("topic_match".to_owned());
    }
    if same_region {
        matched_factors.push("region_match".to_owned());
    }
    if within_time_window {
        matched_factors.push("within_30_days".to_owned());
    }

    Some(SimilarTicket {
        ticket_id,
        score,
        relation: relation.to_owned(),
        topic_id: candidate_topic_id,
        region_id: candidate_region_id,
        created_at: candidate_created_at.map(DateTime::to_rfc3339),
        matched_factors,
        suggestion: RelationSuggestionSnapshot {
            score,
            threshold,
            rule_version: RELATED_CANDIDATE_RULE_VERSION.to_owned(),
            model_version: model_version.to_owned(),
            distance_metric: distance_metric.to_owned(),
        },
    })
}

// Offline fixture-only related-ticket behavior. Real mode uses Qdrant vector search.
fn demo_related_tickets(store: &Store, ticket: &Ticket, topic_id: &str) -> Vec<SimilarTicket> {
    let current_created_at = DateTime::parse_from_rfc3339(&ticket.created_at)
        .ok()
        .map(|value| value.with_timezone(&Utc));
    store
        .tickets
        .values()
        .filter(|candidate| candidate.id != ticket.id)
        .filter(|candidate| candidate.topic_id == topic_id)
        .take(5)
        .enumerate()
        .filter_map(|(index, candidate)| {
            let candidate_created_at = DateTime::parse_from_rfc3339(&candidate.created_at)
                .ok()
                .map(|value| value.with_timezone(&Utc));
            related_ticket_candidate(
                candidate.id.clone(),
                (0.91 - index as f32 * 0.11).max(0.5),
                topic_id,
                &ticket.region_id,
                current_created_at.as_ref(),
                candidate.topic_id.clone(),
                candidate.region_id.clone(),
                candidate_created_at.as_ref(),
                "unavailable",
                "fixture",
            )
        })
        .collect()
}

fn require_template_manager(headers: &HeaderMap, config: &Config) -> Result<Actor, ApiError> {
    require_role(headers, config, &[Role::Manager, Role::Admin])
}

fn template_repository_error(error: String) -> ApiError {
    if let Some(message) = error.strip_prefix("bad request: ") {
        ApiError::BadRequest(message.to_owned())
    } else if let Some(message) = error.strip_prefix("not found: ") {
        ApiError::NotFound(message.to_owned())
    } else if let Some(message) = error.strip_prefix("conflict: ") {
        ApiError::Conflict(message.to_owned())
    } else {
        ApiError::Internal(error)
    }
}

fn validate_memory_template_references(
    input: &ResponseTemplateInput,
    store: &Store,
) -> Result<(), ApiError> {
    if !store.topics.iter().any(|topic| topic.id == input.topic_id) {
        return Err(ApiError::BadRequest(format!(
            "unknown topic_id: {}",
            input.topic_id
        )));
    }
    if !store
        .topics
        .iter()
        .any(|topic| service_for_topic(&topic.id) == input.service_id)
    {
        return Err(ApiError::BadRequest(format!(
            "unknown service_id: {}",
            input.service_id
        )));
    }
    Ok(())
}

fn insert_memory_templates(
    store: &mut Store,
    inputs: &[ResponseTemplateInput],
    actor_id: &str,
) -> Vec<ResponseTemplateRecord> {
    let mut records = Vec::with_capacity(inputs.len());
    for input in inputs {
        let version = store
            .response_templates
            .values()
            .filter(|record| {
                record.template_key == input.template_key && record.language == input.language
            })
            .map(|record| record.version)
            .max()
            .unwrap_or(0)
            + 1;
        let id = format!("template-{:04}", store.next_response_template_number);
        store.next_response_template_number += 1;
        let timestamp = Utc::now().to_rfc3339();
        let record = ResponseTemplateRecord {
            id: id.clone(),
            template_key: input.template_key.clone(),
            language: input.language.clone(),
            topic_id: Some(input.topic_id.clone()),
            service_id: Some(input.service_id.clone()),
            body: input.body.clone(),
            approved: false,
            version,
            created_by: Some(actor_id.to_owned()),
            updated_by: Some(actor_id.to_owned()),
            approved_by: None,
            approved_at: None,
            created_at: timestamp.clone(),
            updated_at: timestamp,
        };
        store.response_templates.insert(id, record.clone());
        records.push(record);
    }
    records
}

fn ordered_memory_templates(store: &Store) -> Vec<ResponseTemplateRecord> {
    let mut items = store
        .response_templates
        .values()
        .cloned()
        .collect::<Vec<_>>();
    items.sort_by(|left, right| {
        left.template_key
            .cmp(&right.template_key)
            .then_with(|| left.language.cmp(&right.language))
            .then_with(|| right.version.cmp(&left.version))
    });
    items
}

fn normalize_response_template_body(body: String) -> Result<String, ApiError> {
    let body = body.trim().to_owned();
    let body_length = body.chars().count();
    if body_length == 0 || body_length > MAX_RESPONSE_TEMPLATE_BODY_CHARS {
        return Err(ApiError::BadRequest(format!(
            "body must be between 1 and {MAX_RESPONSE_TEMPLATE_BODY_CHARS} characters"
        )));
    }
    validate_template_body(&body).map_err(|_| {
        ApiError::BadRequest("body contains an unsupported template variable".to_owned())
    })?;
    Ok(body)
}

async fn list_response_templates(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<ResponseTemplatesResponse>, ApiError> {
    require_template_manager(&headers, &state.config)?;
    if let Some(repository) = state.repository() {
        return repository
            .list_response_templates()
            .await
            .map(Json)
            .map_err(template_repository_error);
    }
    let store = state.read_store()?;
    Ok(Json(ResponseTemplatesResponse {
        items: ordered_memory_templates(&store),
    }))
}

async fn get_response_template(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(template_id): Path<String>,
) -> Result<Json<ResponseTemplateRecord>, ApiError> {
    require_template_manager(&headers, &state.config)?;
    if let Some(repository) = state.repository() {
        return repository
            .get_response_template(&template_id)
            .await
            .map(Json)
            .map_err(template_repository_error);
    }
    let store = state.read_store()?;
    store
        .response_templates
        .get(&template_id)
        .cloned()
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("response template {template_id} not found")))
}

async fn create_response_template(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(input): Json<ResponseTemplateInput>,
) -> Result<Json<ResponseTemplateRecord>, ApiError> {
    let actor = require_template_manager(&headers, &state.config)?;
    let input = normalize_response_template_input(input)?;
    let record = if let Some(repository) = state.repository() {
        let mut records = repository
            .create_response_templates(std::slice::from_ref(&input), &actor.user_id)
            .await
            .map_err(template_repository_error)?;
        let record = records
            .pop()
            .ok_or_else(|| ApiError::Internal("template insert returned no record".to_owned()))?;
        repository
            .audit(
                &actor.user_id,
                "CREATE_RESPONSE_TEMPLATE",
                "response_template",
                Some(&record.id),
                Some(&request_id_from_headers(&headers)),
                None,
                json!({"template_key": record.template_key, "version": record.version}),
            )
            .await
            .map_err(ApiError::Internal)?;
        record
    } else {
        let mut store = state.write_store()?;
        validate_memory_template_references(&input, &store)?;
        insert_memory_templates(&mut store, &[input], &actor.user_id)
            .into_iter()
            .next()
            .ok_or_else(|| ApiError::Internal("template insert returned no record".to_owned()))?
    };
    Ok(Json(record))
}

async fn import_response_templates(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ImportResponseTemplatesRequest>,
) -> Result<Json<ResponseTemplatesResponse>, ApiError> {
    let actor = require_template_manager(&headers, &state.config)?;
    if request.items.is_empty() || request.items.len() > MAX_RESPONSE_TEMPLATE_IMPORT_ITEMS {
        return Err(ApiError::BadRequest(format!(
            "items must contain between 1 and {MAX_RESPONSE_TEMPLATE_IMPORT_ITEMS} templates"
        )));
    }
    let inputs = request
        .items
        .into_iter()
        .map(normalize_response_template_input)
        .collect::<Result<Vec<_>, _>>()?;
    let records = if let Some(repository) = state.repository() {
        let records = repository
            .create_response_templates(&inputs, &actor.user_id)
            .await
            .map_err(template_repository_error)?;
        let record_ids = records
            .iter()
            .map(|record| record.id.as_str())
            .collect::<Vec<_>>();
        repository
            .audit(
                &actor.user_id,
                "IMPORT_RESPONSE_TEMPLATES",
                "response_template",
                None,
                Some(&request_id_from_headers(&headers)),
                None,
                json!({"count": records.len(), "template_ids": record_ids}),
            )
            .await
            .map_err(ApiError::Internal)?;
        records
    } else {
        let mut store = state.write_store()?;
        for input in &inputs {
            validate_memory_template_references(input, &store)?;
        }
        insert_memory_templates(&mut store, &inputs, &actor.user_id)
    };
    Ok(Json(ResponseTemplatesResponse { items: records }))
}

async fn update_response_template(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(template_id): Path<String>,
    Json(request): Json<UpdateResponseTemplateRequest>,
) -> Result<Json<ResponseTemplateRecord>, ApiError> {
    let actor = require_template_manager(&headers, &state.config)?;
    let body = normalize_response_template_body(request.body)?;
    let record = if let Some(repository) = state.repository() {
        let record = repository
            .update_response_template(&template_id, &body, &actor.user_id)
            .await
            .map_err(template_repository_error)?;
        repository
            .audit(
                &actor.user_id,
                "UPDATE_RESPONSE_TEMPLATE",
                "response_template",
                Some(&record.id),
                Some(&request_id_from_headers(&headers)),
                None,
                json!({"template_key": record.template_key, "version": record.version}),
            )
            .await
            .map_err(ApiError::Internal)?;
        record
    } else {
        let mut store = state.write_store()?;
        let record = store
            .response_templates
            .get_mut(&template_id)
            .ok_or_else(|| {
                ApiError::NotFound(format!("response template {template_id} not found"))
            })?;
        if record.approved {
            return Err(ApiError::Conflict(
                "approved template versions are immutable; create a new version".to_owned(),
            ));
        }
        record.body = body;
        record.updated_by = Some(actor.user_id);
        record.updated_at = Utc::now().to_rfc3339();
        record.clone()
    };
    Ok(Json(record))
}

async fn delete_response_template(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(template_id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let actor = require_template_manager(&headers, &state.config)?;
    if let Some(repository) = state.repository() {
        repository
            .delete_response_template(&template_id)
            .await
            .map_err(template_repository_error)?;
        repository
            .audit(
                &actor.user_id,
                "DELETE_RESPONSE_TEMPLATE",
                "response_template",
                Some(&template_id),
                Some(&request_id_from_headers(&headers)),
                None,
                json!({}),
            )
            .await
            .map_err(ApiError::Internal)?;
    } else {
        let mut store = state.write_store()?;
        let record = store.response_templates.get(&template_id).ok_or_else(|| {
            ApiError::NotFound(format!("response template {template_id} not found"))
        })?;
        if record.approved {
            return Err(ApiError::Conflict(
                "approved template versions cannot be deleted".to_owned(),
            ));
        }
        store.response_templates.remove(&template_id);
    }
    Ok(StatusCode::NO_CONTENT)
}

async fn approve_response_template(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(template_id): Path<String>,
) -> Result<Json<ResponseTemplateRecord>, ApiError> {
    let actor = require_template_manager(&headers, &state.config)?;
    let record = if let Some(repository) = state.repository() {
        let record = repository
            .approve_response_template(&template_id, &actor.user_id)
            .await
            .map_err(template_repository_error)?;
        repository
            .audit(
                &actor.user_id,
                "APPROVE_RESPONSE_TEMPLATE",
                "response_template",
                Some(&record.id),
                Some(&request_id_from_headers(&headers)),
                None,
                json!({"template_key": record.template_key, "version": record.version}),
            )
            .await
            .map_err(ApiError::Internal)?;
        record
    } else {
        let mut store = state.write_store()?;
        let record = store
            .response_templates
            .get_mut(&template_id)
            .ok_or_else(|| {
                ApiError::NotFound(format!("response template {template_id} not found"))
            })?;
        if record.approved {
            return Err(ApiError::Conflict(
                "template version is already approved".to_owned(),
            ));
        }
        let timestamp = Utc::now().to_rfc3339();
        record.approved = true;
        record.approved_by = Some(actor.user_id.clone());
        record.approved_at = Some(timestamp.clone());
        record.updated_by = Some(actor.user_id);
        record.updated_at = timestamp;
        record.clone()
    };
    Ok(Json(record))
}

async fn assist_preview(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<PreviewRequest>,
) -> Result<Json<AssistPreviewResponse>, ApiError> {
    let started = Instant::now();
    let actor = require_role(
        &headers,
        &state.config,
        &[Role::Operator, Role::Manager, Role::Admin],
    )?;
    let request_id = request_id_from_headers(&headers);
    let trace_id = trace_id_from_headers(&headers);
    if request.ticket_id.is_none()
        && request
            .text
            .as_deref()
            .is_none_or(|text| text.trim().is_empty())
    {
        return Err(ApiError::BadRequest(
            "ticket_id or non-empty text is required".to_owned(),
        ));
    }
    if request.ticket_id.is_none() {
        if let Some(text) = request.text.as_deref() {
            if text.chars().count() > 10_000 {
                return Err(ApiError::BadRequest(
                    "text must be at most 10000 characters".to_owned(),
                ));
            }
            normalize_preview_language(request.language.as_deref(), text)?;
        }
    }
    if let Some(repository) = state.repository() {
        let response = repository
            .assist_preview(
                request.ticket_id.as_deref(),
                request.text.as_deref(),
                request.language.as_deref(),
                request.region_id.as_deref(),
                &request_id,
                &trace_id,
            )
            .await
            .map_err(|error| {
                if error.contains("not found") {
                    ApiError::NotFound(error)
                } else if error.contains("required") || error.contains("empty") {
                    ApiError::BadRequest(error)
                } else {
                    ApiError::Internal(error)
                }
            })?;
        repository
            .audit(
                &actor.user_id,
                "ASSIST_PREVIEW",
                "ticket",
                request.ticket_id.as_deref(),
                Some(&log_request_id_from_headers(&headers)),
                None,
                json!({"source": &response.source}),
            )
            .await
            .map_err(ApiError::Internal)?;
        return Ok(Json(response));
    }
    let store = state.read_store()?;
    let (ticket, source) = if let Some(ticket_id) = request.ticket_id {
        (
            store
                .tickets
                .get(&ticket_id)
                .cloned()
                .ok_or_else(|| ApiError::NotFound(format!("ticket {ticket_id} not found")))?,
            "stored-ticket".to_owned(),
        )
    } else {
        let text = request.text.unwrap_or_default().trim().to_owned();
        if text.chars().count() > 10_000 {
            return Err(ApiError::BadRequest(
                "text must be at most 10000 characters".to_owned(),
            ));
        }
        let language_state = normalize_preview_language(request.language.as_deref(), &text)?;
        let uncertain_language = matches!(language_state.as_str(), "MIXED" | "UNKNOWN");
        let (topic_id, topic_label, priority) = if uncertain_language {
            ("UNKNOWN", "Не определено".to_owned(), "UNKNOWN")
        } else {
            let (topic_id, _, _, priority) = demo_topic_for_text(&text);
            (topic_id, topic_label(&store.topics, topic_id), priority)
        };
        let region_id = normalize_region_id(request.region_id.unwrap_or_else(|| "R01".to_owned()));
        let region = store
            .regions
            .iter()
            .find(|region| region.id == region_id)
            .cloned()
            .ok_or_else(|| ApiError::BadRequest(format!("unknown region_id: {region_id}")))?;
        let language = match language_state.as_str() {
            "RU" => "ru",
            "KZ" => "kk",
            "MIXED" => "mixed",
            _ => "unknown",
        }
        .to_owned();
        (
            Ticket {
                id: "preview-001".to_owned(),
                external_ref: "PREVIEW-0001".to_owned(),
                text,
                language,
                region_id: region.id,
                region_name: region.name,
                topic_id: topic_id.to_owned(),
                topic_label,
                priority: priority.to_owned(),
                status: "preview".to_owned(),
                source: "preview".to_owned(),
                created_at: DEMO_TIMESTAMP.to_owned(),
                closed_at: None,
                updated_at: DEMO_TIMESTAMP.to_owned(),
            },
            "deterministic-demo".to_owned(),
        )
    };
    let language_started = Instant::now();
    let language_state = normalize_preview_language(Some(&ticket.language), &ticket.text)
        .unwrap_or_else(|_| "UNKNOWN".to_owned());
    let language_latency_ms = language_started.elapsed().as_secs_f64() * 1000.0;
    let uncertain_language = matches!(language_state.as_str(), "MIXED" | "UNKNOWN");
    let classification_started = Instant::now();
    let prediction = if uncertain_language {
        unavailable_demo_prediction(&ticket)
    } else {
        store
            .predictions
            .get(&ticket.id)
            .cloned()
            .unwrap_or_else(|| prediction_for_ticket(&ticket, &store.topics))
    };
    let classification_latency_ms = classification_started.elapsed().as_secs_f64() * 1000.0;
    let latest_decision = store
        .decisions
        .iter()
        .rev()
        .find(|decision| decision.ticket_id == ticket.id);
    let downstream_topic_id = latest_decision
        .map(|decision| decision.confirmed_topic_id.as_str())
        .unwrap_or(&prediction.topic_id);
    let related_started = Instant::now();
    let related = if uncertain_language {
        Vec::new()
    } else {
        demo_related_tickets(&store, &ticket, downstream_topic_id)
    };
    let retrieval_latency_ms = related_started.elapsed().as_secs_f64() * 1000.0;
    let duplicate_candidates = related
        .iter()
        .filter(|item| item.relation == "duplicate")
        .cloned()
        .collect();
    let repeat_candidates = related
        .iter()
        .filter(|item| item.relation == "repeat")
        .cloned()
        .collect();
    let template_started = Instant::now();
    let response_template = if uncertain_language {
        unavailable_demo_response_template(&language_state)
    } else if let Some(decision) = latest_decision {
        match decision.confirmed_service_id.as_deref() {
            Some(service_id) => response_template(
                &store,
                &ticket.language,
                &decision.confirmed_topic_id,
                service_id,
                &decision.service,
                &ticket.region_name,
            ),
            None => manual_response_template(&ticket.language),
        }
    } else {
        manual_response_template(&ticket.language)
    };
    let template_latency_ms = template_started.elapsed().as_secs_f64() * 1000.0;
    let mut model_versions = std::collections::BTreeMap::new();
    if !uncertain_language {
        model_versions.insert("classifier".to_owned(), prediction.model_version.clone());
    }
    let has_human_decision = latest_decision.is_some();
    let stages = vec![
        AssistStage {
            name: "language".to_owned(),
            status: if language_state == "UNKNOWN" {
                "unknown"
            } else {
                "completed"
            }
            .to_owned(),
            latency_ms: language_latency_ms,
            model_version: None,
            error_code: (language_state == "UNKNOWN").then(|| "LANGUAGE_UNKNOWN".to_owned()),
        },
        AssistStage {
            name: "classification".to_owned(),
            status: if uncertain_language {
                "unavailable"
            } else {
                "completed"
            }
            .to_owned(),
            latency_ms: classification_latency_ms,
            model_version: (!uncertain_language).then(|| prediction.model_version.clone()),
            error_code: uncertain_language.then(|| "LANGUAGE_UNCERTAIN".to_owned()),
        },
        AssistStage {
            name: "routing".to_owned(),
            status: if uncertain_language {
                "skipped"
            } else {
                "completed"
            }
            .to_owned(),
            latency_ms: 0.0,
            model_version: None,
            error_code: uncertain_language.then(|| "CLASSIFICATION_UNAVAILABLE".to_owned()),
        },
        AssistStage {
            name: "priority".to_owned(),
            status: if uncertain_language {
                "skipped"
            } else {
                "completed"
            }
            .to_owned(),
            latency_ms: 0.0,
            model_version: None,
            error_code: uncertain_language.then(|| "CLASSIFICATION_UNAVAILABLE".to_owned()),
        },
        AssistStage {
            name: "retrieval".to_owned(),
            status: if uncertain_language {
                "skipped"
            } else {
                "completed"
            }
            .to_owned(),
            latency_ms: retrieval_latency_ms,
            model_version: None,
            error_code: uncertain_language.then(|| "CLASSIFICATION_UNAVAILABLE".to_owned()),
        },
        AssistStage {
            name: "duplicate_repeat".to_owned(),
            status: if uncertain_language {
                "skipped"
            } else {
                "completed"
            }
            .to_owned(),
            latency_ms: 0.0,
            model_version: None,
            error_code: uncertain_language.then(|| "RETRIEVAL_UNAVAILABLE".to_owned()),
        },
        AssistStage {
            name: "response_template".to_owned(),
            status: if uncertain_language {
                "unavailable"
            } else if response_template.approved {
                "completed"
            } else {
                "manual"
            }
            .to_owned(),
            latency_ms: template_latency_ms,
            model_version: None,
            error_code: if uncertain_language {
                Some("LANGUAGE_UNCERTAIN".to_owned())
            } else if response_template.source == "MANUAL_REQUIRED" {
                Some("APPROVED_TEMPLATE_NOT_FOUND".to_owned())
            } else {
                None
            },
        },
    ];
    let partial = stages
        .iter()
        .any(|stage| matches!(stage.status.as_str(), "unavailable" | "skipped" | "unknown"));
    let needs_review = !has_human_decision
        || uncertain_language
        || prediction.confidence < 0.85
        || !response_template.approved
        || partial;
    Ok(Json(AssistPreviewResponse {
        response_template,
        ticket,
        prediction,
        similar_tickets: related,
        duplicate_candidates,
        repeat_candidates,
        source,
        orchestration: AssistOrchestration {
            request_id,
            trace_id,
            status: if partial { "partial" } else { "complete" }.to_owned(),
            needs_review,
            language: language_state,
            latency_ms: started.elapsed().as_secs_f64() * 1000.0,
            model_versions,
            stages,
        },
    }))
}

#[derive(Debug, Deserialize, Default)]
pub struct DecisionRequest {
    #[serde(alias = "ticketId")]
    pub ticket_id: Option<String>,
    #[serde(alias = "topicId")]
    pub topic_id: Option<String>,
    pub service: Option<String>,
    pub priority: Option<String>,
    pub note: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DecisionResponse {
    pub ticket: Ticket,
    pub prediction: Prediction,
    pub decision: OperatorDecision,
    pub learning_feedback_cycle_id: Option<String>,
    pub learning_feedback_status: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct PulseStateRequest {
    pub operator_confirmed_decision: Value,
    pub feedback: Option<Value>,
}

fn value_string(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| value.get(*key).and_then(Value::as_str))
        .map(str::to_owned)
}

async fn store_pulse_state(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(ticket_id): Path<String>,
    Json(request): Json<PulseStateRequest>,
) -> Result<Json<DecisionResponse>, ApiError> {
    if !request.operator_confirmed_decision.is_object() {
        return Err(ApiError::BadRequest(
            "operator_confirmed_decision must be an object".to_owned(),
        ));
    }
    let action_value = value_string(
        &request.operator_confirmed_decision,
        &["action", "status", "decision"],
    )
    .map(|value| value.to_ascii_lowercase())
    .unwrap_or_default();
    let action = if matches!(action_value.as_str(), "correct" | "corrected") {
        "correct"
    } else {
        "confirm"
    };
    let note = value_string(&request.operator_confirmed_decision, &["note", "comment"])
        .or_else(|| {
            request
                .feedback
                .as_ref()
                .and_then(|value| value_string(value, &["note", "comment"]))
        })
        .or_else(|| request.feedback.as_ref().map(Value::to_string));
    let decision_request = DecisionRequest {
        ticket_id: None,
        topic_id: value_string(
            &request.operator_confirmed_decision,
            &["topic_id", "topicId", "topic"],
        ),
        service: value_string(
            &request.operator_confirmed_decision,
            &["service", "routing"],
        ),
        priority: value_string(&request.operator_confirmed_decision, &["priority"]),
        note,
    };
    apply_decision(state, headers, ticket_id, action, decision_request).await
}

async fn confirm_ticket(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(ticket_id): Path<String>,
    Json(request): Json<DecisionRequest>,
) -> Result<Json<DecisionResponse>, ApiError> {
    apply_decision(state, headers, ticket_id, "confirm", request).await
}

async fn correct_ticket(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(ticket_id): Path<String>,
    Json(request): Json<DecisionRequest>,
) -> Result<Json<DecisionResponse>, ApiError> {
    apply_decision(state, headers, ticket_id, "correct", request).await
}

async fn confirm_ticket_from_body(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<DecisionRequest>,
) -> Result<Json<DecisionResponse>, ApiError> {
    let ticket_id = request
        .ticket_id
        .clone()
        .ok_or_else(|| ApiError::BadRequest("ticket_id is required".to_owned()))?;
    apply_decision(state, headers, ticket_id, "confirm", request).await
}

async fn correct_ticket_from_body(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<DecisionRequest>,
) -> Result<Json<DecisionResponse>, ApiError> {
    let ticket_id = request
        .ticket_id
        .clone()
        .ok_or_else(|| ApiError::BadRequest("ticket_id is required".to_owned()))?;
    apply_decision(state, headers, ticket_id, "correct", request).await
}

async fn apply_decision(
    state: AppState,
    headers: HeaderMap,
    ticket_id: String,
    action: &str,
    request: DecisionRequest,
) -> Result<Json<DecisionResponse>, ApiError> {
    let actor = require_role(
        &headers,
        &state.config,
        &[Role::Operator, Role::Manager, Role::Admin],
    )?;
    if let Some(repository) = state.repository() {
        let response = repository
            .apply_decision(&ticket_id, action, &request, &actor.user_id)
            .await
            .map_err(|error| {
                if error.contains("not found") {
                    ApiError::NotFound(error)
                } else if error.contains("required") || error.contains("unknown topic") {
                    ApiError::BadRequest(error)
                } else {
                    ApiError::Internal(error)
                }
            })?;
        repository
            .audit(
                &actor.user_id,
                "OPERATOR_DECISION",
                "ticket",
                Some(&ticket_id),
                Some(&request_id_from_headers(&headers)),
                Some(action),
                json!({"topic_id": request.topic_id, "service": request.service, "priority": request.priority}),
            )
            .await
            .map_err(ApiError::Internal)?;
        info!(
            service = SERVICE_NAME,
            ticket_id = %ticket_id,
            decision = %action,
            model_version = %response.prediction.model_version,
            request_id = %log_request_id_from_headers(&headers),
            trace_id = %log_trace_id_from_headers(&headers),
            endpoint = "/api/v1/assist/decision",
            latency_ms = 0.0_f64,
            status = 200_u16,
            error_code = "none",
            storage = "postgres",
            result_state = "recorded",
            "operator_decision_recorded"
        );
        return Ok(Json(response));
    }
    let mut store = state.write_store()?;
    let prediction = store
        .predictions
        .get(&ticket_id)
        .cloned()
        .ok_or_else(|| ApiError::NotFound(format!("prediction for {ticket_id} not found")))?;
    let topic_id = if action == "correct" {
        request.topic_id.clone().ok_or_else(|| {
            ApiError::BadRequest("topic_id is required for a correction".to_owned())
        })?
    } else {
        request
            .topic_id
            .clone()
            .unwrap_or_else(|| prediction.topic_id.clone())
    };
    if !store.topics.iter().any(|topic| topic.id == topic_id) {
        return Err(ApiError::BadRequest(format!(
            "unknown topic_id: {topic_id}"
        )));
    }
    let service_overridden = request
        .service
        .as_deref()
        .is_some_and(|service| !service.trim().is_empty());
    let priority_overridden = request
        .priority
        .as_deref()
        .is_some_and(|priority| !priority.trim().is_empty());
    let service = request
        .service
        .map(|service| {
            if service == "service_other" {
                "Другая служба".to_owned()
            } else {
                service
            }
        })
        .unwrap_or_else(|| service_for_topic(&topic_id).to_owned());
    let priority = request
        .priority
        .unwrap_or_else(|| priority_for_topic(&topic_id).to_owned());
    let confirmed_topic_label = topic_label(&store.topics, &topic_id);
    let decision_at = Utc::now().to_rfc3339();
    {
        let ticket = store
            .tickets
            .get_mut(&ticket_id)
            .ok_or_else(|| ApiError::NotFound(format!("ticket {ticket_id} not found")))?;
        ticket.updated_at = decision_at.clone();
    }
    let note = request.note;
    let decision_id = format!("decision-{:03}", store.next_decision_number);
    store.next_decision_number += 1;
    let decision = OperatorDecision {
        id: decision_id,
        ticket_id: ticket_id.clone(),
        action: action.to_owned(),
        predicted_topic_id: prediction.topic_id.clone(),
        predicted_service: Some(prediction.recommended_service.clone()),
        predicted_priority: Some(prediction.predicted_priority.clone()),
        model_version: Some(prediction.model_version.clone()),
        confirmed_topic_id: topic_id,
        confirmed_topic_label,
        confirmed_service_id: Some(service.clone()),
        service,
        priority,
        service_provenance: RuleProvenance::manual(if service_overridden {
            "Служба переопределена оператором"
        } else {
            "Служба подтверждена оператором"
        }),
        priority_provenance: RuleProvenance::manual(if priority_overridden {
            "Приоритет переопределён оператором"
        } else {
            "Приоритет подтверждён оператором"
        }),
        note,
        user_id: actor.user_id.clone(),
        created_at: decision_at.clone(),
    };
    store.decisions.push(decision.clone());
    let learning_feedback_cycle_id = store
        .learning_cycles
        .values()
        .find(|cycle| {
            cycle.state == "COLLECT"
                && !learning_collect_end_is_due(&cycle.collect_ends_at, Utc::now())
        })
        .map(|cycle| cycle.id.clone());
    if let Some(cycle_id) = learning_feedback_cycle_id.as_deref() {
        let feedback = LearningFeedback {
            id: format!("feedback-{:03}", store.next_feedback_number),
            ticket_id: ticket_id.clone(),
            cycle_id: Some(cycle_id.to_owned()),
            feedback_type: if action == "confirm" {
                "accepted".to_owned()
            } else {
                "corrected".to_owned()
            },
            comment: decision.note.clone(),
            suggestion: None,
            user_id: decision.user_id.clone(),
            created_at: decision_at.clone(),
        };
        store.next_feedback_number += 1;
        store.learning_feedback.push(feedback);
        if let Some(cycle) = store.learning_cycles.get_mut(cycle_id) {
            cycle.feedback_count += 1;
            cycle.updated_at = decision_at;
        }
    }
    let ticket = store
        .tickets
        .get(&ticket_id)
        .cloned()
        .ok_or_else(|| ApiError::NotFound(format!("ticket {ticket_id} not found")))?;
    info!(
        service = SERVICE_NAME,
        ticket_id = %ticket_id,
        decision = %action,
        model_version = %prediction.model_version,
        request_id = "n/a",
        trace_id = "n/a",
        endpoint = "/api/v1/assist/decision",
        latency_ms = 0.0_f64,
        status = 200_u16,
        error_code = "none",
        result_state = "recorded",
        "operator_decision_recorded"
    );
    Ok(Json(DecisionResponse {
        ticket,
        prediction,
        decision,
        learning_feedback_cycle_id: learning_feedback_cycle_id.clone(),
        learning_feedback_status: if learning_feedback_cycle_id.is_some() {
            "COLLECTED".to_owned()
        } else {
            "NO_ACTIVE_COLLECT_CYCLE".to_owned()
        },
    }))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AnalyticsQuery {
    pub region_id: Option<String>,
    pub topic_id: Option<String>,
    pub service_id: Option<String>,
    pub status: Option<String>,
    pub district: Option<String>,
    pub channel: Option<String>,
    pub range: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AnalyticsDrilldownQuery {
    #[serde(flatten)]
    pub filters: AnalyticsQuery,
    pub dimension: Option<String>,
    pub value: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AnalyticsDrilldownTicket {
    pub id: String,
    pub region_id: String,
    pub region_name: String,
    pub topic_id: String,
    pub topic_label: String,
    pub priority: String,
    pub status: String,
    pub created_at: String,
}

impl From<&Ticket> for AnalyticsDrilldownTicket {
    fn from(ticket: &Ticket) -> Self {
        Self {
            id: ticket.id.clone(),
            region_id: ticket.region_id.clone(),
            region_name: ticket.region_name.clone(),
            topic_id: ticket.topic_id.clone(),
            topic_label: ticket.topic_label.clone(),
            priority: ticket.priority.clone(),
            status: ticket.status.clone(),
            created_at: ticket.created_at.clone(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct AnalyticsDrilldownResponse {
    pub items: Vec<AnalyticsDrilldownTicket>,
    pub total: usize,
    pub limit: usize,
    pub offset: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct MetricBucket {
    pub id: String,
    pub label: String,
    pub tickets: usize,
    pub high_priority: usize,
    pub avg_confidence: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub change_abs: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub change_pct: Option<f32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TimeSeriesPoint {
    pub date: String,
    pub tickets: u32,
    pub resolved: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct AnalyticsResponse {
    pub generated_at: String,
    pub source: String,
    pub range: String,
    pub overview: Value,
    pub runtime_metrics: RuntimeMetrics,
    pub by_region: Vec<MetricBucket>,
    pub by_topic: Vec<MetricBucket>,
    pub time_series: Vec<TimeSeriesPoint>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RuntimeMetrics {
    pub operator_decision_time_minutes: Option<f64>,
    pub operator_decision_time_samples: usize,
    pub classification_correction_rate: Option<f64>,
    pub classification_corrections: usize,
    pub classification_decisions: usize,
    pub routing_correction_rate: Option<f64>,
    pub routing_corrections: usize,
    pub routing_decisions: usize,
    pub priority_correction_rate: Option<f64>,
    pub priority_corrections: usize,
    pub priority_decisions: usize,
    pub similarity_usefulness: Option<f64>,
    pub similarity_feedback_count: usize,
    pub duplicate_precision: Option<f64>,
    pub duplicate_feedback_count: usize,
}

pub(crate) fn metric_rate(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

#[derive(Clone, Debug, Serialize)]
struct ReportSlice {
    filters: AnalyticsQuery,
    analytics: AnalyticsResponse,
    alerts: Vec<Alert>,
    forecast: ForecastResponse,
}

fn metric_value_is_known(value: &str) -> bool {
    let normalized = value.trim();
    !normalized.is_empty()
        && !normalized.eq_ignore_ascii_case("unknown")
        && !normalized.eq_ignore_ascii_case("unavailable")
}

fn same_metric_value(left: &str, right: &str) -> bool {
    left.trim().to_lowercase() == right.trim().to_lowercase()
}

fn demo_runtime_metrics(store: &Store, filtered_tickets: &[&Ticket]) -> RuntimeMetrics {
    let includes_ticket =
        |ticket_id: &str| filtered_tickets.iter().any(|ticket| ticket.id == ticket_id);
    let decisions = store
        .decisions
        .iter()
        .filter(|decision| includes_ticket(&decision.ticket_id))
        .collect::<Vec<_>>();

    let mut first_decision_times = BTreeMap::<String, DateTime<Utc>>::new();
    for decision in &decisions {
        let Ok(decided_at) = DateTime::parse_from_rfc3339(&decision.created_at) else {
            continue;
        };
        let decided_at = decided_at.with_timezone(&Utc);
        first_decision_times
            .entry(decision.ticket_id.clone())
            .and_modify(|current| {
                if decided_at < *current {
                    *current = decided_at;
                }
            })
            .or_insert(decided_at);
    }
    let decision_durations = first_decision_times
        .iter()
        .filter_map(|(ticket_id, decided_at)| {
            let ticket = filtered_tickets
                .iter()
                .find(|ticket| ticket.id == *ticket_id)?;
            let created_at = DateTime::parse_from_rfc3339(&ticket.created_at)
                .ok()?
                .with_timezone(&Utc);
            let milliseconds = decided_at
                .signed_duration_since(created_at)
                .num_milliseconds();
            (milliseconds >= 0).then_some(milliseconds as f64 / 60_000.0)
        })
        .collect::<Vec<_>>();
    let operator_decision_time_minutes = (!decision_durations.is_empty())
        .then(|| decision_durations.iter().sum::<f64>() / decision_durations.len() as f64);

    let classification_decisions = decisions
        .iter()
        .filter(|decision| metric_value_is_known(&decision.predicted_topic_id))
        .count();
    let classification_corrections = decisions
        .iter()
        .filter(|decision| {
            metric_value_is_known(&decision.predicted_topic_id)
                && !same_metric_value(&decision.predicted_topic_id, &decision.confirmed_topic_id)
        })
        .count();
    let routing_decisions = decisions
        .iter()
        .filter(|decision| {
            decision
                .predicted_service
                .as_deref()
                .is_some_and(metric_value_is_known)
        })
        .count();
    let routing_corrections = decisions
        .iter()
        .filter(|decision| {
            decision
                .predicted_service
                .as_deref()
                .is_some_and(|predicted| {
                    metric_value_is_known(predicted)
                        && !same_metric_value(predicted, &decision.service)
                })
        })
        .count();
    let priority_decisions = decisions
        .iter()
        .filter(|decision| {
            decision
                .predicted_priority
                .as_deref()
                .is_some_and(metric_value_is_known)
        })
        .count();
    let priority_corrections = decisions
        .iter()
        .filter(|decision| {
            decision
                .predicted_priority
                .as_deref()
                .is_some_and(|predicted| {
                    metric_value_is_known(predicted)
                        && !same_metric_value(predicted, &decision.priority)
                })
        })
        .count();

    let relation_feedback = store
        .learning_feedback
        .iter()
        .filter(|feedback| includes_ticket(&feedback.ticket_id))
        .filter_map(|feedback| {
            let (_, relation_decision) = feedback.feedback_type.split_once(':')?;
            let (relation, decision) = relation_decision.rsplit_once(':')?;
            matches!(decision, "CONFIRMED" | "REJECTED").then_some((relation, decision))
        })
        .collect::<Vec<_>>();
    let similarity_confirmations = relation_feedback
        .iter()
        .filter(|(_, decision)| *decision == "CONFIRMED")
        .count();
    let duplicate_feedback = relation_feedback
        .iter()
        .filter(|(relation, _)| *relation == "DUPLICATE")
        .collect::<Vec<_>>();
    let duplicate_confirmations = duplicate_feedback
        .iter()
        .filter(|(_, decision)| *decision == "CONFIRMED")
        .count();

    RuntimeMetrics {
        operator_decision_time_minutes,
        operator_decision_time_samples: decision_durations.len(),
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
        similarity_usefulness: metric_rate(similarity_confirmations, relation_feedback.len()),
        similarity_feedback_count: relation_feedback.len(),
        duplicate_precision: metric_rate(duplicate_confirmations, duplicate_feedback.len()),
        duplicate_feedback_count: duplicate_feedback.len(),
    }
}

async fn analytics(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<AnalyticsQuery>,
) -> Result<Json<AnalyticsResponse>, ApiError> {
    require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    if let Some(repository) = state.repository() {
        return repository
            .analytics(&query)
            .await
            .map(Json)
            .map_err(ApiError::Internal);
    }
    let store = state.read_store()?;
    Ok(Json(memory_analytics_response(&store, &query)?))
}

fn memory_analytics_response(
    store: &Store,
    query: &AnalyticsQuery,
) -> Result<AnalyticsResponse, ApiError> {
    let generated_at = demo_analytics_as_of(store)?;
    let days = analytics_range_days(query.range.as_deref());
    let current_since = generated_at - Duration::days(days);
    let previous_since = current_since - Duration::days(days);
    let filtered: Vec<&Ticket> = store
        .tickets
        .values()
        .filter(|ticket| {
            ticket_matches_analytics_filters(ticket, query)
                && ticket_created_at(ticket).is_some_and(|created_at| {
                    created_at >= current_since && created_at <= generated_at
                })
        })
        .collect();
    let previous: Vec<&Ticket> = store
        .tickets
        .values()
        .filter(|ticket| {
            ticket_matches_analytics_filters(ticket, query)
                && ticket_created_at(ticket).is_some_and(|created_at| {
                    created_at >= previous_since && created_at < current_since
                })
        })
        .collect();
    let avg_confidence = average_ticket_confidence(&filtered, store);
    let open_tickets = filtered
        .iter()
        .filter(|ticket| is_open_ticket_status(&ticket.status))
        .count();
    let resolved = filtered
        .iter()
        .filter(|ticket| is_resolved_ticket_status(&ticket.status))
        .count();
    let high_priority = filtered
        .iter()
        .filter(|ticket| is_high_priority(&ticket.priority))
        .count();
    let runtime_metrics = demo_runtime_metrics(store, &filtered);
    let total_change = filtered.len() as i64 - previous.len() as i64;
    let overview = json!({
        "total_tickets": filtered.len(),
        "open_tickets": open_tickets,
        "resolved_tickets": resolved,
        "high_priority_tickets": high_priority,
        "operator_decisions": store.decisions.iter().filter(|decision| filtered.iter().any(|ticket| ticket.id == decision.ticket_id)).count(),
        "confirmed_decisions": store.decisions.iter().filter(|decision| decision.action == "confirm" && filtered.iter().any(|ticket| ticket.id == decision.ticket_id)).count(),
        "corrected_decisions": store.decisions.iter().filter(|decision| decision.action == "correct" && filtered.iter().any(|ticket| ticket.id == decision.ticket_id)).count(),
        "avg_decision_minutes": runtime_metrics.operator_decision_time_minutes,
        "average_confidence": avg_confidence,
        "previous_total_tickets": previous.len(),
        "change_abs": total_change,
        "change_pct": analytics_percent_change(filtered.len(), previous.len()),
    });
    let by_region = store
        .regions
        .iter()
        .filter(|region| {
            query
                .region_id
                .as_deref()
                .is_none_or(|id| id.trim() == region.id)
        })
        .map(|region| {
            metric_bucket(
                region.id.clone(),
                region.name.clone(),
                filtered
                    .iter()
                    .filter(|ticket| ticket.region_id == region.id)
                    .copied()
                    .collect(),
                previous
                    .iter()
                    .filter(|ticket| ticket.region_id == region.id)
                    .count(),
                store,
            )
        })
        .collect();
    let by_topic = store
        .topics
        .iter()
        .filter(|topic| {
            query
                .topic_id
                .as_deref()
                .is_none_or(|id| id.trim() == topic.id)
        })
        .map(|topic| {
            metric_bucket(
                topic.id.clone(),
                topic.label.clone(),
                filtered
                    .iter()
                    .filter(|ticket| ticket.topic_id == topic.id)
                    .copied()
                    .collect(),
                previous
                    .iter()
                    .filter(|ticket| ticket.topic_id == topic.id)
                    .count(),
                store,
            )
        })
        .collect();
    let time_series = demo_time_series(&filtered, current_since, generated_at);
    Ok(AnalyticsResponse {
        generated_at: generated_at.to_rfc3339(),
        source: "deterministic-demo".to_owned(),
        range: query.range.clone().unwrap_or_else(|| format!("{days}d")),
        overview,
        runtime_metrics,
        by_region,
        by_topic,
        time_series,
    })
}

async fn analytics_drilldown(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<AnalyticsDrilldownQuery>,
) -> Result<Json<AnalyticsDrilldownResponse>, ApiError> {
    require_role(
        &headers,
        &state.config,
        &[Role::Operator, Role::Manager, Role::Admin],
    )?;
    let dimension = query
        .dimension
        .as_deref()
        .unwrap_or("overview")
        .trim()
        .to_ascii_lowercase();
    if !matches!(
        dimension.as_str(),
        "overview" | "region" | "topic" | "date" | "alert"
    ) {
        return Err(ApiError::BadRequest(format!(
            "unsupported drilldown dimension: {dimension}"
        )));
    }
    if dimension != "overview" && query.value.as_deref().is_none_or(str::is_empty) {
        return Err(ApiError::BadRequest(
            "drilldown value is required for this dimension".to_owned(),
        ));
    }
    if let Some(repository) = state.repository() {
        return repository
            .analytics_drilldown(&query, &dimension)
            .await
            .map(Json)
            .map_err(ApiError::Internal);
    }
    let store = state.read_store()?;
    let generated_at = demo_analytics_as_of(&store)?;
    let days = analytics_range_days(query.filters.range.as_deref());
    let current_since = generated_at - Duration::days(days);
    let value = query.value.as_deref().unwrap_or_default().trim();
    let selected_date = if dimension == "date" {
        Some(
            NaiveDate::parse_from_str(value, "%Y-%m-%d").map_err(|error| {
                ApiError::BadRequest(format!("invalid drilldown date: {error}"))
            })?,
        )
    } else {
        None
    };
    let selected_alert_ticket_ids = if dimension == "alert" {
        Some(
            store
                .alerts
                .get(value)
                .map(|alert| {
                    alert
                        .linked_ticket_ids
                        .iter()
                        .map(String::as_str)
                        .collect::<std::collections::BTreeSet<_>>()
                })
                .unwrap_or_default(),
        )
    } else {
        None
    };
    let mut items = store
        .tickets
        .values()
        .filter(|ticket| {
            ticket_matches_analytics_filters(ticket, &query.filters)
                && ticket_created_at(ticket).is_some_and(|created_at| {
                    created_at >= current_since
                        && created_at <= generated_at
                        && selected_date.is_none_or(|date| created_at.date_naive() == date)
                })
                && match dimension.as_str() {
                    "region" => ticket.region_id == value.trim(),
                    "topic" => ticket.topic_id == value.trim(),
                    "alert" => selected_alert_ticket_ids
                        .as_ref()
                        .is_some_and(|ids| ids.contains(ticket.id.as_str())),
                    "overview" => match value.to_ascii_lowercase().as_str() {
                        "high_priority" => is_high_priority(&ticket.priority),
                        "open" => is_open_ticket_status(&ticket.status),
                        "resolved" => is_resolved_ticket_status(&ticket.status),
                        "" | "all" => true,
                        _ => false,
                    },
                    _ => true,
                }
        })
        .cloned()
        .collect::<Vec<_>>();
    let total = items.len();
    let limit = query.limit.unwrap_or(100).clamp(1, 100);
    let offset = query.offset.unwrap_or(0);
    items = items.into_iter().skip(offset).take(limit).collect();
    Ok(Json(AnalyticsDrilldownResponse {
        items: items.iter().map(AnalyticsDrilldownTicket::from).collect(),
        total,
        limit,
        offset,
    }))
}

async fn taxonomy(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    require_role(
        &headers,
        &state.config,
        &[Role::Operator, Role::Manager, Role::Admin],
    )?;
    if let Some(repository) = state.repository() {
        return repository
            .taxonomy()
            .await
            .map(Json)
            .map_err(ApiError::Internal);
    }
    let store = state.read_store()?;
    Ok(Json(json!({
        "topics": store.topics,
        "services": store.topics.iter().map(|topic| json!({"id": service_for_topic(&topic.id), "label": service_for_topic(&topic.id)})).collect::<Vec<_>>(),
        "regions": store.regions.iter().map(|region| json!({"id": region.id, "label": region.name})).collect::<Vec<_>>(),
        "statuses": store.tickets.values().map(|ticket| ticket.status.clone()).collect::<std::collections::BTreeSet<_>>().into_iter().map(|value| json!({"id": value, "label": value})).collect::<Vec<_>>(),
        "districts": [],
        "channels": [],
        "source": "memory"
    })))
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct QueryIntentFilters {
    pub region_id: Option<String>,
    pub topic_id: Option<String>,
    pub service_id: Option<String>,
    pub status: Option<String>,
    pub district: Option<String>,
    pub channel: Option<String>,
    pub range: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct QueryIntentRequest {
    pub intent: Option<String>,
    /// Optional natural-language question.  Core maps it to the small
    /// allow-list below; no free-form SQL or model-generated query reaches
    /// the repository.
    #[serde(alias = "query")]
    pub text: Option<String>,
    pub region_id: Option<String>,
    pub topic_id: Option<String>,
    pub range: Option<String>,
    pub filters: Option<QueryIntentFilters>,
    pub group_by: Option<String>,
    pub limit: Option<usize>,
    pub horizon_days: Option<u32>,
}

const QUERY_SPIKE_MIN_COUNT: u64 = 3;
const QUERY_SPIKE_MIN_RATIO: f64 = 1.5;

fn infer_query_intent(text: &str) -> Option<&'static str> {
    let value = text.trim().to_lowercase();
    if value.is_empty() {
        return None;
    }
    if value.contains("прогноз") || value.contains("forecast") {
        Some("forecast")
    } else if value.contains("всплес")
        || value.contains("аномал")
        || value.contains("пик")
        || value.contains("spike")
    {
        Some("spikes")
    } else if value.contains("тренд")
        || value.contains("динамик")
        || value.contains("измен")
        || value.contains("trend")
    {
        Some("trend")
    } else if value.contains("регион")
        || value.contains("област")
        || value.contains("сравн")
        || value.contains("region")
    {
        Some("compare_regions")
    } else if value.contains("тем") || value.contains("категор") || value.contains("topic")
    {
        Some("top_topics")
    } else if value.contains("сколько")
        || value.contains("колич")
        || value.contains("count")
        || value.contains("how many")
    {
        Some("count")
    } else {
        None
    }
}

fn resolved_query_intent(query: &QueryIntentRequest) -> Option<String> {
    query
        .intent
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .or_else(|| {
            query
                .text
                .as_deref()
                .and_then(infer_query_intent)
                .map(str::to_owned)
        })
}

fn validate_query_intent(query: &QueryIntentRequest) -> Result<String, String> {
    let explicit_intent = query
        .intent
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let question = query
        .text
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if explicit_intent.is_some() == question.is_some() {
        return Err("provide exactly one allow-listed intent or text question".to_owned());
    }
    if question.is_some_and(|value| value.chars().count() > 500) {
        return Err("text question exceeds 500 characters".to_owned());
    }
    let intent = resolved_query_intent(query)
        .ok_or_else(|| "unsupported or empty QueryIntent question".to_owned())?
        .to_ascii_lowercase();
    let intent = match intent.as_str() {
        "count" | "trend" | "compare_regions" | "top_topics" | "spikes" | "forecast" => intent,
        "compare" => "compare_regions".to_owned(),
        _ => return Err(format!("unsupported QueryIntent: {intent}")),
    };
    if query.limit.is_some_and(|limit| !(1..=100).contains(&limit)) {
        return Err("limit must be between 1 and 100".to_owned());
    }
    if query
        .horizon_days
        .is_some_and(|horizon| !matches!(horizon, 30 | 60 | 90))
    {
        return Err("horizon_days must be 30, 60 or 90".to_owned());
    }
    if query.horizon_days.is_some() && intent != "forecast" {
        return Err("horizon_days is only valid for forecast intent".to_owned());
    }
    resolved_query_grouping(&intent, query.group_by.as_deref())?;
    Ok(intent)
}

fn merge_query_filter(
    top_level: Option<&String>,
    nested: Option<&String>,
    field: &str,
) -> Result<Option<String>, String> {
    let normalize = |value: Option<&String>| {
        value
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    };
    let top_level = normalize(top_level);
    let nested = normalize(nested);
    match (top_level, nested) {
        (Some(left), Some(right)) if left != right => {
            Err(format!("conflicting top-level and filters.{field} values"))
        }
        (Some(value), _) | (_, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}

pub(crate) fn query_analytics_filters(
    query: &QueryIntentRequest,
) -> Result<AnalyticsQuery, String> {
    let nested = query.filters.as_ref();
    let region_id = merge_query_filter(
        query.region_id.as_ref(),
        nested.and_then(|filters| filters.region_id.as_ref()),
        "region_id",
    )?;
    let topic_id = merge_query_filter(
        query.topic_id.as_ref(),
        nested.and_then(|filters| filters.topic_id.as_ref()),
        "topic_id",
    )?;
    let range = merge_query_filter(
        query.range.as_ref(),
        nested.and_then(|filters| filters.range.as_ref()),
        "range",
    )?
    .unwrap_or_else(|| "30d".to_owned());
    let days = range
        .strip_suffix('d')
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|days| (1..=366).contains(days))
        .ok_or_else(|| "range must be a day period from 1d through 366d".to_owned())?;
    Ok(AnalyticsQuery {
        region_id,
        topic_id,
        service_id: nested
            .and_then(|filters| filters.service_id.as_ref())
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .map(str::to_owned),
        status: nested
            .and_then(|filters| filters.status.as_ref())
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .map(str::to_owned),
        district: nested
            .and_then(|filters| filters.district.as_ref())
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .map(str::to_owned),
        channel: nested
            .and_then(|filters| filters.channel.as_ref())
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .map(str::to_owned),
        range: Some(format!("{days}d")),
    })
}

fn resolved_query_grouping(intent: &str, requested: Option<&str>) -> Result<&'static str, String> {
    let requested = requested.map(str::trim).filter(|value| !value.is_empty());
    let default = match intent {
        "compare_regions" => "region",
        "top_topics" => "topic",
        "trend" | "spikes" | "forecast" => "day",
        _ => "none",
    };
    let grouping = requested.unwrap_or(default).to_ascii_lowercase();
    let valid = match intent {
        "compare_regions" => grouping == "region",
        "top_topics" => grouping == "topic",
        "trend" | "spikes" => matches!(grouping.as_str(), "day" | "week" | "month"),
        "forecast" => grouping == "day",
        "count" => matches!(
            grouping.as_str(),
            "none" | "region" | "topic" | "day" | "week" | "month"
        ),
        _ => false,
    };
    if !valid {
        return Err(format!(
            "group_by is not supported for {intent}: {grouping}"
        ));
    }
    match grouping.as_str() {
        "none" => Ok("none"),
        "region" => Ok("region"),
        "topic" => Ok("topic"),
        "day" => Ok("day"),
        "week" => Ok("week"),
        "month" => Ok("month"),
        _ => Err(format!("unsupported group_by: {grouping}")),
    }
}

#[derive(Clone, Debug)]
struct QueryTimeBucket {
    date: String,
    label: String,
    count: u64,
    resolved: u64,
}

fn query_time_buckets(points: &[TimeSeriesPoint], grouping: &str) -> Vec<QueryTimeBucket> {
    if grouping == "day" {
        return points
            .iter()
            .map(|point| QueryTimeBucket {
                date: point.date.clone(),
                label: point.date.clone(),
                count: u64::from(point.tickets),
                resolved: u64::from(point.resolved),
            })
            .collect();
    }
    let mut buckets = BTreeMap::<String, QueryTimeBucket>::new();
    for point in points {
        let Ok(date) = NaiveDate::parse_from_str(&point.date, "%Y-%m-%d") else {
            continue;
        };
        let (key, bucket_date) = match grouping {
            "week" => {
                let week = date.iso_week();
                let monday =
                    date - Duration::days(i64::from(date.weekday().num_days_from_monday()));
                (
                    format!("{}-W{:02}", week.year(), week.week()),
                    monday.to_string(),
                )
            }
            "month" => (
                format!("{}-{:02}", date.year(), date.month()),
                format!("{}-{:02}-01", date.year(), date.month()),
            ),
            _ => continue,
        };
        let bucket = buckets
            .entry(key.clone())
            .or_insert_with(|| QueryTimeBucket {
                date: bucket_date,
                label: key,
                count: 0,
                resolved: 0,
            });
        bucket.count += u64::from(point.tickets);
        bucket.resolved += u64::from(point.resolved);
    }
    buckets.into_values().collect()
}

fn metric_query_rows(buckets: &[MetricBucket]) -> Vec<Value> {
    buckets
        .iter()
        .map(|bucket| {
            json!({
                "key": bucket.id,
                "label": bucket.label,
                "count": bucket.tickets,
                "tickets": bucket.tickets,
                "change_abs": bucket.change_abs,
                "change_pct": bucket.change_pct,
            })
        })
        .collect()
}

pub(crate) fn build_query_intent_result(
    query: &QueryIntentRequest,
    filters: &AnalyticsQuery,
    report: &AnalyticsResponse,
    forecast: Option<&ForecastResponse>,
) -> Result<Value, String> {
    let intent = validate_query_intent(query)?;
    let grouping = resolved_query_grouping(&intent, query.group_by.as_deref())?;
    let total = report
        .overview
        .get("total_tickets")
        .and_then(Value::as_i64)
        .unwrap_or_default()
        .max(0);
    let previous_total = report
        .overview
        .get("previous_total_tickets")
        .and_then(Value::as_i64)
        .unwrap_or_default()
        .max(0);
    let range = filters.range.as_deref().unwrap_or("30d");
    let range_days = range
        .strip_suffix('d')
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(30);
    let generated_at = DateTime::parse_from_rfc3339(&report.generated_at)
        .map(|value| value.with_timezone(&Utc))
        .map_err(|error| format!("invalid analytics generated_at: {error}"))?;
    let period_start = generated_at - Duration::days(range_days);
    let comparison_start = period_start - Duration::days(range_days);
    let time_buckets = query_time_buckets(&report.time_series, grouping);

    let forecast = if intent == "forecast" {
        Some(forecast.ok_or_else(|| "forecast result is unavailable".to_owned())?)
    } else {
        None
    };
    let mut rows = match intent.as_str() {
        "count" if grouping == "region" => metric_query_rows(&report.by_region),
        "count" if grouping == "topic" => metric_query_rows(&report.by_topic),
        "count" if matches!(grouping, "day" | "week" | "month") => time_buckets
            .iter()
            .map(|bucket| {
                json!({"date": bucket.date, "period": bucket.label, "label": bucket.label, "count": bucket.count, "tickets": bucket.count})
            })
            .collect(),
        "count" => vec![json!({
            "period": range,
            "label": range,
            "count": total,
            "tickets": total,
        })],
        "trend" => time_buckets
            .iter()
            .map(|bucket| {
                json!({"date": bucket.date, "period": bucket.label, "label": bucket.label, "count": bucket.count, "tickets": bucket.count, "resolved": bucket.resolved})
            })
            .collect(),
        "compare_regions" => metric_query_rows(&report.by_region),
        "top_topics" => metric_query_rows(&report.by_topic),
        "spikes" => time_buckets
            .windows(2)
            .filter(|window| {
                window[1].count >= QUERY_SPIKE_MIN_COUNT
                    && (window[1].count as f64)
                        > window[0].count as f64 * QUERY_SPIKE_MIN_RATIO
            })
            .map(|window| {
                json!({
                    "date": window[1].date,
                    "period": window[1].label,
                    "label": window[1].label,
                    "count": window[1].count,
                    "baseline": window[0].count,
                    "deviation": window[1].count as i64 - window[0].count as i64,
                    "is_spike": true,
                })
            })
            .collect(),
        "forecast" => forecast
            .expect("forecast presence was validated")
            .points
            .iter()
            .map(|point| {
                json!({"date": point.date, "period": point.date, "label": point.date, "count": point.tickets, "tickets": point.tickets})
            })
            .collect(),
        _ => return Err(format!("unsupported QueryIntent: {intent}")),
    };
    let matching_row_count = rows.len();
    if let Some(limit) = query.limit {
        rows.truncate(limit);
    }

    let mut series = if intent == "spikes" {
        time_buckets
            .iter()
            .enumerate()
            .map(|(index, bucket)| {
                let baseline = index
                    .checked_sub(1)
                    .and_then(|previous| time_buckets.get(previous))
                    .map(|previous| previous.count);
                let is_spike = baseline.is_some_and(|baseline| {
                    bucket.count >= QUERY_SPIKE_MIN_COUNT
                        && (bucket.count as f64) > baseline as f64 * QUERY_SPIKE_MIN_RATIO
                });
                json!({
                    "date": bucket.date,
                    "label": bucket.label,
                    "count": bucket.count,
                    "baseline": baseline,
                    "is_spike": is_spike,
                })
            })
            .collect()
    } else {
        rows.clone()
    };
    if let Some(forecast) = forecast {
        series = forecast
            .history
            .iter()
            .map(|point| {
                json!({"date": point.date, "label": point.date, "count": point.tickets, "segment": "history"})
            })
            .chain(forecast.points.iter().map(|point| {
                json!({"date": point.date, "label": point.date, "count": point.tickets, "segment": "forecast"})
            }))
            .collect();
    }

    let (number, summary_label, summary_text) = if let Some(forecast) = forecast {
        let predicted_total = forecast
            .points
            .iter()
            .map(|point| i64::from(point.tickets))
            .sum::<i64>();
        let value = if forecast.insufficient_history {
            Value::Null
        } else {
            json!(predicted_total)
        };
        (
            value,
            "Ожидаемый объём".to_owned(),
            if forecast.insufficient_history {
                format!(
                    "Недостаточно истории для прогноза на {} дней.",
                    forecast.horizon_days
                )
            } else {
                format!(
                    "Ожидаемый объём за {} дней по модели {}.",
                    forecast.horizon_days, forecast.model
                )
            },
        )
    } else if intent == "spikes" {
        let count = matching_row_count;
        (
            json!(count),
            "Периоды необычного роста".to_owned(),
            format!("Найдено {count} периодов по сравнению с предыдущим равным интервалом."),
        )
    } else {
        (
            json!(total),
            "Обращения в текущем срезе".to_owned(),
            format!("{total} обращений за {range} по выбранным фильтрам."),
        )
    };
    let comparison_change = total - previous_total;
    let comparison = json!({
        "type": "previous_equal_length_period",
        "current_total": total,
        "previous_total": previous_total,
        "change_abs": comparison_change,
        "change_pct": report.overview.get("change_pct").cloned().unwrap_or(Value::Null),
        "current_start": period_start.to_rfc3339(),
        "current_end": generated_at.to_rfc3339(),
        "previous_start": comparison_start.to_rfc3339(),
        "previous_end": period_start.to_rfc3339(),
    });
    let grouping_label = match grouping {
        "none" => "без группировки",
        "region" => "регион",
        "topic" => "тема",
        "day" => "день",
        "week" => "неделя",
        "month" => "месяц",
        _ => "не задано",
    };
    let columns: Vec<Value> = match intent.as_str() {
        "compare_regions" => vec![
            json!({"key":"label","label":"Регион"}),
            json!({"key":"count","label":"Обращения"}),
            json!({"key":"change_abs","label":"Изменение"}),
            json!({"key":"change_pct","label":"Изменение, %"}),
        ],
        "top_topics" => vec![
            json!({"key":"label","label":"Тема"}),
            json!({"key":"count","label":"Обращения"}),
            json!({"key":"change_abs","label":"Изменение"}),
            json!({"key":"change_pct","label":"Изменение, %"}),
        ],
        "spikes" => vec![
            json!({"key":"label","label":"Период"}),
            json!({"key":"count","label":"Обращения"}),
            json!({"key":"baseline","label":"Предыдущий период"}),
            json!({"key":"deviation","label":"Разница"}),
        ],
        "forecast" => vec![
            json!({"key":"date","label":"Дата"}),
            json!({"key":"count","label":"Ожидаемые обращения"}),
        ],
        _ => vec![
            json!({"key":"label","label":"Период"}),
            json!({"key":"count","label":"Обращения"}),
            json!({"key":"resolved","label":"Закрыто"}),
        ],
    };
    let chart_type = if matches!(intent.as_str(), "trend" | "spikes" | "forecast")
        || (intent == "count" && matches!(grouping, "day" | "week" | "month"))
    {
        "line"
    } else {
        "bar"
    };
    let horizon = forecast.map(|value| value.horizon_days);
    let chart_boundary = forecast.and_then(|value| value.forecast_start.clone());
    let filters_json = json!({
        "region_id": filters.region_id,
        "topic_id": filters.topic_id,
        "service_id": filters.service_id,
        "status": filters.status,
        "district": filters.district,
        "channel": filters.channel,
        "range": range,
    });
    let interpreted_filters = json!({
        "region_id": filters.region_id,
        "topic_id": filters.topic_id,
        "service_id": filters.service_id,
        "status": filters.status,
        "district": filters.district,
        "channel": filters.channel,
        "range": range,
        "group_by": grouping,
        "limit": query.limit,
        "horizon_days": horizon,
    });
    Ok(json!({
        "intent": intent,
        "summary": {"label": summary_label, "value": number, "text": summary_text},
        "number": number,
        "rows": rows,
        "table": rows,
        "table_columns": columns,
        "series": series,
        "chart": {
            "type": chart_type,
            "x": if grouping == "day" { "date" } else { "label" },
            "y": "count",
            "title": summary_label,
            "forecast_start": chart_boundary,
        },
        "filters": filters_json,
        "interpreted_filters": interpreted_filters,
        "period": {
            "range": range,
            "days": range_days,
            "start": period_start.to_rfc3339(),
            "end": generated_at.to_rfc3339(),
        },
        "grouping": grouping_label,
        "comparison": comparison,
        "comparison_definition": if intent == "spikes" {
            format!("Текущий период сравнивается с предыдущим равным периодом. Всплеск: не менее {QUERY_SPIKE_MIN_COUNT} обращений и объём выше предыдущего периода более чем в {QUERY_SPIKE_MIN_RATIO} раза.")
        } else {
            "Текущий период сопоставляется с предыдущим периодом такой же длительности; для регионов и тем также показано изменение по группам.".to_owned()
        },
        "generated_at": report.generated_at,
        "source": forecast.map(|value| value.source.as_str()).unwrap_or(&report.source),
        "forecast_status": forecast.map(|value| value.status.as_str()),
        "forecast_insufficient_history": forecast.map(|value| value.insufficient_history),
        "forecast_start": forecast.and_then(|value| value.forecast_start.clone()),
        "forecast_model_version": forecast.map(|value| value.model_version.as_str()),
        "forecast_model": forecast.map(|value| value.model.as_str()),
        "forecast_horizon_days": horizon,
        "forecast_history": forecast.map(|value| &value.history),
        "forecast_points": forecast.map(|value| &value.points),
        "expected_peaks": forecast.map(|value| &value.expected_peaks),
        "backtest": forecast.map(|value| &value.backtest),
    }))
}

/// Execute a deliberately small allow-listed analytics language.  The public
/// endpoint accepts an intent object, never SQL, so an optional LLM adapter can
/// only propose parameters that Core validates before reading the store.
async fn analytics_query(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(query): Json<QueryIntentRequest>,
) -> Result<Json<Value>, ApiError> {
    require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    let mut query = query;
    let intent = validate_query_intent(&query).map_err(ApiError::BadRequest)?;
    query.intent = Some(intent.clone());
    query.text = None;
    let filters = query_analytics_filters(&query).map_err(ApiError::BadRequest)?;
    if state.repository().is_none() && (filters.district.is_some() || filters.channel.is_some()) {
        return Err(ApiError::BadRequest(
            "district and channel filters are unavailable in the memory demo source".to_owned(),
        ));
    }
    if let Some(repository) = state.repository() {
        return repository
            .analytics_query(&query)
            .await
            .map(Json)
            .map_err(ApiError::Internal);
    }
    let store = state.read_store()?;
    let report = memory_analytics_response(&store, &filters)?;
    let forecast = if intent == "forecast" {
        Some(memory_forecast_response(
            &store,
            &ForecastQuery {
                horizon: Some(query.horizon_days.unwrap_or(30)),
                horizon_days: None,
                region_id: filters.region_id.clone(),
                topic_id: filters.topic_id.clone(),
                service_id: filters.service_id.clone(),
                status: filters.status.clone(),
                district: filters.district.clone(),
                channel: filters.channel.clone(),
            },
            query.horizon_days.unwrap_or(30),
        )?)
    } else {
        None
    };
    let result = build_query_intent_result(&query, &filters, &report, forecast.as_ref())
        .map_err(ApiError::Internal)?;
    Ok(Json(result))
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = u32::MAX;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xedb88320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

/// Build a small uncompressed ZIP archive for a dependency-free XLSX export.
fn zip_store(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut archive = Vec::new();
    let mut offsets = Vec::with_capacity(entries.len());
    for (name, data) in entries {
        let name_bytes = name.as_bytes();
        let checksum = crc32(data);
        offsets.push(archive.len() as u32);
        push_u32(&mut archive, 0x04034b50);
        push_u16(&mut archive, 20);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u32(&mut archive, checksum);
        push_u32(&mut archive, data.len() as u32);
        push_u32(&mut archive, data.len() as u32);
        push_u16(&mut archive, name_bytes.len() as u16);
        push_u16(&mut archive, 0);
        archive.extend_from_slice(name_bytes);
        archive.extend_from_slice(data);
    }

    let central_offset = archive.len() as u32;
    for ((name, data), local_offset) in entries.iter().zip(offsets.iter().copied()) {
        let name_bytes = name.as_bytes();
        let checksum = crc32(data);
        push_u32(&mut archive, 0x02014b50);
        push_u16(&mut archive, 20);
        push_u16(&mut archive, 20);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u32(&mut archive, checksum);
        push_u32(&mut archive, data.len() as u32);
        push_u32(&mut archive, data.len() as u32);
        push_u16(&mut archive, name_bytes.len() as u16);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u16(&mut archive, 0);
        push_u32(&mut archive, 0);
        push_u32(&mut archive, local_offset);
        archive.extend_from_slice(name_bytes);
    }

    let central_size = archive.len() as u32 - central_offset;
    push_u32(&mut archive, 0x06054b50);
    push_u16(&mut archive, 0);
    push_u16(&mut archive, 0);
    push_u16(&mut archive, entries.len() as u16);
    push_u16(&mut archive, entries.len() as u16);
    push_u32(&mut archive, central_size);
    push_u32(&mut archive, central_offset);
    push_u16(&mut archive, 0);
    archive
}

async fn build_report_slice(
    repository: &PgRepository,
    query: &AnalyticsQuery,
) -> Result<ReportSlice, String> {
    let analytics = repository.analytics(query).await?;
    let alerts = repository.list_alerts_for_analytics(query).await?;
    let forecast = repository
        .forecast(&ForecastQuery {
            horizon: Some(30),
            horizon_days: None,
            region_id: query.region_id.clone(),
            topic_id: query.topic_id.clone(),
            service_id: query.service_id.clone(),
            status: query.status.clone(),
            district: query.district.clone(),
            channel: query.channel.clone(),
        })
        .await?;
    Ok(ReportSlice {
        filters: query.clone(),
        analytics,
        alerts,
        forecast,
    })
}

fn memory_report_slice(store: &Store, query: &AnalyticsQuery) -> Result<ReportSlice, ApiError> {
    let analytics = memory_analytics_response(store, query)?;
    let generated_at = demo_analytics_as_of(store)?;
    let period_start = generated_at - Duration::days(analytics_range_days(query.range.as_deref()));
    let filtered_ticket_ids = store
        .tickets
        .values()
        .filter(|ticket| {
            ticket_matches_analytics_filters(ticket, query)
                && ticket_created_at(ticket).is_some_and(|created_at| {
                    created_at >= period_start && created_at <= generated_at
                })
        })
        .map(|ticket| ticket.id.clone())
        .collect::<std::collections::BTreeSet<_>>();
    let alerts = store
        .alerts
        .values()
        .filter(|alert| !alert.status.eq_ignore_ascii_case("CLOSED"))
        .filter_map(|alert| {
            let ticket_ids = alert
                .linked_ticket_ids
                .iter()
                .filter(|ticket_id| filtered_ticket_ids.contains(*ticket_id))
                .cloned()
                .collect::<Vec<_>>();
            if ticket_ids.is_empty() {
                return None;
            }
            let mut alert = alert.clone();
            alert.ticket_count = ticket_ids.len().min(u32::MAX as usize) as u32;
            alert.linked_ticket_ids = ticket_ids;
            Some(alert)
        })
        .collect();
    Ok(ReportSlice {
        filters: query.clone(),
        analytics,
        alerts,
        forecast: ForecastResponse {
            source: "deterministic-demo".to_owned(),
            model_version: "not-run".to_owned(),
            model: "not-available".to_owned(),
            status: "DEMO_ONLY".to_owned(),
            insufficient_history: true,
            horizon_days: 30,
            history: Vec::new(),
            forecast_start: None,
            points: Vec::new(),
            expected_peaks: Vec::new(),
            backtest: json!({"status": "DEMO_ONLY", "reason": "memory report has no forecast history"}),
        },
    })
}

fn report_metric(report: Option<&ReportSlice>, key: &str) -> String {
    report
        .map(|value| &value.analytics)
        .and_then(|value| value.overview.get(key))
        .map(|value| {
            value.as_str().map(str::to_owned).unwrap_or_else(|| {
                if value.is_null() {
                    "—".to_owned()
                } else {
                    value.to_string()
                }
            })
        })
        .unwrap_or_else(|| "0".to_owned())
}

fn report_range(report: Option<&ReportSlice>) -> String {
    report
        .map(|value| value.analytics.range.clone())
        .unwrap_or_else(|| "demo".to_owned())
}

fn report_filter_summary(report: &ReportSlice) -> String {
    let filters = &report.filters;
    let mut summary = vec![format!(
        "period={}",
        filters.range.as_deref().unwrap_or("30d")
    )];
    for (name, value) in [
        ("region", filters.region_id.as_deref()),
        ("topic", filters.topic_id.as_deref()),
        ("service", filters.service_id.as_deref()),
        ("status", filters.status.as_deref()),
        ("district", filters.district.as_deref()),
        ("channel", filters.channel.as_deref()),
    ] {
        if let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) {
            summary.push(format!("{name}={value}"));
        }
    }
    summary.join("; ")
}

fn report_change_label(change_abs: Option<i64>, change_pct: Option<f64>) -> String {
    let Some(change_abs) = change_abs else {
        return "—".to_owned();
    };
    let absolute = format!("{change_abs:+}");
    change_pct
        .map(|change_pct| format!("{absolute} ({change_pct:+.1}%)"))
        .unwrap_or_else(|| format!("{absolute} (previous period: 0)"))
}

fn report_previous_count(current: usize, change_abs: Option<i64>) -> usize {
    change_abs
        .map(|change| (current as i64 - change).max(0) as usize)
        .unwrap_or(0)
}

/// Canonical HTML template for a report slice.
///
/// Both export formats consume this same slice; the PDF renderer lays out this
/// HTML while the spreadsheet keeps the underlying values in typed columns.
const REPORT_CHART_WIDTH: usize = 760;
const REPORT_TIME_CHART_HEIGHT: usize = 230;
const REPORT_BAR_ROW_HEIGHT: usize = 34;
const REPORT_MAX_BAR_ROWS: usize = 12;

fn report_time_series_svg(series: &[TimeSeriesPoint]) -> String {
    if series.is_empty() {
        return "<p class=\"empty\">Нет точек временного ряда за выбранный период.</p>".to_owned();
    }

    let left = 56.0;
    let right = 18.0;
    let top = 18.0;
    let bottom = 36.0;
    let width = REPORT_CHART_WIDTH as f64;
    let height = REPORT_TIME_CHART_HEIGHT as f64;
    let max_count = series
        .iter()
        .map(|point| point.tickets)
        .max()
        .unwrap_or(0)
        .max(1) as f64;
    let plot_width = width - left - right;
    let plot_height = height - top - bottom;
    let denominator = series.len().saturating_sub(1).max(1) as f64;
    let points = series
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let x = if series.len() == 1 {
                left + plot_width / 2.0
            } else {
                left + plot_width * index as f64 / denominator
            };
            let y = top + plot_height * (1.0 - point.tickets as f64 / max_count);
            format!("{x:.1},{y:.1}")
        })
        .collect::<Vec<_>>()
        .join(" ");
    let first_date = xml_escape(&series[0].date);
    let last_date = xml_escape(&series[series.len() - 1].date);
    let max_label = series.iter().map(|point| point.tickets).max().unwrap_or(0);
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Динамика обращений" viewBox="0 0 {REPORT_CHART_WIDTH} {REPORT_TIME_CHART_HEIGHT}">
<line x1="{left}" y1="{top}" x2="{left}" y2="{}" stroke="#b7c5cf"/><line x1="{left}" y1="{}" x2="{}" y2="{}" stroke="#b7c5cf"/>
<line x1="{left}" y1="{}" x2="{}" y2="{}" stroke="#e5eaee" stroke-dasharray="4 4"/><text x="8" y="{}" font-size="11" fill="#52616d">{max_label}</text>
<polyline points="{points}" fill="none" stroke="#168b63" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
<text x="{left}" y="{}" font-size="10" fill="#52616d">{first_date}</text><text x="{}" y="{}" text-anchor="end" font-size="10" fill="#52616d">{last_date}</text></svg>"##,
        top + plot_height,
        top + plot_height / 2.0,
        width - right,
        top + plot_height / 2.0,
        top + plot_height,
        width - right,
        top + plot_height,
        top + 4.0,
        height - 8.0,
        width - right,
        height - 8.0,
    )
}

fn report_bar_chart_svg(title: &str, values: &[(String, u64)], color: &str) -> String {
    let mut values = values.to_vec();
    values.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    values.truncate(REPORT_MAX_BAR_ROWS);
    if values.is_empty() {
        return "<p class=\"empty\">Нет данных для диаграммы.</p>".to_owned();
    }

    let height = 24 + REPORT_BAR_ROW_HEIGHT * values.len();
    let label_width = 210.0;
    let bar_width = 460.0;
    let max_count = values
        .iter()
        .map(|(_, count)| *count)
        .max()
        .unwrap_or(0)
        .max(1) as f64;
    let rows = values
        .iter()
        .enumerate()
        .map(|(index, (label, count))| {
            let y = 6 + index * REPORT_BAR_ROW_HEIGHT;
            let width = bar_width * *count as f64 / max_count;
            format!(
                "<text x=\"0\" y=\"{}\" font-size=\"11\" fill=\"#344450\">{}</text><rect x=\"{label_width}\" y=\"{}\" width=\"{width:.1}\" height=\"17\" rx=\"3\" fill=\"{color}\"/><text x=\"{}\" y=\"{}\" font-size=\"10\" fill=\"#344450\">{count}</text>",
                y + 13,
                xml_escape(label),
                y,
                label_width + width + 7.0,
                y + 12,
            )
        })
        .collect::<Vec<_>>()
        .join("");
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" role=\"img\" aria-label=\"{}\" viewBox=\"0 0 {REPORT_CHART_WIDTH} {height}\">{rows}</svg>",
        xml_escape(title)
    )
}

fn report_optional_number(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.2}"))
        .unwrap_or_else(|| "—".to_owned())
}

fn report_html(report: Option<&ReportSlice>) -> String {
    let range = xml_escape(&report_range(report));
    let generated_at = report
        .map(|value| value.analytics.generated_at.as_str())
        .map(xml_escape)
        .unwrap_or_else(|| "—".to_owned());
    let filter_summary = report
        .map(report_filter_summary)
        .map(|summary| xml_escape(&summary))
        .unwrap_or_else(|| "нет".to_owned());
    let metrics = [
        ("Всего обращений", "total_tickets"),
        ("Открыто", "open_tickets"),
        ("Решено", "resolved_tickets"),
        ("Высокий приоритет", "high_priority_tickets"),
        ("Решения операторов", "operator_decisions"),
        ("Подтверждено", "confirmed_decisions"),
        ("Исправлено", "corrected_decisions"),
        ("Средняя уверенность", "average_confidence"),
        ("Время до решения, мин", "avg_decision_minutes"),
    ]
    .iter()
    .map(|(label, key)| {
        format!(
            "<div class=\"metric\"><span>{}</span><strong>{}</strong></div>",
            xml_escape(label),
            xml_escape(&report_metric(report, key)),
        )
    })
    .collect::<Vec<_>>()
    .join("");

    let mut region_rows = String::new();
    let mut topic_rows = String::new();
    let mut series_rows = String::new();
    let mut alert_rows = String::new();
    let mut forecast_rows = String::new();
    let mut region_chart_values = Vec::new();
    let mut topic_chart_values = Vec::new();
    let mut forecast_status = "NO_DATA".to_owned();
    let mut forecast_model = "—".to_owned();
    let mut forecast_version = "—".to_owned();
    let mut forecast_horizon = "—".to_owned();
    let mut forecast_backtest = "—".to_owned();
    let mut forecast_peaks = "—".to_owned();
    if let Some(value) = report {
        let total_tickets = value
            .analytics
            .overview
            .get("total_tickets")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        for bucket in &value.analytics.by_region {
            region_chart_values.push((bucket.label.clone(), bucket.tickets as u64));
            region_rows.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.2}</td></tr>",
                xml_escape(&bucket.id),
                xml_escape(&bucket.label),
                bucket.tickets,
                report_previous_count(bucket.tickets, bucket.change_abs),
                report_change_label(bucket.change_abs, bucket.change_pct.map(f64::from)),
                bucket.avg_confidence,
            ));
        }
        for bucket in &value.analytics.by_topic {
            topic_chart_values.push((bucket.label.clone(), bucket.tickets as u64));
            let share = if total_tickets == 0 {
                0.0
            } else {
                bucket.tickets as f64 / total_tickets as f64 * 100.0
            };
            topic_rows.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{share:.1}%</td><td>{}</td><td>{:.2}</td></tr>",
                xml_escape(&bucket.id),
                xml_escape(&bucket.label),
                bucket.tickets,
                report_previous_count(bucket.tickets, bucket.change_abs),
                report_change_label(bucket.change_abs, bucket.change_pct.map(f64::from)),
                bucket.avg_confidence,
            ));
        }
        for point in &value.analytics.time_series {
            series_rows.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td></tr>",
                xml_escape(&point.date),
                point.tickets,
                point.resolved,
            ));
        }
        for alert in &value.alerts {
            alert_rows.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{} — {}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                xml_escape(&alert.alert_type),
                xml_escape(&alert.severity),
                xml_escape(&alert.status),
                xml_escape(&alert.region_id),
                xml_escape(&alert.topic_id),
                xml_escape(alert.period_start.as_deref().unwrap_or("—")),
                xml_escape(alert.period_end.as_deref().unwrap_or("—")),
                alert.current_count,
                report_optional_number(alert.baseline),
                alert.ticket_count,
            ));
        }
        forecast_status = value.forecast.status.clone();
        forecast_model = value.forecast.model.clone();
        forecast_version = value.forecast.model_version.clone();
        forecast_horizon = value.forecast.horizon_days.to_string();
        forecast_backtest = value.forecast.backtest.to_string();
        forecast_peaks = if value.forecast.expected_peaks.is_empty() {
            "—".to_owned()
        } else {
            value.forecast.expected_peaks.join(", ")
        };
        for point in &value.forecast.history {
            forecast_rows.push_str(&format!(
                "<tr><td>История</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                xml_escape(&point.date),
                point.tickets,
                point.resolved,
            ));
        }
        for point in &value.forecast.points {
            forecast_rows.push_str(&format!(
                "<tr><td>Прогноз</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                xml_escape(&point.date),
                point.tickets,
                point.resolved,
            ));
        }
    }
    let region_chart =
        report_bar_chart_svg("Обращения по регионам", &region_chart_values, "#2788aa");
    let topic_chart = report_bar_chart_svg("Обращения по темам", &topic_chart_values, "#168b63");
    let time_series = report
        .map(|value| report_time_series_svg(&value.analytics.time_series))
        .unwrap_or_else(|| "<p class=\"empty\">Нет данных для временного ряда.</p>".to_owned());
    let alerts_empty = if alert_rows.is_empty() {
        "<p class=\"empty\">В выбранном срезе активных сигналов нет.</p>"
    } else {
        ""
    };
    let forecast_empty = if forecast_rows.is_empty() {
        "<p class=\"empty\">Точки прогноза недоступны для этого среза.</p>"
    } else {
        ""
    };
    format!(
        r#"<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>Отчёт Pulse 109</title>
<style>
@page {{ size: A4; margin: 16mm 14mm 18mm; @bottom-right {{ content: counter(page) " / " counter(pages); color: #6b7780; font-size: 9pt; }} }}
body {{ font: 9pt "DejaVu Sans", sans-serif; color: #17202a; line-height: 1.35; }}
h1 {{ margin: 0 0 6px; color: #123c31; font-size: 22pt; }} h2 {{ margin: 19px 0 7px; padding-bottom: 4px; border-bottom: 1px solid #cbd5d9; color: #174b3c; font-size: 14pt; }}
h3 {{ margin: 11px 0 5px; color: #344450; font-size: 10pt; }} p {{ margin: 4px 0; }} .meta {{ color: #52616d; }} .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; margin: 10px 0; }}
.metric {{ padding: 8px; border: 1px solid #d7e0e3; border-radius: 4px; background: #f5f8f8; }} .metric span {{ display: block; color: #52616d; font-size: 8pt; }} .metric strong {{ display: block; margin-top: 3px; color: #123c31; font-size: 13pt; }}
table {{ width: 100%; margin: 7px 0 12px; border-collapse: collapse; }} th,td {{ padding: 5px 6px; border: 1px solid #d7e0e3; text-align: left; vertical-align: top; }} th {{ background: #edf3f2; color: #344450; font-size: 8pt; }} td {{ overflow-wrap: anywhere; }} tr {{ page-break-inside: avoid; }} .chart {{ width: 100%; margin: 4px 0 12px; }} .empty {{ padding: 8px; color: #687780; background: #f5f8f8; }} .small {{ color: #52616d; font-size: 8pt; }}
</style></head><body>
<h1>Отчёт Pulse 109</h1><p class="meta">Период: {range} · Сформировано: {generated_at}</p><p class="meta">Фильтры: {filter_summary}</p>
<h2>Ключевые показатели</h2><div class="metric-grid">{metrics}</div>
<h2>Графики</h2><h3>Динамика обращений</h3><div class="chart">{time_series}</div><h3>Регионы</h3><div class="chart">{region_chart}</div><h3>Темы</h3><div class="chart">{topic_chart}</div>
<h2>Регионы</h2><table><thead><tr><th>ID</th><th>Регион</th><th>Текущий период</th><th>Предыдущий период</th><th>Изменение</th><th>Средняя уверенность</th></tr></thead><tbody>{region_rows}</tbody></table>
<h2>Темы</h2><table><thead><tr><th>ID</th><th>Тема</th><th>Текущий период</th><th>Предыдущий период</th><th>Доля</th><th>Изменение</th><th>Средняя уверенность</th></tr></thead><tbody>{topic_rows}</tbody></table>
<h2>Временной ряд</h2><table><thead><tr><th>Дата</th><th>Обращения</th><th>Решено</th></tr></thead><tbody>{series_rows}</tbody></table>
<h2>Активные сигналы выбранного среза</h2>{alerts_empty}<table><thead><tr><th>Тип</th><th>Уровень</th><th>Статус</th><th>Регион</th><th>Тема</th><th>Период</th><th>Текущее</th><th>База</th><th>Обращения</th></tr></thead><tbody>{alert_rows}</tbody></table>
<h2>Прогноз</h2><p>Статус: {forecast_status} · Модель: {forecast_model} · Версия: {forecast_version} · Горизонт: {forecast_horizon} дней</p><p class="small">Ожидаемые пики: {forecast_peaks} · Backtest: {forecast_backtest}</p>{forecast_empty}<table><thead><tr><th>Ряд</th><th>Дата</th><th>Обращения</th><th>Решено</th></tr></thead><tbody>{forecast_rows}</tbody></table>
</body></html>"#,
        forecast_status = xml_escape(&forecast_status),
        forecast_model = xml_escape(&forecast_model),
        forecast_version = xml_escape(&forecast_version),
        forecast_horizon = xml_escape(&forecast_horizon),
        forecast_peaks = xml_escape(&forecast_peaks),
        forecast_backtest = xml_escape(&forecast_backtest),
    )
}

async fn pdf_report(report: Option<&ReportSlice>) -> Result<Vec<u8>, String> {
    let html = report_html(report);
    let renderer = env::var_os("PULSE_WEASYPRINT_BIN").unwrap_or_else(|| "weasyprint".into());
    tokio::task::spawn_blocking(move || {
        let mut child = Command::new(renderer)
            .args(["--quiet", "--encoding", "utf-8", "-", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|_| "PDF renderer is unavailable".to_owned())?;
        child
            .stdin
            .take()
            .ok_or_else(|| "PDF renderer input is unavailable".to_owned())?
            .write_all(html.as_bytes())
            .map_err(|_| "PDF renderer could not read the report".to_owned())?;
        let output = child
            .wait_with_output()
            .map_err(|_| "PDF renderer could not finish the report".to_owned())?;
        if !output.status.success()
            || !output.stdout.starts_with(b"%PDF-")
            || !output.stdout.windows(5).any(|window| window == b"%%EOF")
        {
            return Err("PDF renderer did not produce a valid document".to_owned());
        }
        Ok(output.stdout)
    })
    .await
    .map_err(|_| "PDF renderer task failed".to_owned())?
}

const REPORT_XLSX_HEADERS: [&str; 25] = [
    "period",
    "region_id",
    "region",
    "topic_id",
    "topic",
    "record_type",
    "metric",
    "value",
    "previous_value",
    "change_abs",
    "change_pct",
    "share_pct",
    "high_priority",
    "avg_confidence",
    "resolved",
    "status",
    "details",
    "generated_at",
    "alert_type",
    "severity",
    "ticket_count",
    "deviation",
    "ratio",
    "robust_z",
    "detector_version",
];
const REPORT_XLSX_COLUMNS: [&str; REPORT_XLSX_HEADERS.len()] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y",
];

#[derive(Default)]
struct ReportExportRow {
    period: String,
    region_id: String,
    region: String,
    topic_id: String,
    topic: String,
    record_type: String,
    metric: String,
    value: Option<String>,
    previous_value: Option<String>,
    change_abs: Option<String>,
    change_pct: Option<String>,
    share_pct: Option<String>,
    high_priority: Option<String>,
    avg_confidence: Option<String>,
    resolved: Option<String>,
    status: String,
    details: String,
    generated_at: String,
    alert_type: String,
    severity: String,
    ticket_count: Option<String>,
    deviation: Option<String>,
    ratio: Option<String>,
    robust_z: Option<String>,
    detector_version: String,
}

enum ReportXlsxCell {
    Text(String),
    Number(String),
    Empty,
}

impl ReportExportRow {
    fn into_cells(self) -> [ReportXlsxCell; REPORT_XLSX_HEADERS.len()] {
        [
            ReportXlsxCell::Text(self.period),
            ReportXlsxCell::Text(self.region_id),
            ReportXlsxCell::Text(self.region),
            ReportXlsxCell::Text(self.topic_id),
            ReportXlsxCell::Text(self.topic),
            ReportXlsxCell::Text(self.record_type),
            ReportXlsxCell::Text(self.metric),
            report_xlsx_number_cell(self.value),
            report_xlsx_number_cell(self.previous_value),
            report_xlsx_number_cell(self.change_abs),
            report_xlsx_number_cell(self.change_pct),
            report_xlsx_number_cell(self.share_pct),
            report_xlsx_number_cell(self.high_priority),
            report_xlsx_number_cell(self.avg_confidence),
            report_xlsx_number_cell(self.resolved),
            ReportXlsxCell::Text(self.status),
            ReportXlsxCell::Text(self.details),
            ReportXlsxCell::Text(self.generated_at),
            ReportXlsxCell::Text(self.alert_type),
            ReportXlsxCell::Text(self.severity),
            report_xlsx_number_cell(self.ticket_count),
            report_xlsx_number_cell(self.deviation),
            report_xlsx_number_cell(self.ratio),
            report_xlsx_number_cell(self.robust_z),
            ReportXlsxCell::Text(self.detector_version),
        ]
    }
}

fn report_xlsx_number_cell(value: Option<String>) -> ReportXlsxCell {
    value.map_or(ReportXlsxCell::Empty, ReportXlsxCell::Number)
}

fn report_xlsx_float(value: Option<f64>) -> Option<String> {
    value
        .filter(|value| value.is_finite())
        .map(|value| value.to_string())
}

fn report_xlsx_f32(value: Option<f32>) -> Option<String> {
    value
        .filter(|value| value.is_finite())
        .map(|value| value.to_string())
}

fn report_export_row(
    period: &str,
    record_type: &str,
    metric: &str,
    value: Option<String>,
    details: &str,
) -> ReportExportRow {
    ReportExportRow {
        period: period.to_owned(),
        record_type: record_type.to_owned(),
        metric: metric.to_owned(),
        value,
        details: details.to_owned(),
        ..ReportExportRow::default()
    }
}

fn xlsx_report(report: Option<&ReportSlice>) -> Vec<u8> {
    let content_types = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>"#;
    let root_rels = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#;
    let workbook = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Report" sheetId="1" r:id="rId1"/></sheets></workbook>"#;
    let workbook_rels = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>"#;
    let range = report_range(report);
    let mut rows = Vec::new();
    let mut filters_row = report_export_row(
        &range,
        "metadata",
        "filters",
        None,
        &report
            .map(report_filter_summary)
            .unwrap_or_else(|| "none".to_owned()),
    );
    if let Some(report) = report {
        filters_row.region_id = report.filters.region_id.clone().unwrap_or_default();
        filters_row.topic_id = report.filters.topic_id.clone().unwrap_or_default();
    }
    filters_row.status = "READY".to_owned();
    rows.push(filters_row);
    if let Some(value) = report {
        let mut generated_row = report_export_row(
            &range,
            "metadata",
            "generated_at",
            None,
            &format!("source={}", value.analytics.source),
        );
        generated_row.generated_at = value.analytics.generated_at.clone();
        rows.push(generated_row);

        if let Some(overview) = value.analytics.overview.as_object() {
            for (metric, metric_value) in overview {
                let value = metric_value.as_number().map(ToString::to_string);
                let details = if metric_value.is_null() {
                    "unavailable"
                } else {
                    ""
                };
                rows.push(report_export_row(&range, "metric", metric, value, details));
            }
        }

        let total_tickets = value
            .analytics
            .overview
            .get("total_tickets")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        for bucket in &value.analytics.by_region {
            let mut row = report_export_row(
                &range,
                "region",
                "tickets",
                Some(bucket.tickets.to_string()),
                "",
            );
            row.region_id = bucket.id.clone();
            row.region = bucket.label.clone();
            row.previous_value =
                Some(report_previous_count(bucket.tickets, bucket.change_abs).to_string());
            row.change_abs = bucket.change_abs.map(|change| change.to_string());
            row.change_pct = report_xlsx_f32(bucket.change_pct);
            row.share_pct = (total_tickets > 0)
                .then(|| (bucket.tickets as f64 / total_tickets as f64 * 100.0).to_string());
            row.high_priority = Some(bucket.high_priority.to_string());
            row.avg_confidence = report_xlsx_f32(Some(bucket.avg_confidence));
            rows.push(row);
        }
        for bucket in &value.analytics.by_topic {
            let share = if total_tickets == 0 {
                0.0
            } else {
                bucket.tickets as f64 / total_tickets as f64 * 100.0
            };
            let mut row = report_export_row(
                &range,
                "topic",
                "tickets",
                Some(bucket.tickets.to_string()),
                "",
            );
            row.topic_id = bucket.id.clone();
            row.topic = bucket.label.clone();
            row.previous_value =
                Some(report_previous_count(bucket.tickets, bucket.change_abs).to_string());
            row.change_abs = bucket.change_abs.map(|change| change.to_string());
            row.change_pct = report_xlsx_f32(bucket.change_pct);
            row.share_pct = (total_tickets > 0).then(|| share.to_string());
            row.high_priority = Some(bucket.high_priority.to_string());
            row.avg_confidence = report_xlsx_f32(Some(bucket.avg_confidence));
            rows.push(row);
        }
        for point in &value.analytics.time_series {
            let mut row = report_export_row(
                &point.date,
                "series",
                "tickets",
                Some(point.tickets.to_string()),
                "",
            );
            row.resolved = Some(point.resolved.to_string());
            rows.push(row);
        }
        for alert in &value.alerts {
            let period = format!(
                "{} — {}",
                alert.period_start.as_deref().unwrap_or("—"),
                alert.period_end.as_deref().unwrap_or("—"),
            );
            let mut row = report_export_row(
                &period,
                "alert",
                &alert.alert_type,
                Some(alert.current_count.to_string()),
                "",
            );
            row.region_id = alert.region_id.clone();
            row.topic_id = alert.topic_id.clone();
            row.alert_type = alert.alert_type.clone();
            row.severity = alert.severity.clone();
            row.ticket_count = Some(alert.ticket_count.to_string());
            row.previous_value = report_xlsx_float(alert.baseline);
            row.deviation = report_xlsx_float(alert.deviation);
            row.ratio = report_xlsx_float(alert.ratio);
            row.robust_z = report_xlsx_float(alert.robust_z);
            row.detector_version = alert.detector_version.clone().unwrap_or_default();
            row.status = alert.status.clone();
            rows.push(row);
        }
        let forecast_metadata = json!({
            "source": value.forecast.source,
            "model": value.forecast.model,
            "model_version": value.forecast.model_version,
            "horizon_days": value.forecast.horizon_days,
            "forecast_start": value.forecast.forecast_start,
            "expected_peaks": value.forecast.expected_peaks,
            "backtest": value.forecast.backtest,
        });
        let mut forecast_row = report_export_row(
            &range,
            "forecast_metadata",
            "forecast",
            Some(value.forecast.points.len().to_string()),
            &forecast_metadata.to_string(),
        );
        forecast_row.status = value.forecast.status.clone();
        rows.push(forecast_row);
        for (record_type, points) in [
            ("forecast_history", &value.forecast.history),
            ("forecast", &value.forecast.points),
        ] {
            for point in points {
                let mut row = report_export_row(
                    &point.date,
                    record_type,
                    "tickets",
                    Some(point.tickets.to_string()),
                    &format!(
                        "model={}; version={}",
                        value.forecast.model, value.forecast.model_version
                    ),
                );
                row.resolved = Some(point.resolved.to_string());
                row.status = value.forecast.status.clone();
                rows.push(row);
            }
        }
    } else {
        let mut forecast_row = report_export_row(&range, "forecast_metadata", "forecast", None, "");
        forecast_row.status = "NO_DATA".to_owned();
        rows.push(forecast_row);
    }
    let mut sheet_data = String::new();
    let header_cells = REPORT_XLSX_HEADERS
        .iter()
        .map(|value| ReportXlsxCell::Text((*value).to_owned()))
        .collect::<Vec<_>>();
    let mut all_rows = Vec::with_capacity(rows.len() + 1);
    all_rows.push(header_cells);
    all_rows.extend(rows.into_iter().map(|row| row.into_cells().into()));
    for (index, row) in all_rows.iter().enumerate() {
        let row_number = index + 1;
        let cells = row
            .iter()
            .zip(REPORT_XLSX_COLUMNS)
            .filter_map(|(cell, column)| match cell {
                ReportXlsxCell::Text(value) => Some(format!(
                    r#"<c r="{column}{row_number}" t="inlineStr"><is><t>{}</t></is></c>"#,
                    xml_escape(value)
                )),
                ReportXlsxCell::Number(value) => {
                    Some(format!(r#"<c r="{column}{row_number}"><v>{value}</v></c>"#))
                }
                ReportXlsxCell::Empty => None,
            })
            .collect::<Vec<_>>()
            .join("");
        sheet_data.push_str(&format!(r#"<row r="{row_number}">{cells}</row>"#));
    }
    let sheet = format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:Y{}"/><sheetData>{sheet_data}</sheetData></worksheet>"#,
        all_rows.len()
    );
    let entries = [
        ("[Content_Types].xml", content_types.as_bytes()),
        ("_rels/.rels", root_rels.as_bytes()),
        ("xl/workbook.xml", workbook.as_bytes()),
        ("xl/_rels/workbook.xml.rels", workbook_rels.as_bytes()),
        ("xl/worksheets/sheet1.xml", sheet.as_bytes()),
    ];
    zip_store(&entries)
}

fn xml_escape(value: &str) -> String {
    value
        .chars()
        .filter(|character| {
            matches!(*character, '\t' | '\n' | '\r')
                || ('\u{20}'..='\u{d7ff}').contains(character)
                || ('\u{e000}'..='\u{fffd}').contains(character)
                || ('\u{10000}'..='\u{10ffff}').contains(character)
        })
        .collect::<String>()
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

async fn report_response(format: &str, report: Option<&ReportSlice>) -> Result<Response, String> {
    let (body, content_type, filename) = if format == "pdf" {
        (
            pdf_report(report).await?,
            "application/pdf",
            "pulse109-report.pdf",
        )
    } else {
        (
            xlsx_report(report),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "pulse109-report.xlsx",
        )
    };
    let mut response = Response::new(Body::from(body));
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
    let disposition = format!("attachment; filename=\"{filename}\"");
    if let Ok(value) = HeaderValue::from_str(&disposition) {
        response
            .headers_mut()
            .insert(header::CONTENT_DISPOSITION, value);
    }
    Ok(response)
}

async fn export_pdf(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<AnalyticsQuery>,
) -> Result<Response, ApiError> {
    let actor = require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    if let Some(repository) = state.repository() {
        let report = build_report_slice(&repository, &query)
            .await
            .map_err(ApiError::Internal)?;
        repository
            .audit(
                &actor.user_id,
                "EXPORT_REPORT",
                "report",
                None,
                Some(&request_id_from_headers(&headers)),
                Some("pdf"),
                serde_json::to_value(&query).unwrap_or_else(|_| json!({})),
            )
            .await
            .map_err(ApiError::Internal)?;
        return report_response("pdf", Some(&report))
            .await
            .map_err(ApiError::Internal);
    }
    let report = {
        let store = state.read_store()?;
        memory_report_slice(&store, &query)?
    };
    report_response("pdf", Some(&report))
        .await
        .map_err(ApiError::Internal)
}

async fn export_xlsx(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<AnalyticsQuery>,
) -> Result<Response, ApiError> {
    let actor = require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    if let Some(repository) = state.repository() {
        let report = build_report_slice(&repository, &query)
            .await
            .map_err(ApiError::Internal)?;
        repository
            .audit(
                &actor.user_id,
                "EXPORT_REPORT",
                "report",
                None,
                Some(&request_id_from_headers(&headers)),
                Some("xlsx"),
                serde_json::to_value(&query).unwrap_or_else(|_| json!({})),
            )
            .await
            .map_err(ApiError::Internal)?;
        return report_response("xlsx", Some(&report))
            .await
            .map_err(ApiError::Internal);
    }
    let report = {
        let store = state.read_store()?;
        memory_report_slice(&store, &query)?
    };
    report_response("xlsx", Some(&report))
        .await
        .map_err(ApiError::Internal)
}

async fn reports(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<AnalyticsQuery>,
) -> Result<Json<Value>, ApiError> {
    require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    if let Some(repository) = state.repository() {
        let report = build_report_slice(&repository, &query)
            .await
            .map_err(ApiError::Internal)?;
        return Ok(Json(json!({
            "items": [
                {"id": "report-current-slice", "format": "pdf", "status": "ready", "download": "/api/v1/analytics/export.pdf"},
                {"id": "report-current-slice", "format": "xlsx", "status": "ready", "download": "/api/v1/analytics/export.xlsx"}
            ],
            "slice": report,
            "source": "postgres"
        })));
    }
    let store = state.read_store()?;
    let report = memory_report_slice(&store, &query)?;
    Ok(Json(json!({
        "items": [
            {"id": "report-current-slice", "format": "pdf", "status": "ready", "download": "/api/v1/analytics/export.pdf"},
            {"id": "report-current-slice", "format": "xlsx", "status": "ready", "download": "/api/v1/analytics/export.xlsx"}
        ],
        "slice": report,
        "source": "deterministic-demo"
    })))
}

async fn events(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, ApiError> {
    require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    let receiver = state.alert_events.subscribe();
    let (source, alerts) = if let Some(repository) = state.repository() {
        let alerts = repository
            .list_alerts(&AlertQuery {
                status: None,
                severity: None,
                region_id: None,
            })
            .await
            .map_err(ApiError::Internal)?;
        ("postgres", alerts)
    } else {
        let store = state.read_store()?;
        (
            "deterministic-demo",
            store.alerts.values().cloned().collect(),
        )
    };
    let snapshot = Event::default()
        .event("alerts.snapshot")
        .data(json!({"type": "alerts.snapshot", "source": source, "alerts": alerts}).to_string());
    let updates = BroadcastStream::new(receiver).map(|result| {
        let event = match result {
            Ok(payload) => Event::default().event("alerts.changed").data(payload),
            Err(_) => Event::default()
                .event("alerts.resync")
                .data(r#"{"type":"alerts.resync"}"#),
        };
        Ok::<Event, Infallible>(event)
    });
    let stream = tokio_stream::once(Ok::<Event, Infallible>(snapshot)).chain(updates);
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(StdDuration::from_secs(15)))
        .into_response())
}

fn demo_analytics_as_of(store: &Store) -> Result<DateTime<Utc>, ApiError> {
    let fixture_time = DateTime::parse_from_rfc3339(DEMO_TIMESTAMP)
        .map(|value| value.with_timezone(&Utc))
        .map_err(|error| {
            ApiError::Internal(format!("invalid configured demo timestamp: {error}"))
        })?;
    let latest_ticket_time = store
        .tickets
        .values()
        .filter_map(ticket_created_at)
        .filter(|created_at| *created_at <= Utc::now())
        .max()
        .unwrap_or(fixture_time);
    Ok(fixture_time.max(latest_ticket_time))
}

fn analytics_range_days(range: Option<&str>) -> i64 {
    range
        .map(str::trim)
        .map(|value| value.trim_end_matches('d'))
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(30)
        .clamp(1, 366)
}

fn ticket_created_at(ticket: &Ticket) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(&ticket.created_at)
        .ok()
        .map(|value| value.with_timezone(&Utc))
}

fn ticket_matches_analytics_filters(ticket: &Ticket, query: &AnalyticsQuery) -> bool {
    query
        .region_id
        .as_deref()
        .is_none_or(|value| ticket.region_id == value.trim())
        && query
            .topic_id
            .as_deref()
            .is_none_or(|value| ticket.topic_id == value.trim())
        && query.service_id.as_deref().is_none_or(|value| {
            service_for_topic(&ticket.topic_id).eq_ignore_ascii_case(value.trim())
        })
        && query
            .status
            .as_deref()
            .is_none_or(|value| ticket.status.eq_ignore_ascii_case(value.trim()))
        // The memory fixture has no district or channel columns. Keep those
        // filters from returning misleading, unfiltered aggregates.
        && query.district.is_none()
        && query.channel.is_none()
}

fn average_ticket_confidence(tickets: &[&Ticket], store: &Store) -> f32 {
    if tickets.is_empty() {
        return 0.0;
    }
    tickets
        .iter()
        .filter_map(|ticket| store.predictions.get(&ticket.id))
        .map(|prediction| prediction.confidence)
        .sum::<f32>()
        / tickets.len() as f32
}

fn is_open_ticket_status(status: &str) -> bool {
    matches!(
        status.to_ascii_uppercase().as_str(),
        "OPEN" | "IN_PROGRESS" | "TRIAGED"
    )
}

fn is_resolved_ticket_status(status: &str) -> bool {
    matches!(status.to_ascii_uppercase().as_str(), "RESOLVED" | "CLOSED")
}

fn is_high_priority(priority: &str) -> bool {
    matches!(priority.to_ascii_lowercase().as_str(), "high" | "critical")
}

fn analytics_percent_change(current: usize, previous: usize) -> Option<f32> {
    (previous > 0).then(|| ((current as f32 - previous as f32) / previous as f32) * 100.0)
}

fn demo_time_series(
    tickets: &[&Ticket],
    period_start: DateTime<Utc>,
    as_of: DateTime<Utc>,
) -> Vec<TimeSeriesPoint> {
    let mut daily_counts = BTreeMap::<NaiveDate, (u32, u32)>::new();
    for ticket in tickets {
        if let Some(created_at) = ticket_created_at(ticket) {
            let counts = daily_counts.entry(created_at.date_naive()).or_default();
            counts.0 += 1;
            if is_resolved_ticket_status(&ticket.status) {
                counts.1 += 1;
            }
        }
    }

    let mut points = Vec::new();
    let mut date = period_start.date_naive();
    while date <= as_of.date_naive() {
        let (tickets, resolved) = daily_counts.get(&date).copied().unwrap_or_default();
        points.push(TimeSeriesPoint {
            date: date.to_string(),
            tickets,
            resolved,
        });
        date += Duration::days(1);
    }
    points
}

fn metric_bucket(
    id: String,
    label: String,
    tickets: Vec<&Ticket>,
    previous_count: usize,
    store: &Store,
) -> MetricBucket {
    let avg_confidence = if tickets.is_empty() {
        0.0
    } else {
        tickets
            .iter()
            .filter_map(|ticket| store.predictions.get(&ticket.id))
            .map(|prediction| prediction.confidence)
            .sum::<f32>()
            / tickets.len() as f32
    };
    MetricBucket {
        id,
        label,
        tickets: tickets.len(),
        high_priority: tickets
            .iter()
            .filter(|ticket| is_high_priority(&ticket.priority))
            .count(),
        avg_confidence,
        change_abs: Some(tickets.len() as i64 - previous_count as i64),
        change_pct: analytics_percent_change(tickets.len(), previous_count),
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct ForecastResponse {
    pub source: String,
    pub model_version: String,
    pub model: String,
    pub status: String,
    pub insufficient_history: bool,
    pub horizon_days: u32,
    pub history: Vec<TimeSeriesPoint>,
    pub forecast_start: Option<String>,
    pub points: Vec<TimeSeriesPoint>,
    pub expected_peaks: Vec<String>,
    pub backtest: Value,
}

#[derive(Debug, Deserialize, Default)]
pub struct ForecastQuery {
    pub horizon: Option<u32>,
    pub horizon_days: Option<u32>,
    pub region_id: Option<String>,
    pub topic_id: Option<String>,
    pub service_id: Option<String>,
    pub status: Option<String>,
    pub district: Option<String>,
    pub channel: Option<String>,
}

async fn forecast(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<ForecastQuery>,
) -> Result<Json<ForecastResponse>, ApiError> {
    require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    let horizon_days = query.horizon.or(query.horizon_days).unwrap_or(30);
    if !matches!(horizon_days, 30 | 60 | 90) {
        return Err(ApiError::BadRequest(
            "horizon must be 30, 60 or 90 days".to_owned(),
        ));
    }
    if let Some(repository) = state.repository() {
        return repository
            .forecast(&query)
            .await
            .map(Json)
            .map_err(ApiError::Internal);
    }
    let store = state.read_store()?;
    Ok(Json(memory_forecast_response(
        &store,
        &query,
        horizon_days,
    )?))
}

fn memory_forecast_response(
    store: &Store,
    query: &ForecastQuery,
    horizon_days: u32,
) -> Result<ForecastResponse, ApiError> {
    let analytics_query = AnalyticsQuery {
        range: Some(format!("{FORECAST_HISTORY_DAYS}d")),
        region_id: query.region_id.clone(),
        topic_id: query.topic_id.clone(),
        service_id: query.service_id.clone(),
        status: query.status.clone(),
        district: query.district.clone(),
        channel: query.channel.clone(),
    };
    let generated_at = demo_analytics_as_of(store)?;
    let current_since = generated_at - Duration::days(FORECAST_HISTORY_DAYS);
    let filtered = store
        .tickets
        .values()
        .filter(|ticket| {
            ticket_matches_analytics_filters(ticket, &analytics_query)
                && ticket_created_at(ticket).is_some_and(|created_at| {
                    created_at >= current_since && created_at <= generated_at
                })
        })
        .collect::<Vec<_>>();

    if filtered.is_empty() {
        return Ok(ForecastResponse {
            source: "deterministic-demo".to_owned(),
            model_version: "forecast-seasonal-naive-demo-v1".to_owned(),
            model: "seasonal-naive-demo".to_owned(),
            status: "INSUFFICIENT_HISTORY".to_owned(),
            insufficient_history: true,
            horizon_days,
            history: Vec::new(),
            forecast_start: None,
            points: Vec::new(),
            expected_peaks: Vec::new(),
            backtest: json!({"sample_count": 0}),
        });
    }

    let history = demo_time_series(&filtered, current_since, generated_at);
    let active_days = filtered
        .iter()
        .filter_map(|ticket| ticket_created_at(ticket).map(|created_at| created_at.date_naive()))
        .collect::<std::collections::HashSet<_>>()
        .len();
    if active_days < FORECAST_SEASON_LENGTH_DAYS {
        return Ok(ForecastResponse {
            source: "deterministic-demo".to_owned(),
            model_version: "forecast-seasonal-naive-demo-v1".to_owned(),
            model: "seasonal-naive-demo".to_owned(),
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
                "observed_days": active_days,
                "required_days": FORECAST_SEASON_LENGTH_DAYS,
            }),
        });
    }
    let seasonal_pattern = history
        .iter()
        .rev()
        .take(FORECAST_SEASON_LENGTH_DAYS)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    let start_date = generated_at.date_naive() + Duration::days(1);
    let points = (0..horizon_days)
        .map(|index| {
            let history_point = seasonal_pattern[index as usize % seasonal_pattern.len()];
            TimeSeriesPoint {
                date: (start_date + Duration::days(i64::from(index))).to_string(),
                tickets: history_point.tickets,
                resolved: 0,
            }
        })
        .collect::<Vec<_>>();
    let peak_value = points
        .iter()
        .map(|point| point.tickets)
        .max()
        .unwrap_or_default();
    let expected_peaks = points
        .iter()
        .filter(|point| peak_value > 0 && point.tickets == peak_value)
        .map(|point| point.date.clone())
        .collect();
    let forecast_start = points.first().map(|point| point.date.clone());

    Ok(ForecastResponse {
        source: "deterministic-demo".to_owned(),
        model_version: "forecast-seasonal-naive-demo-v1".to_owned(),
        model: "seasonal-naive-demo".to_owned(),
        status: "DEMO_ONLY".to_owned(),
        insufficient_history: false,
        horizon_days,
        history,
        forecast_start,
        points,
        expected_peaks,
        backtest: json!({"status": "DEMO_ONLY", "history_days": FORECAST_HISTORY_DAYS}),
    })
}

#[derive(Debug, Deserialize)]
pub struct AlertQuery {
    pub status: Option<String>,
    pub severity: Option<String>,
    pub region_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AlertListResponse {
    pub items: Vec<Alert>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct AlertDetectionResponse {
    pub status: String,
    pub source: String,
    pub detector_version: String,
    pub evaluated_series: usize,
    pub insufficient_history_series: usize,
    pub config: AlertDetectorConfig,
    pub items: Vec<Alert>,
    pub total: usize,
}

async fn list_alerts(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<AlertQuery>,
) -> Result<Json<AlertListResponse>, ApiError> {
    require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    if let Some(repository) = state.repository() {
        let items = repository
            .list_alerts(&query)
            .await
            .map_err(ApiError::Internal)?;
        let total = items.len();
        return Ok(Json(AlertListResponse { items, total }));
    }
    let store = state.read_store()?;
    let items = store
        .alerts
        .values()
        .filter(|alert| {
            query
                .status
                .as_deref()
                .is_none_or(|value| alert.status == value)
        })
        .filter(|alert| {
            query
                .severity
                .as_deref()
                .is_none_or(|value| alert.severity == value)
        })
        .filter(|alert| {
            query
                .region_id
                .as_deref()
                .is_none_or(|value| alert.region_id == value)
        })
        .cloned()
        .collect::<Vec<_>>();
    let total = items.len();
    Ok(Json(AlertListResponse { items, total }))
}

async fn get_alert(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(alert_id): Path<String>,
) -> Result<Json<Alert>, ApiError> {
    require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    if let Some(repository) = state.repository() {
        return repository
            .get_alert(&alert_id)
            .await
            .map(Json)
            .map_err(|error| {
                if error.contains("not found") {
                    ApiError::NotFound(error)
                } else {
                    ApiError::Internal(error)
                }
            });
    }
    let store = state.read_store()?;
    store
        .alerts
        .get(&alert_id)
        .cloned()
        .ok_or_else(|| ApiError::NotFound(format!("alert {alert_id} not found")))
        .map(Json)
}

async fn ack_alert(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(alert_id): Path<String>,
) -> Result<Json<Alert>, ApiError> {
    let actor = require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    if let Some(repository) = state.repository() {
        let response = repository
            .acknowledge_alert(&alert_id, &actor.user_id)
            .await
            .map_err(|error| {
                if error.contains("not found") {
                    ApiError::NotFound(error)
                } else {
                    ApiError::Internal(error)
                }
            })?;
        repository
            .audit(
                &actor.user_id,
                "ACK_ALERT",
                "alert",
                Some(&alert_id),
                Some(&request_id_from_headers(&headers)),
                None,
                json!({}),
            )
            .await
            .map_err(ApiError::Internal)?;
        state.publish_alerts_changed();
        return Ok(Json(response));
    }
    let mut store = state.write_store()?;
    let alert = store
        .alerts
        .get_mut(&alert_id)
        .ok_or_else(|| ApiError::NotFound(format!("alert {alert_id} not found")))?;
    alert.status = "acknowledged".to_owned();
    alert.acknowledged_by = Some(actor.user_id);
    alert.acknowledged_at = Some(DEMO_TIMESTAMP.to_owned());
    state.publish_alerts_changed();
    Ok(Json(alert.clone()))
}

async fn close_alert(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(alert_id): Path<String>,
) -> Result<Json<Alert>, ApiError> {
    let actor = require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    if let Some(repository) = state.repository() {
        let response = repository
            .close_alert(&alert_id, &actor.user_id)
            .await
            .map_err(|error| {
                if error.contains("not found") {
                    ApiError::NotFound(error)
                } else {
                    ApiError::Internal(error)
                }
            })?;
        repository
            .audit(
                &actor.user_id,
                "CLOSE_ALERT",
                "alert",
                Some(&alert_id),
                Some(&request_id_from_headers(&headers)),
                None,
                json!({}),
            )
            .await
            .map_err(ApiError::Internal)?;
        state.publish_alerts_changed();
        return Ok(Json(response));
    }
    let mut store = state.write_store()?;
    let alert = store
        .alerts
        .get_mut(&alert_id)
        .ok_or_else(|| ApiError::NotFound(format!("alert {alert_id} not found")))?;
    alert.status = "closed".to_owned();
    alert.closed_by = Some(actor.user_id);
    alert.closed_at = Some(Utc::now().to_rfc3339());
    state.publish_alerts_changed();
    Ok(Json(alert.clone()))
}

async fn detect_alerts(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<AlertDetectionResponse>, ApiError> {
    require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    let config = state.config.alert_detector.clone();
    if let Some(repository) = state.repository() {
        let run = repository
            .detect_alerts(&config)
            .await
            .map_err(ApiError::Internal)?;
        let total = run.items.len();
        if run.new_alerts > 0 {
            state.publish_alerts_changed();
        }
        return Ok(Json(AlertDetectionResponse {
            status: run.status.to_owned(),
            source: run.source.to_owned(),
            detector_version: config.detector_version.clone(),
            evaluated_series: run.evaluated_series,
            insufficient_history_series: run.insufficient_history_series,
            config,
            items: run.items,
            total,
        }));
    }
    let store = state.read_store()?;
    let insufficient_history_series = store
        .tickets
        .values()
        .map(|ticket| (ticket.region_id.clone(), ticket.topic_id.clone()))
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    Ok(Json(AlertDetectionResponse {
        status: "INSUFFICIENT_HISTORY".to_owned(),
        source: "memory_demo".to_owned(),
        detector_version: config.detector_version.clone(),
        evaluated_series: 0,
        insufficient_history_series,
        config,
        items: Vec::new(),
        total: 0,
    }))
}

#[derive(Debug, Serialize)]
pub struct LearningOverview {
    pub items: Vec<LearningCycle>,
    pub active_cycle: Option<LearningCycle>,
    pub production_model: Option<ModelVersion>,
    pub controlled_loop: Value,
}

async fn learning_overview(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<LearningOverview>, ApiError> {
    require_role(&headers, &state.config, &[Role::MlReviewer, Role::Admin])?;
    if let Some(repository) = state.repository() {
        return repository
            .learning_overview()
            .await
            .map(Json)
            .map_err(ApiError::Internal);
    }
    let store = state.read_store()?;
    let items = store.learning_cycles.values().cloned().collect::<Vec<_>>();
    let active_cycle = items
        .iter()
        .find(|cycle| is_active_learning_cycle_state(&cycle.state))
        .cloned();
    let production_model = store
        .models
        .values()
        .find(|model| model.status == "production")
        .cloned();
    Ok(Json(LearningOverview {
        items,
        active_cycle,
        production_model,
        controlled_loop: json!({
            "stages": ["COLLECT", "TRAINING", "EVALUATE", "DECISION", "PROMOTED", "REJECTED"],
            "production_auto_update": false,
            "operator_correction_triggers_training": false,
        }),
    }))
}

#[derive(Debug, Deserialize, Default)]
pub struct CreateLearningCycleRequest {
    pub candidate_model_version: Option<String>,
    pub evaluation_dataset_version: Option<String>,
}

async fn create_learning_cycle(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CreateLearningCycleRequest>,
) -> Result<(StatusCode, Json<LearningCycle>), ApiError> {
    let actor = require_role(&headers, &state.config, &[Role::MlReviewer, Role::Admin])?;
    if let Some(repository) = state.repository() {
        let cycle = repository
            .create_learning_cycle(
                &request,
                &state.config,
                &actor.user_id,
                &request_id_from_headers(&headers),
            )
            .await
            .map_err(|error| {
                if error.contains("active learning cycle")
                    || error.contains("frozen evaluation dataset")
                {
                    ApiError::Conflict(error)
                } else {
                    ApiError::Internal(error)
                }
            })?;
        return Ok((StatusCode::CREATED, Json(cycle)));
    }
    let mut store = state.write_store()?;
    if store
        .learning_cycles
        .values()
        .any(|cycle| is_active_learning_cycle_state(&cycle.state))
    {
        return Err(ApiError::Conflict(
            "an active classifier learning cycle already exists".to_owned(),
        ));
    }
    let started_at = Utc::now();
    let ends_at =
        started_at + Duration::hours(i64::from(state.config.learning_cycle_duration_hours));
    let production_model_version = store
        .models
        .values()
        .find(|model| model.status.eq_ignore_ascii_case("production"))
        .map(|model| model.id.clone());
    let cycle = LearningCycle {
        id: format!("cycle-{:03}", store.next_cycle_number),
        cycle_id: format!("cycle-{:03}", store.next_cycle_number),
        state: "COLLECT".to_owned(),
        dataset_version: "pending".to_owned(),
        candidate_model_version: request.candidate_model_version.unwrap_or_else(|| {
            format!(
                "classifier-candidate-2026-09-{:03}",
                store.next_cycle_number
            )
        }),
        collect_started_at: started_at.to_rfc3339(),
        collect_ends_at: ends_at.to_rfc3339(),
        evaluation_started_at: None,
        evaluation_ends_at: None,
        shadow_prediction_count: 0,
        shadow_inference_failures: 0,
        shadow_operator_decision_count: 0,
        blind_ab_enabled: false,
        production_model_version,
        frozen_evaluation_dataset_version: request
            .evaluation_dataset_version
            .or_else(|| state.config.learning_evaluation_dataset_version.clone()),
        candidate_dataset_checksum: None,
        min_feedback_count: state.config.learning_min_feedback_count as u32,
        promotion_policy_version: state.config.learning_promotion_policy_version.clone(),
        manual_close_enabled: state.config.learning_manual_close_enabled,
        metrics: LearningMetrics {
            macro_f1: 0.0,
            accuracy: 0.0,
            evaluated_samples: 0,
        },
        feedback_count: 0,
        decision_note: None,
        created_at: started_at.to_rfc3339(),
        updated_at: started_at.to_rfc3339(),
    };
    store.next_cycle_number += 1;
    store
        .learning_cycles
        .insert(cycle.id.clone(), cycle.clone());
    Ok((StatusCode::CREATED, Json(cycle)))
}

async fn get_learning_cycle(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(cycle_id): Path<String>,
) -> Result<Json<LearningCycle>, ApiError> {
    require_role(&headers, &state.config, &[Role::MlReviewer, Role::Admin])?;
    if let Some(repository) = state.repository() {
        return repository
            .get_learning_cycle(&cycle_id)
            .await
            .map(Json)
            .map_err(|error| {
                if error.contains("not found") {
                    ApiError::NotFound(error)
                } else {
                    ApiError::Internal(error)
                }
            });
    }
    let store = state.read_store()?;
    store
        .learning_cycles
        .get(&cycle_id)
        .cloned()
        .ok_or_else(|| ApiError::NotFound(format!("learning cycle {cycle_id} not found")))
        .map(Json)
}

#[derive(Debug, Deserialize)]
pub struct LearningFeedbackRequest {
    #[serde(alias = "ticketId")]
    pub ticket_id: String,
    pub feedback_type: Option<String>,
    pub decision: Option<String>,
    pub source: Option<String>,
    pub comment: Option<String>,
}

async fn add_learning_feedback(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(cycle_id): Path<String>,
    Json(request): Json<LearningFeedbackRequest>,
) -> Result<(StatusCode, Json<LearningFeedback>), ApiError> {
    let actor = require_role(
        &headers,
        &state.config,
        &[Role::Operator, Role::Manager, Role::MlReviewer, Role::Admin],
    )?;
    if let Some(repository) = state.repository() {
        let feedback = repository
            .add_learning_feedback(&cycle_id, &request, &actor.user_id)
            .await
            .map_err(|error| {
                if error.contains("not found") {
                    ApiError::NotFound(error)
                } else if error.contains("cannot accept feedback")
                    || error.contains("collection period has ended")
                {
                    ApiError::Conflict(error)
                } else {
                    ApiError::Internal(error)
                }
            })?;
        return Ok((StatusCode::CREATED, Json(feedback)));
    }
    let mut store = state.write_store()?;
    let cycle = store
        .learning_cycles
        .get(&cycle_id)
        .ok_or_else(|| ApiError::NotFound(format!("learning cycle {cycle_id} not found")))?;
    if cycle.state != "COLLECT" {
        return Err(ApiError::Conflict(format!(
            "learning cycle {cycle_id} cannot accept feedback in state {}",
            cycle.state
        )));
    }
    if learning_collect_end_is_due(&cycle.collect_ends_at, Utc::now()) {
        return Err(ApiError::Conflict(format!(
            "learning cycle {cycle_id} collection period has ended"
        )));
    }
    if !store.tickets.contains_key(&request.ticket_id) {
        return Err(ApiError::NotFound(format!(
            "ticket {} not found",
            request.ticket_id
        )));
    }
    let feedback_type = request
        .feedback_type
        .or(request.decision)
        .or(request.source)
        .unwrap_or_else(|| "accepted".to_owned());
    let feedback = LearningFeedback {
        id: format!("feedback-{:03}", store.next_feedback_number),
        ticket_id: request.ticket_id,
        cycle_id: Some(cycle_id.clone()),
        feedback_type,
        comment: request.comment,
        suggestion: None,
        user_id: actor.user_id,
        created_at: Utc::now().to_rfc3339(),
    };
    store.next_feedback_number += 1;
    store.learning_feedback.push(feedback.clone());
    if let Some(cycle) = store.learning_cycles.get_mut(&cycle_id) {
        cycle.feedback_count += 1;
        cycle.updated_at = feedback.created_at.clone();
    }
    Ok((StatusCode::CREATED, Json(feedback)))
}

#[derive(Debug, Deserialize)]
pub struct RelationFeedbackRequest {
    pub related_ticket_id: Option<String>,
    pub relation: String,
    pub decision: Option<String>,
    pub comment: Option<String>,
    pub suggestion: Option<RelationSuggestionSnapshot>,
}

fn validate_relation_suggestion(suggestion: &RelationSuggestionSnapshot) -> Result<(), ApiError> {
    let valid_score = |value: f32| value.is_finite() && (0.0..=1.0).contains(&value);
    if !valid_score(suggestion.score) || !valid_score(suggestion.threshold) {
        return Err(ApiError::BadRequest(
            "suggestion score and threshold must be between 0 and 1".to_owned(),
        ));
    }
    if suggestion.rule_version.trim().is_empty()
        || suggestion.rule_version.len() > 128
        || suggestion.model_version.trim().is_empty()
        || suggestion.model_version.len() > 128
        || suggestion.distance_metric.trim().is_empty()
        || suggestion.distance_metric.len() > 128
    {
        return Err(ApiError::BadRequest(
            "suggestion versions must contain 1 to 128 characters".to_owned(),
        ));
    }
    Ok(())
}

async fn relation_feedback(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(ticket_id): Path<String>,
    Json(request): Json<RelationFeedbackRequest>,
) -> Result<(StatusCode, Json<LearningFeedback>), ApiError> {
    let actor = require_role(
        &headers,
        &state.config,
        &[Role::Operator, Role::Manager, Role::Admin],
    )?;
    let relation = request.relation.trim().to_ascii_uppercase();
    if !matches!(
        relation.as_str(),
        "DUPLICATE" | "REPEAT" | "SIMILAR" | "UNRELATED"
    ) {
        return Err(ApiError::BadRequest(
            "relation must be DUPLICATE, REPEAT, SIMILAR or UNRELATED".to_owned(),
        ));
    }
    let decision = request
        .decision
        .as_deref()
        .unwrap_or("confirmed")
        .trim()
        .to_ascii_uppercase();
    if !matches!(decision.as_str(), "CONFIRMED" | "REJECTED") {
        return Err(ApiError::BadRequest(
            "decision must be CONFIRMED or REJECTED".to_owned(),
        ));
    }
    if let Some(suggestion) = request.suggestion.as_ref() {
        validate_relation_suggestion(suggestion)?;
    }
    if let Some(repository) = state.repository() {
        repository
            .relation_feedback(
                &ticket_id,
                request.related_ticket_id.as_deref(),
                &relation,
                &decision,
                &actor.user_id,
                request.suggestion.as_ref(),
            )
            .await
            .map_err(|error| {
                if error.contains("not found") {
                    ApiError::NotFound(error)
                } else {
                    ApiError::Internal(error)
                }
            })?;
        repository
            .audit(
                &actor.user_id,
                "RELATION_FEEDBACK",
                "ticket",
                Some(&ticket_id),
                Some(&request_id_from_headers(&headers)),
                Some(&decision),
                json!({
                    "relation": &relation,
                    "related_ticket_id": &request.related_ticket_id,
                    "suggestion": &request.suggestion,
                }),
            )
            .await
            .map_err(ApiError::Internal)?;
        return Ok((
            StatusCode::CREATED,
            Json(LearningFeedback {
                id: "db-relation-feedback".to_owned(),
                ticket_id,
                cycle_id: None,
                feedback_type: format!("relation:{relation}:{decision}"),
                comment: request.comment,
                suggestion: request.suggestion,
                user_id: actor.user_id,
                created_at: Utc::now().to_rfc3339(),
            }),
        ));
    }
    let mut store = state.write_store()?;
    if !store.tickets.contains_key(&ticket_id) {
        return Err(ApiError::NotFound(format!("ticket {ticket_id} not found")));
    }
    if let Some(related_ticket_id) = request.related_ticket_id.as_deref() {
        if !store.tickets.contains_key(related_ticket_id) {
            return Err(ApiError::NotFound(format!(
                "related ticket {related_ticket_id} not found"
            )));
        }
    }
    let feedback = LearningFeedback {
        id: format!("feedback-{:03}", store.next_feedback_number),
        ticket_id,
        cycle_id: None,
        feedback_type: format!("relation:{relation}:{decision}"),
        comment: request.comment.or_else(|| {
            request
                .related_ticket_id
                .map(|id| format!("related_ticket_id={id}"))
        }),
        suggestion: request.suggestion,
        user_id: actor.user_id,
        created_at: DEMO_TIMESTAMP.to_owned(),
    };
    store.next_feedback_number += 1;
    store.learning_feedback.push(feedback.clone());
    Ok((StatusCode::CREATED, Json(feedback)))
}

#[derive(Debug, Deserialize, Default)]
pub struct CycleDecisionRequest {
    pub note: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct CloseLearningCycleRequest {
    pub cycle_id: Option<String>,
}

async fn close_learning_cycle(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CloseLearningCycleRequest>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    let actor = require_role(&headers, &state.config, &[Role::MlReviewer, Role::Admin])?;
    if let Some(repository) = state.repository() {
        let result = repository
            .close_learning_cycle(&request, &actor.user_id, &request_id_from_headers(&headers))
            .await
            .map_err(|error| {
                if error.contains("not found") {
                    ApiError::NotFound(error)
                } else if error.contains("COLLECT cycle")
                    || error.contains("no eligible")
                    || error.contains("cannot close")
                    || error.contains("not eligible")
                {
                    ApiError::Conflict(error)
                } else {
                    ApiError::Internal(error)
                }
            })?;
        return Ok((StatusCode::ACCEPTED, Json(result)));
    }
    let mut store = state.write_store()?;
    let cycle_id = request.cycle_id.or_else(|| {
        store
            .learning_cycles
            .values()
            .find(|cycle| matches!(cycle.state.as_str(), "COLLECT" | "EVALUATE"))
            .map(|cycle| cycle.id.clone())
    });
    let cycle_id = cycle_id.ok_or_else(|| {
        ApiError::Conflict("no COLLECT or EVALUATE cycle is available".to_owned())
    })?;
    let cycle = store
        .learning_cycles
        .get_mut(&cycle_id)
        .ok_or_else(|| ApiError::NotFound(format!("learning cycle {cycle_id} not found")))?;
    if !matches!(cycle.state.as_str(), "COLLECT" | "EVALUATE") {
        return Err(ApiError::Conflict(format!(
            "cycle {} cannot close from state {}",
            cycle.id, cycle.state
        )));
    }
    let evaluation_stage = cycle.state == "EVALUATE";
    let window_end = if evaluation_stage {
        cycle.evaluation_ends_at.as_deref()
    } else {
        Some(cycle.collect_ends_at.as_str())
    };
    if !cycle.manual_close_enabled
        && !window_end.is_some_and(|value| learning_collect_end_is_due(value, Utc::now()))
    {
        return Err(ApiError::Conflict(format!(
            "learning cycle {} is not eligible to close before its window ends",
            cycle.id
        )));
    }
    if evaluation_stage {
        let closed_at = Utc::now().to_rfc3339();
        cycle.evaluation_ends_at = Some(closed_at.clone());
        cycle.state = "DECISION".to_owned();
        cycle.decision_note = Some("EVALUATION_WINDOW_CLOSED".to_owned());
        cycle.updated_at = closed_at;
        return Ok((
            StatusCode::ACCEPTED,
            Json(json!({
                "job_id": Value::Null,
                "state": "DECISION",
                "cycle": cycle,
                "production_model_unchanged": true,
            })),
        ));
    }
    if cycle.feedback_count < cycle.min_feedback_count {
        cycle.state = "INSUFFICIENT_FEEDBACK".to_owned();
        cycle.decision_note = Some("INSUFFICIENT_FEEDBACK".to_owned());
        cycle.updated_at = Utc::now().to_rfc3339();
        return Ok((
            StatusCode::ACCEPTED,
            Json(json!({
                "job_id": Value::Null,
                "state": "INSUFFICIENT_FEEDBACK",
                "cycle": cycle,
                "production_model_unchanged": true,
            })),
        ));
    }
    return Err(ApiError::Unavailable(
        "persistent background training requires PostgreSQL storage".to_owned(),
    ));
}

async fn candidate_evaluation(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    require_role(&headers, &state.config, &[Role::MlReviewer, Role::Admin])?;
    if let Some(repository) = state.repository() {
        return repository
            .candidate_evaluation()
            .await
            .map(Json)
            .map_err(|error| {
                if error.contains("no active") {
                    ApiError::NotFound(error)
                } else {
                    ApiError::Internal(error)
                }
            });
    }
    let store = state.read_store()?;
    let cycle_id = active_cycle_id(&store)
        .ok_or_else(|| ApiError::NotFound("no active learning cycle".to_owned()))?;
    let cycle = store
        .learning_cycles
        .get(&cycle_id)
        .ok_or_else(|| ApiError::NotFound("no active learning cycle".to_owned()))?;
    if !matches!(cycle.state.as_str(), "EVALUATE" | "DECISION") {
        return Err(ApiError::Conflict(format!(
            "cycle {} has no evaluation to review from state {}",
            cycle.id, cycle.state
        )));
    }
    let cycle = cycle.clone();
    let production_model_version = cycle
        .production_model_version
        .as_deref()
        .unwrap_or("unconfigured-production");
    let dataset_version = cycle
        .frozen_evaluation_dataset_version
        .as_deref()
        .unwrap_or("unconfigured-evaluation-dataset");
    let evaluation_status = if cycle.state == "EVALUATE" {
        "PENDING"
    } else {
        "FAILED"
    };
    let status_reason = if cycle.state == "EVALUATE" {
        "SHADOW_WINDOW_OPEN"
    } else {
        "IN_MEMORY_EVALUATION_UNAVAILABLE"
    };
    let policy_thresholds = if cycle.promotion_policy_version == "policy-v1" {
        json!({
            "minimum_offline_samples": 30,
            "minimum_shadow_samples": 20,
            "maximum_macro_f1_regression": 0.02,
            "maximum_class_f1_regression": 0.05,
            "maximum_shadow_correction_rate_delta": 0.05,
            "maximum_shadow_inference_failures": 0
        })
    } else {
        json!({})
    };
    let offline_threshold = policy_thresholds
        .get("minimum_offline_samples")
        .cloned()
        .unwrap_or(Value::Null);
    let shadow_threshold = policy_thresholds
        .get("minimum_shadow_samples")
        .cloned()
        .unwrap_or(Value::Null);
    let pending_model_evaluation = |evaluation_id: String, model_version: &str| {
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
            "sample_count": 0,
            "metrics": {"reason": status_reason},
            "critical_regressions": [],
            "created_at": cycle.updated_at,
            "evaluator": "pulse109.core.demo-status",
            "synthetic": true
        })
    };
    Ok(Json(json!({
        "schema_version": "candidate-evaluation.v1",
        "status": evaluation_status,
        "cycle_id": cycle.id,
        "candidate_model_version": cycle.candidate_model_version,
        "production_model_version": production_model_version,
        "candidate_dataset_version": cycle.dataset_version,
        "evaluation_version": "candidate-evaluation.v1",
        "policy_version": cycle.promotion_policy_version,
        "promotion_policy": {
            "version": cycle.promotion_policy_version,
            "thresholds": policy_thresholds
        },
        "offline_evaluation": pending_model_evaluation(
            format!("demo-candidate-offline-{}", cycle.id),
            &cycle.candidate_model_version,
        ),
        "baseline_evaluation": pending_model_evaluation(
            format!("demo-production-offline-{}", cycle.id),
            production_model_version,
        ),
        "shadow_evaluation": {
            "sample_count": 0,
            "agreement_with_confirmed": Value::Null,
            "correction_rate_delta": Value::Null,
            "critical_regressions": [],
            "metrics": {
                "decision_count": cycle.shadow_operator_decision_count,
                "candidate_inference_failures": cycle.shadow_inference_failures,
            },
            "blind_ab": if cycle.blind_ab_enabled { "ENABLED" } else { "DISABLED" }
        },
        "gates": [{
            "key": "candidate_evaluation_job",
            "status": if evaluation_status == "PENDING" { "PENDING" } else { "INSUFFICIENT_EVIDENCE" },
            "reason": status_reason
        }, {
            "key": "offline_sample_count",
            "status": "INSUFFICIENT_EVIDENCE",
            "observed": 0,
            "threshold": offline_threshold
        }, {
            "key": "shadow_sample_count",
            "status": "INSUFFICIENT_EVIDENCE",
            "observed": 0,
            "threshold": shadow_threshold
        }, {
            "key": "synthetic_evidence",
            "status": "INSUFFICIENT_EVIDENCE",
            "reason": "DEMO_SYNTHETIC_EVIDENCE_NOT_ELIGIBLE"
        }],
        "decision": "INSUFFICIENT_EVIDENCE",
        "evaluated_at": cycle.updated_at,
        "synthetic": true
    })))
}

fn active_cycle_id(store: &Store) -> Option<String> {
    store
        .learning_cycles
        .values()
        .find(|cycle| is_active_learning_cycle_state(&cycle.state))
        .map(|cycle| cycle.id.clone())
}

async fn promote_active_learning_cycle(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CycleDecisionRequest>,
) -> Result<Json<LearningCycle>, ApiError> {
    let actor = require_role(&headers, &state.config, &[Role::MlReviewer, Role::Admin])?;
    if let Some(repository) = state.repository() {
        let cycle_id = repository
            .learning_overview()
            .await
            .map_err(ApiError::Internal)?
            .active_cycle
            .ok_or_else(|| ApiError::NotFound("no active learning cycle".to_owned()))?
            .id;
        let cycle = repository
            .promote_learning_cycle(&cycle_id, request.note.as_deref(), &actor.user_id)
            .await
            .map_err(ApiError::Conflict)?;
        return Ok(Json(cycle));
    }
    let cycle_id = {
        let store = state.read_store()?;
        active_cycle_id(&store)
            .ok_or_else(|| ApiError::NotFound("no active learning cycle".to_owned()))?
    };
    promote_learning_cycle(State(state), headers, Path(cycle_id), Json(request)).await
}

async fn reject_active_learning_cycle(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CycleDecisionRequest>,
) -> Result<Json<LearningCycle>, ApiError> {
    let actor = require_role(&headers, &state.config, &[Role::MlReviewer, Role::Admin])?;
    if let Some(repository) = state.repository() {
        let cycle_id = repository
            .learning_overview()
            .await
            .map_err(ApiError::Internal)?
            .active_cycle
            .ok_or_else(|| ApiError::NotFound("no active learning cycle".to_owned()))?
            .id;
        let cycle = repository
            .reject_learning_cycle(&cycle_id, request.note.as_deref(), &actor.user_id)
            .await
            .map_err(ApiError::Conflict)?;
        return Ok(Json(cycle));
    }
    let cycle_id = {
        let store = state.read_store()?;
        active_cycle_id(&store)
            .ok_or_else(|| ApiError::NotFound("no active learning cycle".to_owned()))?
    };
    reject_learning_cycle(State(state), headers, Path(cycle_id), Json(request)).await
}

async fn promote_learning_cycle(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(cycle_id): Path<String>,
    Json(request): Json<CycleDecisionRequest>,
) -> Result<Json<LearningCycle>, ApiError> {
    let actor = require_role(&headers, &state.config, &[Role::MlReviewer, Role::Admin])?;
    if let Some(repository) = state.repository() {
        let cycle = repository
            .promote_learning_cycle(&cycle_id, request.note.as_deref(), &actor.user_id)
            .await
            .map_err(ApiError::Conflict)?;
        return Ok(Json(cycle));
    }
    let store = state.read_store()?;
    let cycle = store
        .learning_cycles
        .get(&cycle_id)
        .ok_or_else(|| ApiError::NotFound(format!("learning cycle {cycle_id} not found")))?;
    if cycle.state != "DECISION" {
        return Err(ApiError::Conflict(format!(
            "cycle {} cannot be promoted from state {}",
            cycle.id, cycle.state
        )));
    }
    Err(ApiError::Conflict(
        "candidate evaluation evidence is insufficient in the in-memory runtime".to_owned(),
    ))
}

async fn reject_learning_cycle(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(cycle_id): Path<String>,
    Json(request): Json<CycleDecisionRequest>,
) -> Result<Json<LearningCycle>, ApiError> {
    let actor = require_role(&headers, &state.config, &[Role::MlReviewer, Role::Admin])?;
    if let Some(repository) = state.repository() {
        let cycle = repository
            .reject_learning_cycle(&cycle_id, request.note.as_deref(), &actor.user_id)
            .await
            .map_err(ApiError::Conflict)?;
        return Ok(Json(cycle));
    }
    let mut store = state.write_store()?;
    let (candidate_model_version, rejected_cycle) = {
        let cycle = store
            .learning_cycles
            .get_mut(&cycle_id)
            .ok_or_else(|| ApiError::NotFound(format!("learning cycle {cycle_id} not found")))?;
        if !matches!(cycle.state.as_str(), "EVALUATE" | "DECISION") {
            return Err(ApiError::Conflict(format!(
                "cycle {} cannot be rejected from state {}",
                cycle.id, cycle.state
            )));
        }
        cycle.state = "REJECTED".to_owned();
        cycle.decision_note = Some(
            request
                .note
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| format!("Rejected by {}", actor.user_id)),
        );
        cycle.updated_at = DEMO_TIMESTAMP.to_owned();
        (cycle.candidate_model_version.clone(), cycle.clone())
    };
    if let Some(model) = store.models.get_mut(&candidate_model_version) {
        if matches!(
            model.status.as_str(),
            "candidate" | "shadow" | "CANDIDATE" | "SHADOW"
        ) {
            model.status = "rejected".to_owned();
        }
    }
    Ok(Json(rejected_cycle))
}

#[derive(Debug, Deserialize)]
pub struct ModelQuery {
    pub status: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub items: Vec<ModelVersion>,
    pub total: usize,
}

async fn list_models(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<ModelQuery>,
) -> Result<Json<ModelListResponse>, ApiError> {
    require_role(
        &headers,
        &state.config,
        &[Role::Manager, Role::MlReviewer, Role::Admin],
    )?;
    if let Some(repository) = state.repository() {
        let items = repository
            .list_models(&query)
            .await
            .map_err(ApiError::Internal)?;
        let total = items.len();
        return Ok(Json(ModelListResponse { items, total }));
    }
    let store = state.read_store()?;
    let items = store
        .models
        .values()
        .filter(|model| {
            query
                .status
                .as_deref()
                .is_none_or(|value| model.status == value)
        })
        .cloned()
        .collect::<Vec<_>>();
    let total = items.len();
    Ok(Json(ModelListResponse { items, total }))
}

async fn get_model(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(model_id): Path<String>,
) -> Result<Json<ModelVersion>, ApiError> {
    require_role(
        &headers,
        &state.config,
        &[Role::Manager, Role::MlReviewer, Role::Admin],
    )?;
    if let Some(repository) = state.repository() {
        return repository
            .get_model(&model_id)
            .await
            .map(Json)
            .map_err(|error| {
                if error.contains("not found") {
                    ApiError::NotFound(error)
                } else {
                    ApiError::Internal(error)
                }
            });
    }
    let store = state.read_store()?;
    store
        .models
        .get(&model_id)
        .cloned()
        .ok_or_else(|| ApiError::NotFound(format!("model {model_id} not found")))
        .map(Json)
}

async fn promote_model(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(model_id): Path<String>,
) -> Result<Json<ModelVersion>, ApiError> {
    let actor = require_role(&headers, &state.config, &[Role::MlReviewer, Role::Admin])?;
    if let Some(repository) = state.repository() {
        return repository
            .promote_model(&model_id, &actor.user_id)
            .await
            .map(Json)
            .map_err(|error| {
                if error.contains("not found") {
                    ApiError::NotFound(error)
                } else {
                    ApiError::Conflict(error)
                }
            });
    }
    let mut store = state.write_store()?;
    if !store.models.contains_key(&model_id) {
        return Err(ApiError::NotFound(format!("model {model_id} not found")));
    }
    if store
        .learning_cycles
        .values()
        .any(|cycle| cycle.candidate_model_version == model_id)
    {
        return Err(ApiError::Conflict(
            "candidate is managed by a learning cycle; use candidate promotion after all policy gates pass"
                .to_owned(),
        ));
    }
    for model in store.models.values_mut() {
        if model.status == "production" {
            model.status = "archived".to_owned();
        }
    }
    let model = store
        .models
        .get_mut(&model_id)
        .ok_or_else(|| ApiError::NotFound(format!("model {model_id} not found")))?;
    model.status = "production".to_owned();
    model.promoted_at = Some(DEMO_TIMESTAMP.to_owned());
    info!(
        service = SERVICE_NAME,
        model_version = %model_id,
        request_id = "n/a",
        trace_id = "n/a",
        endpoint = "/api/v1/models/{model_id}/promote",
        latency_ms = 0.0_f64,
        status = 200_u16,
        error_code = "none",
        result_state = "promoted",
        "model_promoted"
    );
    Ok(Json(model.clone()))
}

async fn openapi() -> Json<Value> {
    Json(json!({
        "openapi": "3.1.0",
        "info": {
            "title": "Pulse 109 Core API",
            "version": API_VERSION,
            "description": "P0 operator workflow and situation center API. Compose uses PostgreSQL/Qdrant/ML baseline; deterministic memory responses are test-only."
        },
        "servers": [{ "url": "/" }],
        "security": [{ "PulseRole": [] }, {}],
        "components": {
            "securitySchemes": {
                "PulseRole": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "x-pulse-role",
                    "description": "Demo/test role header. Production uses x-authenticated-role from the trusted auth gateway."
                }
            }
        },
        "paths": {
            "/healthz": { "get": { "summary": "Liveness" } },
            "/readyz": { "get": {
                "summary": "Dependency and model artifact readiness",
                "responses": {
                    "200": { "description": "Required dependencies are ready" },
                    "503": { "description": "A dependency or model artifact is not ready" }
                }
            } },
            "/api/v1/openapi.json": { "get": { "summary": "Core OpenAPI route index" } },
            "/api/v1/docs": { "get": { "summary": "Core OpenAPI route index" } },
            "/api/v1/tickets": { "get": { "summary": "List tickets" }, "post": { "summary": "Create ticket" } },
            "/api/v1/audit": { "get": { "summary": "Privacy-safe audit log for managers and admins" } },
            "/api/v1/tickets/{ticket_id}": { "get": { "summary": "Get ticket and prediction" } },
            "/api/v1/tickets/{ticket_id}/prediction": { "get": { "summary": "Get current ticket prediction" } },
            "/api/v1/tickets/{ticket_id}/vector": { "delete": { "summary": "Delete one Qdrant vector" } },
            "/api/v1/tickets/{ticket_id}/pulse-state": { "put": { "summary": "Store human-confirmed Pulse state" } },
            "/api/v1/import": { "post": { "summary": "Import validated tickets into PostgreSQL and Qdrant" } },
            "/api/v1/datasets/provenance": { "get": { "summary": "Read synthetic, real and unassigned dataset counts" } },
            "/api/v1/assist/preview": { "post": { "summary": "Preview prediction and related tickets" } },
            "/api/v1/response-templates": { "get": { "summary": "List response templates for managers" }, "post": { "summary": "Create an unapproved response template version" } },
            "/api/v1/response-templates/import": { "post": { "summary": "Import unapproved response template versions" } },
            "/api/v1/response-templates/{template_id}": { "get": { "summary": "Get response template details" }, "put": { "summary": "Edit a pending response template" }, "delete": { "summary": "Delete a pending response template" } },
            "/api/v1/response-templates/{template_id}/approve": { "post": { "summary": "Explicitly approve a response template version" } },
            "/api/v1/assist/{ticket_id}/confirm": { "post": { "summary": "Confirm prediction" } },
            "/api/v1/assist/{ticket_id}/correct": { "post": { "summary": "Correct prediction" } },
            "/api/v1/assist/confirm": { "post": { "summary": "Confirm prediction using ticket_id in body" } },
            "/api/v1/assist/correct": { "post": { "summary": "Correct prediction using ticket_id in body" } },
            "/api/v1/assist/confirm/{ticket_id}": { "post": { "summary": "Confirm prediction alias" } },
            "/api/v1/assist/correct/{ticket_id}": { "post": { "summary": "Correct prediction alias" } },
            "/api/v1/analytics": { "get": { "summary": "Situation center analytics" } },
            "/api/v1/analytics/overview": { "get": { "summary": "Situation center overview" } },
            "/api/v1/analytics/drilldown": { "get": { "summary": "Drill analytics metrics down to privacy-safe ticket summaries" } },
            "/api/v1/taxonomy": { "get": { "summary": "Current regions, topics, services and available ticket filters" } },
            "/api/v1/analytics/query": { "post": { "summary": "Validated QueryIntent analytics" } },
            "/api/v1/retrieval/reindex": { "post": { "summary": "Queue a PostgreSQL-backed Qdrant reindex" } },
            "/api/v1/analytics/export.pdf": { "get": { "summary": "Export ReportSlice as PDF" } },
            "/api/v1/analytics/export.xlsx": { "get": { "summary": "Export ReportSlice as XLSX" } },
            "/api/v1/reports": { "get": { "summary": "List generated reports" } },
            "/api/v1/events": { "get": { "summary": "SSE notifications" } },
            "/api/v1/forecast": { "get": { "summary": "Forecast baseline" } },
            "/api/v1/alerts": { "get": { "summary": "List alerts" } },
            "/api/v1/alerts/detect": { "post": { "summary": "Detect and persist region/topic anomaly alerts" } },
            "/api/v1/alerts/{alert_id}": { "get": { "summary": "Get alert detail and linked tickets" } },
            "/api/v1/alerts/{alert_id}/ack": { "post": { "summary": "Acknowledge alert" } },
            "/api/v1/alerts/{alert_id}/acknowledge": { "post": { "summary": "Acknowledge alert alias" } },
            "/api/v1/alerts/{alert_id}/close": { "post": { "summary": "Close alert" } },
            "/api/v1/learning": { "get": { "summary": "Learning loop status" }, "post": { "summary": "Start candidate cycle" } },
            "/api/v1/learning/{cycle_id}": { "get": { "summary": "Get learning cycle" } },
            "/api/v1/learning/{cycle_id}/feedback": { "post": { "summary": "Add validated learning feedback" } },
            "/api/v1/learning/{cycle_id}/promote": { "post": { "summary": "Promote candidate after human review" } },
            "/api/v1/learning/{cycle_id}/reject": { "post": { "summary": "Reject candidate after human review" } },
            "/api/v1/learning/cycle": { "get": { "summary": "Collect learning feedback" } },
            "/api/v1/learning/cycle/close": { "post": { "summary": "Close collect and train candidate" } },
            "/api/v1/learning/candidate/evaluation": { "get": { "summary": "Read candidate evaluation evidence and promotion gates" } },
            "/api/v1/learning/candidate/promote": { "post": { "summary": "Promote candidate" } },
            "/api/v1/learning/candidate/reject": { "post": { "summary": "Reject candidate" } },
            "/api/v1/tickets/{ticket_id}/relation-feedback": { "post": { "summary": "Collect relation feedback" } },
            "/api/v1/models": { "get": { "summary": "List model versions" } },
            "/api/v1/models/{model_id}": { "get": { "summary": "Get model version" } },
            "/api/v1/models/{model_id}/promote": { "post": { "summary": "Promote a model version outside controlled learning cycles" } }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{to_bytes, Body},
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    #[test]
    fn provenance_serializes_only_captured_routing_facts() {
        let empty = serde_json::to_value(RuleProvenance::manual("Ручная проверка")).unwrap();
        assert!(empty.get("facts_used").is_none());

        let provenance = RuleProvenance::manual("Проверено по входным данным")
            .with_fact(ExplainabilityFactField::TopicId, "TOPIC-WATER")
            .with_fact(ExplainabilityFactField::RegionId, "KZ-ASTANA");
        let value = serde_json::to_value(provenance).unwrap();
        assert_eq!(
            value["facts_used"],
            json!([
                {"field": "topic_id", "value": "TOPIC-WATER"},
                {"field": "region_id", "value": "KZ-ASTANA"}
            ])
        );
    }

    async fn body_json(response: Response) -> Value {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn report_exports_render_the_slice_without_ticket_text_or_closed_alerts() {
        let mut store = Store::demo();
        let ticket_id = store.tickets.keys().next().unwrap().clone();
        store.tickets.get_mut(&ticket_id).unwrap().text =
            "REPORT_PRIVATE_TICKET_SENTINEL".to_owned();
        store.alerts.get_mut("alert-001").unwrap().status = "CLOSED".to_owned();
        let query = AnalyticsQuery {
            region_id: None,
            topic_id: None,
            service_id: None,
            status: None,
            district: None,
            channel: None,
            range: Some("30d".to_owned()),
        };
        let report = memory_report_slice(&store, &query).unwrap();
        let html = report_html(Some(&report));
        let xlsx = xlsx_report(Some(&report));

        assert!(html.contains("generated_at") || html.contains("Сформировано"));
        assert!(html.contains("<svg"));
        assert!(html.contains("<table"));
        assert!(report
            .alerts
            .iter()
            .all(|alert| !alert.status.eq_ignore_ascii_case("CLOSED")));
        assert!(!html.contains("REPORT_PRIVATE_TICKET_SENTINEL"));
        assert!(!String::from_utf8_lossy(&xlsx).contains("REPORT_PRIVATE_TICKET_SENTINEL"));
        assert!(!String::from_utf8_lossy(&xlsx).contains(&ticket_id));
    }

    #[tokio::test]
    async fn health_and_readiness_are_available_without_auth() {
        let app = app(AppState::demo());
        let trace_input = "PULSE109_TRACE_SENTINEL";
        let response = app
            .clone()
            .oneshot(
                Request::get("/healthz")
                    .header("x-trace-id", trace_input)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-trace-id"], safe_trace_id(trace_input));
        assert_eq!(body_json(response).await["status"], "ok");

        let response = app
            .oneshot(Request::get("/readyz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let readiness = body_json(response).await;
        assert_eq!(readiness["status"], "ready");
        assert_eq!(readiness["checks"]["store"]["status"], "ready");
        assert_eq!(readiness["checks"]["qdrant"]["status"], "not_applicable");
    }

    #[tokio::test]
    async fn readiness_returns_safe_dependency_details_when_unavailable() {
        let repository = PgRepository::connect_lazy_with_options(
            "postgres://pulse:private-test-value@127.0.0.1:1/pulse",
            "http://127.0.0.1:1",
            "http://127.0.0.1:1",
            "test_collection",
            32,
            "test_embedder",
        )
        .unwrap();
        let mut state = AppState::demo();
        state.repository = Some(Arc::new(repository));
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            app(state).oneshot(Request::get("/readyz").body(Body::empty()).unwrap()),
        )
        .await
        .expect("readiness should have bounded local failure time")
        .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = body_json(response).await;
        assert_eq!(body["status"], "not_ready");
        assert_eq!(body["checks"]["postgres"]["status"], "not_ready");
        assert_eq!(
            body["checks"]["postgres"]["error_code"],
            "POSTGRES_UNAVAILABLE"
        );
        assert_eq!(
            body["checks"]["database_migrations"]["status"],
            "not_evaluated"
        );
        assert!(!body.to_string().contains("private-test-value"));
    }

    #[tokio::test]
    async fn preview_and_decision_keep_prediction_separate() {
        let app = app(AppState::demo());
        let request = Request::post("/api/v1/assist/preview")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"ticket_id":"ticket-001"}"#))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let preview = body_json(response).await;
        assert_eq!(preview["prediction"]["topic_id"], "TOPIC-WATER");

        let request = Request::post("/api/v1/assist/ticket-001/correct")
            .header("content-type", "application/json")
            .header("x-pulse-role", "OPERATOR")
            .header("x-user-id", "operator-test")
            .body(Body::from(
                r#"{"topic_id":"TOPIC-UTILITIES","note":"operator correction"}"#,
            ))
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let decision = body_json(response).await;
        assert_eq!(decision["decision"]["action"], "correct");
        assert_eq!(
            decision["decision"]["confirmed_topic_id"],
            "TOPIC-UTILITIES"
        );
        assert_eq!(decision["prediction"]["topic_id"], "TOPIC-WATER");
    }

    #[tokio::test]
    async fn explicit_roles_are_enforced() {
        let app = app(AppState::demo());
        let request = Request::get("/api/v1/analytics")
            .header("x-pulse-role", "OPERATOR")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[test]
    fn development_auth_cannot_be_enabled_outside_demo_environments() {
        assert_eq!(resolve_dev_auth(Some(true), false, true), Ok(true));
        assert_eq!(resolve_dev_auth(Some(false), true, false), Ok(false));
        assert_eq!(resolve_dev_auth(None, false, false), Ok(false));
        assert_eq!(
            resolve_dev_auth(Some(true), false, false),
            Err("PULSE_DEV_AUTH=true is only allowed in demo, test, or unit environments")
        );
    }

    #[tokio::test]
    async fn tickets_are_deterministic_and_paginated() {
        let app = app(AppState::demo());
        let response = app
            .oneshot(
                Request::get("/api/v1/tickets?limit=2&offset=1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let value = body_json(response).await;
        assert_eq!(value["total"], 12);
        assert_eq!(value["items"].as_array().unwrap().len(), 2);
        assert_eq!(value["items"][0]["id"], "ticket-002");
    }

    #[test]
    fn audit_log_event_serializes_only_allowlisted_fields() {
        let event = pg::AuditLogEvent {
            id: 7,
            actor_id: Some("operator-17".to_owned()),
            action: "READ_TICKET_DETAIL".to_owned(),
            entity_type: "ticket".to_owned(),
            entity_id: Some("ticket-001".to_owned()),
            request_id: Some("core-123-4".to_owned()),
            created_at: DateTime::parse_from_rfc3339("2026-09-26T12:00:00Z")
                .unwrap()
                .with_timezone(&Utc),
        };
        let value = serde_json::to_value(event).unwrap();
        let fields = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();

        assert_eq!(
            fields,
            [
                "action",
                "actor_id",
                "created_at",
                "entity_id",
                "entity_type",
                "id",
                "request_id",
            ]
        );
        assert!(value.get("reason").is_none());
        assert!(value.get("metadata").is_none());
    }

    #[test]
    fn audit_request_id_only_exposes_core_generated_identifiers() {
        assert_eq!(
            pg::safe_audit_request_id(Some("core-123-4".to_owned())),
            Some("core-123-4".to_owned())
        );
        assert_eq!(
            pg::safe_audit_request_id(Some("user-supplied-request".to_owned())),
            None
        );
        assert_eq!(pg::safe_audit_request_id(None), None);
    }

    #[test]
    fn ticket_list_audit_metadata_omits_ticket_text() {
        let store = Store::demo();
        let mut ticket = store.tickets.values().next().unwrap().clone();
        ticket.text = "AUDIT_PRIVATE_TICKET_SENTINEL".to_owned();
        let ticket_id = ticket.id.clone();
        let metadata = ticket_list_audit_metadata(&TicketListResponse {
            items: vec![ticket],
            total: 1,
            limit: 1,
            offset: 0,
        });

        assert_eq!(metadata["ticket_ids"][0], ticket_id);
        assert!(!metadata
            .to_string()
            .contains("AUDIT_PRIVATE_TICKET_SENTINEL"));
    }

    #[tokio::test]
    async fn openapi_documents_core_routes() {
        let app = app(AppState::demo());
        let response = app
            .oneshot(
                Request::get("/api/v1/openapi.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let value = body_json(response).await;
        assert_eq!(value["openapi"], "3.1.0");
        assert!(value["paths"]["/api/v1/audit"].is_object());
        assert!(value["paths"]["/api/v1/assist/preview"].is_object());
        assert!(value["paths"]["/api/v1/learning/{cycle_id}/feedback"].is_object());
        assert!(value["paths"]["/api/v1/models/{model_id}/promote"].is_object());
        assert!(value["paths"]["/readyz"]["get"]["responses"]["503"].is_object());
        assert!(value["paths"]["/internal/v1/classify"].is_null());
    }

    #[tokio::test]
    async fn pulse_state_and_relation_feedback_are_persisted() {
        let app = app(AppState::demo());
        let response = app
            .clone()
            .oneshot(
                Request::put("/api/v1/tickets/ticket-001/pulse-state")
                    .header("content-type", "application/json")
                    .header("x-pulse-role", "OPERATOR")
                    .body(Body::from(
                        r#"{"operator_confirmed_decision":{"action":"confirm","topic_id":"TOPIC-WATER"},"feedback":{"comment":"operator accepted"}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let decision = body_json(response).await;
        assert_eq!(decision["decision"]["action"], "confirm");
        assert_eq!(
            decision["learning_feedback_status"],
            "NO_ACTIVE_COLLECT_CYCLE"
        );

        let cycle = app
            .clone()
            .oneshot(
                Request::get("/api/v1/learning/cycle")
                    .header("x-pulse-role", "ML_REVIEWER")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(cycle.status(), StatusCode::OK);
        assert_eq!(body_json(cycle).await["active_cycle"]["feedback_count"], 3);

        let relation = app
            .oneshot(
                Request::post("/api/v1/tickets/ticket-001/relation-feedback")
                    .header("content-type", "application/json")
                    .header("x-pulse-role", "OPERATOR")
                    .body(Body::from(
                        r#"{"related_ticket_id":"ticket-002","relation":"duplicate","decision":"rejected"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(relation.status(), StatusCode::CREATED);
        assert_eq!(
            body_json(relation).await["feedback_type"],
            "relation:DUPLICATE:REJECTED"
        );
    }

    #[tokio::test]
    async fn report_exports_have_expected_formats() {
        let app = app(AppState::demo());
        let response = app
            .clone()
            .oneshot(
                Request::get("/api/v1/analytics/export.pdf")
                    .header("x-pulse-role", "MANAGER")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/pdf");
        let pdf = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        assert!(pdf.starts_with(b"%PDF-"));
        assert!(pdf.len() > 1_000);
        assert!(pdf[pdf.len().saturating_sub(1_024)..]
            .windows(5)
            .any(|window| window == b"%%EOF"));

        let response = app
            .oneshot(
                Request::get("/api/v1/analytics/export.xlsx")
                    .header("x-pulse-role", "MANAGER")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        );
        assert!(to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .starts_with(b"PK\x03\x04"));
    }

    #[tokio::test]
    async fn query_intent_accepts_strict_filters_and_rejects_unknown_intents() {
        let app = app(AppState::demo());
        let response = app
            .clone()
            .oneshot(
                Request::post("/api/v1/analytics/query")
                    .header("content-type", "application/json")
                    .header("x-pulse-role", "MANAGER")
                    .body(Body::from(
                        r#"{"intent":"compare","filters":{"region_id":"R01","range":"7d"},"limit":5}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let value = body_json(response).await;
        assert_eq!(value["intent"], "compare_regions");
        assert_eq!(value["filters"]["region_id"], "R01");
        assert!(value["table"].as_array().is_some());

        for (question, expected_intent) in [
            ("Сколько обращений за последние 30 дней?", "count"),
            ("Покажи динамику обращений за 30 дней", "trend"),
            ("Сравни обращения по регионам", "compare_regions"),
            ("Топ тем обращений", "top_topics"),
            ("Найди всплески обращений", "spikes"),
            ("Прогноз обращений на 30 дней", "forecast"),
        ] {
            let mut payload = json!({"text": question, "range": "30d", "limit": 8});
            if expected_intent == "forecast" {
                payload["horizon_days"] = json!(30);
            }
            let response = app
                .clone()
                .oneshot(
                    Request::post("/api/v1/analytics/query")
                        .header("content-type", "application/json")
                        .header("x-pulse-role", "MANAGER")
                        .body(Body::from(payload.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "{question}");
            let value = body_json(response).await;
            assert_eq!(value["intent"], expected_intent, "{question}");
            assert!(value["summary"]["text"].as_str().is_some(), "{question}");
            assert!(value["table"].as_array().is_some(), "{question}");
            assert!(value["table_columns"].as_array().is_some(), "{question}");
            assert!(value["series"].as_array().is_some(), "{question}");
            assert!(value["chart"]["type"].as_str().is_some(), "{question}");
            assert_eq!(value["filters"]["range"], "30d", "{question}");
            assert_eq!(value["period"]["days"], 30, "{question}");
            assert!(
                value["comparison_definition"].as_str().is_some(),
                "{question}"
            );
            assert!(value["source"].as_str().is_some(), "{question}");
        }

        let response = app
            .clone()
            .oneshot(
                Request::post("/api/v1/analytics/query")
                    .header("content-type", "application/json")
                    .header("x-pulse-role", "MANAGER")
                    .body(Body::from(r#"{"intent":"drop_table"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        for invalid_payload in [
            r#"{"intent":"count","filters":{"region_id":"R01","unexpected":"x"}}"#,
            r#"{"intent":"count","region_id":"R01","filters":{"region_id":"R02"}}"#,
            r#"{"intent":"trend","group_by":"district"}"#,
            r#"{"intent":"count","range":"0d"}"#,
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::post("/api/v1/analytics/query")
                        .header("content-type", "application/json")
                        .header("x-pulse-role", "MANAGER")
                        .body(Body::from(invalid_payload))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert!(
                matches!(
                    response.status(),
                    StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY
                ),
                "unexpected status {} for {invalid_payload}",
                response.status()
            );
        }
    }

    #[tokio::test]
    async fn forecast_supports_required_horizons() {
        let state = AppState::demo();
        {
            let mut store = state.store.write().unwrap();
            let template = store.tickets.values().next().unwrap().clone();
            store.tickets.clear();
            let as_of = DateTime::parse_from_rfc3339(DEMO_TIMESTAMP)
                .unwrap()
                .with_timezone(&Utc);
            for day in 1..=FORECAST_SEASON_LENGTH_DAYS {
                let mut ticket = template.clone();
                ticket.id = format!("forecast-fixture-{day}");
                ticket.created_at = (as_of - Duration::days(day as i64)).to_rfc3339();
                store.tickets.insert(ticket.id.clone(), ticket);
            }
        }
        let app = app(state);
        for horizon in [30, 60, 90] {
            let response = app
                .clone()
                .oneshot(
                    Request::get(format!("/api/v1/forecast?horizon={horizon}"))
                        .header("x-pulse-role", "MANAGER")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let value = body_json(response).await;
            assert_eq!(value["horizon_days"], horizon);
            assert_eq!(value["points"].as_array().unwrap().len(), horizon as usize);
            assert_eq!(value["forecast_start"], value["points"][0]["date"]);
        }
    }

    #[test]
    fn qdrant_collection_is_stable_for_baseline_and_versioned_for_new_embedder() {
        assert_eq!(
            default_qdrant_collection("embedder-demo-2026-09-21-001", 32),
            "pulse109_tickets_v1"
        );
        assert_eq!(
            default_qdrant_collection("e5-multilingual-v2", 768),
            "pulse109_e5_multilingual_v2_d768"
        );
    }

    #[test]
    fn related_candidates_are_thresholded_and_labeled_only_by_known_factors() {
        let current_date = DateTime::parse_from_rfc3339("2026-09-21T08:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let previous_date = DateTime::parse_from_rfc3339("2026-09-10T08:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let weak = related_ticket_candidate(
            "ticket-weak".to_owned(),
            0.77,
            "TOPIC-ROADS",
            "R01",
            Some(&current_date),
            "TOPIC-ROADS".to_owned(),
            "R01".to_owned(),
            Some(&previous_date),
            "embedder-test-v1",
            "Cosine",
        );
        assert!(weak.is_none());

        let duplicate = related_ticket_candidate(
            "ticket-duplicate".to_owned(),
            0.95,
            "TOPIC-ROADS",
            "R01",
            Some(&current_date),
            "TOPIC-ROADS".to_owned(),
            "R01".to_owned(),
            Some(&previous_date),
            "embedder-test-v1",
            "Cosine",
        )
        .unwrap();
        assert_eq!(duplicate.relation, "duplicate");
        assert_eq!(
            duplicate.matched_factors,
            ["topic_match", "region_match", "within_30_days"]
        );
        assert_eq!(
            duplicate.suggestion.threshold,
            DUPLICATE_CANDIDATE_THRESHOLD
        );
        assert_eq!(duplicate.suggestion.rule_version, "related-ticket-rules.v1");
        assert_eq!(duplicate.suggestion.model_version, "embedder-test-v1");
        assert_eq!(duplicate.suggestion.distance_metric, "Cosine");

        let repeat = related_ticket_candidate(
            "ticket-repeat".to_owned(),
            0.82,
            "TOPIC-ROADS",
            "R01",
            Some(&current_date),
            "TOPIC-ROADS".to_owned(),
            "R02".to_owned(),
            Some(&previous_date),
            "embedder-test-v1",
            "Cosine",
        )
        .unwrap();
        assert_eq!(repeat.relation, "repeat");
        assert_eq!(repeat.matched_factors, ["topic_match", "within_30_days"]);
        assert_eq!(repeat.suggestion.threshold, RELATED_CANDIDATE_THRESHOLD);
    }
}
