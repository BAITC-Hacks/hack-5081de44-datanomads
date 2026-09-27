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
use chrono::{DateTime, Duration, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    convert::Infallible,
    env,
    sync::{Arc, RwLock},
    time::{Duration as StdDuration, Instant},
};
use thiserror::Error;
use tokio::sync::broadcast;
use tokio_stream::{wrappers::BroadcastStream, StreamExt};
use tower_http::cors::CorsLayer;
use tracing::info;

mod pg;
use pg::{default_qdrant_collection, PgRepository};

const SERVICE_NAME: &str = "pulse109-core";
const API_VERSION: &str = "0.1.0";
const DEMO_TIMESTAMP: &str = "2026-09-21T08:00:00Z";
pub(crate) const RELATED_CANDIDATE_THRESHOLD: f32 = 0.78;
pub(crate) const DUPLICATE_CANDIDATE_THRESHOLD: f32 = 0.90;
const RELATED_CANDIDATE_RULE_VERSION: &str = "related-ticket-rules.v1";

/// Runtime configuration.  `dev_auth` is enabled by default for the local
/// deterministic demo: a request without `x-pulse-role` acts as ADMIN, while
/// an explicitly supplied role is always checked.
#[derive(Clone, Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub dev_auth: bool,
    pub storage: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_owned(),
            port: 8080,
            dev_auth: true,
            storage: "memory".to_owned(),
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
        Self {
            host: env::var("PULSE_HOST").unwrap_or(defaults.host),
            port: env::var("PULSE_PORT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(defaults.port),
            dev_auth: env::var("PULSE_DEV_AUTH")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .map(|value| !matches!(value.to_ascii_lowercase().as_str(), "0" | "false" | "no"))
                .unwrap_or(default_dev_auth && defaults.dev_auth),
            storage,
        }
    }
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
}

impl RuleProvenance {
    pub fn manual(reason: impl Into<String>) -> Self {
        Self {
            source: RuleSource::Manual,
            version: None,
            reason: reason.into(),
        }
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
    pub linked_ticket_ids: Vec<String>,
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
    pub state: String,
    pub dataset_version: String,
    pub candidate_model_version: String,
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

fn trace_id_from_headers(headers: &HeaderMap) -> String {
    headers
        .get("x-trace-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| request_id_from_headers(headers))
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
                "message": self.to_string(),
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
        service_provenance: RuleProvenance::manual(
            "Демонстрационное сопоставление; официальные правила 109 не предоставлены",
        ),
        priority_provenance: RuleProvenance::manual(
            "Демонстрационный приоритет; официальные правила 109 не предоставлены",
        ),
        alternatives,
        created_at: ticket.created_at.clone(),
    }
}

fn response_template(language: &str, topic_id: &str) -> ResponseTemplate {
    let title = format!("Обращение: {}", topic_label(&demo_topics(), topic_id));
    let body = if language == "kk" {
        "Өтінішіңіз тіркелді. Жауапты қызметке жолданды, мәртебені осы арнадан қадағалай аласыз."
    } else {
        "Ваше обращение зарегистрировано и направлено в ответственную службу. Статус можно отслеживать в этом канале."
    };
    ResponseTemplate {
        id: format!("template-{}-{}", language, topic_id.to_ascii_lowercase()),
        title,
        body: body.to_owned(),
        language: language.to_owned(),
        approved: false,
        source: "MANUAL_DEMO".to_owned(),
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
                description: "Количество обращений выше сезонного baseline в 2.4 раза.".to_owned(),
                region_id: "R01".to_owned(),
                topic_id: "TOPIC-WATER".to_owned(),
                ticket_count: 18,
                linked_ticket_ids: vec!["ticket-001".to_owned()],
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
                description: "Рост повторных обращений по маршруту 12.".to_owned(),
                region_id: "R02".to_owned(),
                topic_id: "TOPIC-TRANSPORT".to_owned(),
                ticket_count: 9,
                linked_ticket_ids: vec!["ticket-005".to_owned()],
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
            state: "EVALUATE".to_owned(),
            dataset_version: "dataset-demo-2026-09-001".to_owned(),
            candidate_model_version: "classifier-candidate-2026-09-001".to_owned(),
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
    let trace_id = request
        .headers()
        .get("x-trace-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(&request_id)
        .to_owned();
    let started = Instant::now();
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    request.extensions_mut().insert(request_id.clone());
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
    info!(
        service = SERVICE_NAME,
        request_id = %request_id,
        trace_id = %trace_id,
        endpoint = %path,
        method = %method,
        latency_ms,
        status = response.status().as_u16(),
        model_version = "n/a",
        error_code = "none",
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

async fn readyz(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    if let Some(repository) = state.repository() {
        let checks = repository
            .readiness()
            .await
            .map_err(ApiError::Unavailable)?;
        return Ok(Json(json!({
            "status": "ready",
            "service": SERVICE_NAME,
            "storage": "postgres",
            "demo": false,
            "checks": checks,
        })));
    }
    let store = state.read_store()?;
    Ok(Json(json!({
        "status": "ready",
        "service": SERVICE_NAME,
        "storage": "in_memory_demo",
        "demo": true,
        "checks": {
            "store": true,
            "tickets": store.tickets.len(),
            "models": store.models.len(),
        }
    })))
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
    require_role(
        &headers,
        &state.config,
        &[Role::Operator, Role::Manager, Role::Admin],
    )?;
    if let Some(repository) = state.repository() {
        let response = repository
            .list_tickets(&query)
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
            user_id = %actor.user_id,
            ticket_id = %response.ticket.id,
            model_version = %response.prediction.model_version,
            request_id = %request_id_from_headers(&headers),
            trace_id = %request_id_from_headers(&headers),
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
        user_id = %actor.user_id,
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
        user_id = %actor.user_id,
        source_system = %response.source_system,
        import_run_id = %response.import_run_id,
        imported_rows = response.imported_rows,
        quarantined_rows = response.quarantined_rows,
        request_id = %request_id_from_headers(&headers),
        trace_id = %request_id_from_headers(&headers),
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

async fn get_ticket(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(ticket_id): Path<String>,
) -> Result<Json<TicketDetailResponse>, ApiError> {
    require_role(
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
                Some(&request_id),
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
    let template_topic = latest_decision
        .map(|decision| decision.confirmed_topic_id.as_str())
        .unwrap_or(&prediction.topic_id);
    let template_started = Instant::now();
    let response_template = if uncertain_language {
        unavailable_demo_response_template(&language_state)
    } else {
        response_template(&ticket.language, template_topic)
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
            } else {
                "completed"
            }
            .to_owned(),
            latency_ms: template_latency_ms,
            model_version: None,
            error_code: uncertain_language.then(|| "LANGUAGE_UNCERTAIN".to_owned()),
        },
    ];
    let partial = stages
        .iter()
        .any(|stage| matches!(stage.status.as_str(), "unavailable" | "skipped" | "unknown"));
    let needs_review = !has_human_decision
        || uncertain_language
        || prediction.confidence < 0.85
        || response_template.source != "AUTHORITATIVE"
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
            user_id = %actor.user_id,
            model_version = %response.prediction.model_version,
            request_id = %request_id_from_headers(&headers),
            trace_id = %request_id_from_headers(&headers),
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
    if let Some(cycle_id) = store
        .learning_cycles
        .values()
        .find(|cycle| !matches!(cycle.state.as_str(), "PROMOTED" | "REJECTED"))
        .map(|cycle| cycle.id.clone())
    {
        let feedback = LearningFeedback {
            id: format!("feedback-{:03}", store.next_feedback_number),
            ticket_id: ticket_id.clone(),
            cycle_id: Some(cycle_id.clone()),
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
        if let Some(cycle) = store.learning_cycles.get_mut(&cycle_id) {
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
        user_id = %decision.user_id,
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
    let filtered: Vec<&Ticket> = store
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
        })
        .collect();
    let avg_confidence = if filtered.is_empty() {
        0.0
    } else {
        filtered
            .iter()
            .filter_map(|ticket| store.predictions.get(&ticket.id))
            .map(|prediction| prediction.confidence)
            .sum::<f32>()
            / filtered.len() as f32
    };
    let open_tickets = filtered
        .iter()
        .filter(|ticket| ticket.status == "open")
        .count();
    let resolved = filtered
        .iter()
        .filter(|ticket| ticket.status == "resolved")
        .count();
    let high_priority = filtered
        .iter()
        .filter(|ticket| ticket.priority == "high")
        .count();
    let runtime_metrics = demo_runtime_metrics(&store, &filtered);
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
    });
    let by_region = store
        .regions
        .iter()
        .filter(|region| query.region_id.as_deref().is_none_or(|id| id == region.id))
        .map(|region| {
            metric_bucket(
                region.id.clone(),
                region.name.clone(),
                filtered
                    .iter()
                    .filter(|ticket| ticket.region_id == region.id)
                    .copied()
                    .collect(),
                &store,
            )
        })
        .collect();
    let by_topic = store
        .topics
        .iter()
        .filter(|topic| query.topic_id.as_deref().is_none_or(|id| id == topic.id))
        .map(|topic| {
            metric_bucket(
                topic.id.clone(),
                topic.label.clone(),
                filtered
                    .iter()
                    .filter(|ticket| ticket.topic_id == topic.id)
                    .copied()
                    .collect(),
                &store,
            )
        })
        .collect();
    Ok(Json(AnalyticsResponse {
        generated_at: DEMO_TIMESTAMP.to_owned(),
        source: "deterministic-demo".to_owned(),
        range: query.range.unwrap_or_else(|| "7d".to_owned()),
        overview,
        runtime_metrics,
        by_region,
        by_topic,
        time_series: vec![
            TimeSeriesPoint {
                date: "2026-09-15".to_owned(),
                tickets: 7,
                resolved: 4,
            },
            TimeSeriesPoint {
                date: "2026-09-16".to_owned(),
                tickets: 9,
                resolved: 5,
            },
            TimeSeriesPoint {
                date: "2026-09-17".to_owned(),
                tickets: 8,
                resolved: 6,
            },
            TimeSeriesPoint {
                date: "2026-09-18".to_owned(),
                tickets: 12,
                resolved: 7,
            },
            TimeSeriesPoint {
                date: "2026-09-19".to_owned(),
                tickets: 11,
                resolved: 8,
            },
            TimeSeriesPoint {
                date: "2026-09-20".to_owned(),
                tickets: 14,
                resolved: 9,
            },
            TimeSeriesPoint {
                date: "2026-09-21".to_owned(),
                tickets: 10,
                resolved: 6,
            },
        ],
    }))
}

async fn analytics_drilldown(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<AnalyticsDrilldownQuery>,
) -> Result<Json<TicketListResponse>, ApiError> {
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
    let value = query.value.as_deref();
    let mut items = store
        .tickets
        .values()
        .filter(|ticket| {
            query
                .filters
                .region_id
                .as_deref()
                .is_none_or(|filter| ticket.region_id == filter)
                && query
                    .filters
                    .topic_id
                    .as_deref()
                    .is_none_or(|filter| ticket.topic_id == filter)
                && match dimension.as_str() {
                    "region" => value.is_none_or(|filter| ticket.region_id == filter),
                    "topic" => value.is_none_or(|filter| ticket.topic_id == filter),
                    _ => true,
                }
        })
        .cloned()
        .collect::<Vec<_>>();
    let total = items.len();
    let limit = query.limit.unwrap_or(100).clamp(1, 100);
    let offset = query.offset.unwrap_or(0);
    items = items.into_iter().skip(offset).take(limit).collect();
    Ok(Json(TicketListResponse {
        items,
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
}

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
    let intent = resolved_query_intent(&query).ok_or_else(|| {
        ApiError::BadRequest(
            "QueryIntent must provide intent or text for count, trend, compare_regions, top_topics, spikes or forecast".to_owned(),
        )
    })?;
    query.intent = Some(intent.clone());
    let allowed = [
        "count",
        "trend",
        "compare",
        "compare_regions",
        "top_topics",
        "spikes",
        "forecast",
    ];
    if !allowed.contains(&intent.as_str()) {
        return Err(ApiError::BadRequest(format!(
            "unsupported QueryIntent: {intent}"
        )));
    }
    if let Some(repository) = state.repository() {
        return repository
            .analytics_query(&query)
            .await
            .map(Json)
            .map_err(ApiError::Internal);
    }
    let store = state.read_store()?;
    let region_id = query.region_id.clone().or_else(|| {
        query
            .filters
            .as_ref()
            .and_then(|filters| filters.region_id.clone())
    });
    let topic_id = query.topic_id.clone().or_else(|| {
        query
            .filters
            .as_ref()
            .and_then(|filters| filters.topic_id.clone())
    });
    let range = query.range.clone().or_else(|| {
        query
            .filters
            .as_ref()
            .and_then(|filters| filters.range.clone())
    });
    let intent = if intent == "compare" {
        "compare_regions".to_owned()
    } else {
        intent
    };
    let filtered = store.tickets.values().filter(|ticket| {
        region_id
            .as_deref()
            .is_none_or(|value| ticket.region_id == value)
            && topic_id
                .as_deref()
                .is_none_or(|value| ticket.topic_id == value)
    });
    let total = filtered.count();
    let rows = if matches!(intent.as_str(), "compare_regions" | "top_topics") {
        let buckets = if intent == "compare_regions" {
            store
                .regions
                .iter()
                .map(|region| {
                    let count = store
                        .tickets
                        .values()
                        .filter(|ticket| ticket.region_id == region.id)
                        .count();
                    json!({"key": region.id, "label": region.name, "count": count})
                })
                .collect::<Vec<_>>()
        } else {
            store
                .topics
                .iter()
                .map(|topic| {
                    let count = store
                        .tickets
                        .values()
                        .filter(|ticket| ticket.topic_id == topic.id)
                        .count();
                    json!({"key": topic.id, "label": topic.label, "count": count})
                })
                .collect::<Vec<_>>()
        };
        buckets
    } else {
        vec![json!({
            "period": range.clone().unwrap_or_else(|| "7d".to_owned()),
            "count": total,
        })]
    };
    let rows = if let Some(limit) = query.limit {
        rows.into_iter()
            .take(limit.clamp(1, 100))
            .collect::<Vec<_>>()
    } else {
        rows
    };
    Ok(Json(json!({
        "intent": intent,
        "filters": {
            "region_id": region_id,
            "topic_id": topic_id,
            "range": range.unwrap_or_else(|| "7d".to_owned()),
        },
        "number": total,
        "rows": rows.clone(),
        "table": rows.clone(),
        "series": rows,
        "chart": {
            "type": if intent == "trend" { "line" } else { "bar" },
            "x": "label",
            "y": "count",
        },
        "interpreted_filters": {
            "group_by": query.group_by,
            "limit": query.limit,
        },
        "source": "deterministic-demo",
    })))
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
    let alerts = repository
        .list_alerts(&AlertQuery {
            status: None,
            severity: None,
            region_id: query.region_id.clone(),
        })
        .await?;
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

fn report_metric(report: Option<&ReportSlice>, key: &str) -> String {
    report
        .map(|value| &value.analytics)
        .and_then(|value| value.overview.get(key))
        .map(Value::to_string)
        .unwrap_or_else(|| "0".to_owned())
}

fn report_range(report: Option<&ReportSlice>) -> String {
    report
        .map(|value| value.analytics.range.clone())
        .unwrap_or_else(|| "demo".to_owned())
}

/// Canonical HTML template for a report slice.
///
/// The PDF renderer below is intentionally dependency-free for the Core image:
/// it consumes this template and lays out its text in a small, valid PDF.  The
/// The XLSX exporter consumes the same `ReportSlice` directly, so all outputs
/// are reproducible from one backend snapshot.
fn report_html(report: Option<&ReportSlice>) -> String {
    let range = xml_escape(&report_range(report));
    let total = xml_escape(&report_metric(report, "total_tickets"));
    let open = xml_escape(&report_metric(report, "open_tickets"));
    let resolved = xml_escape(&report_metric(report, "resolved_tickets"));
    let mut region_rows = String::new();
    let mut topic_rows = String::new();
    let mut series_rows = String::new();
    let mut alert_rows = String::new();
    let mut forecast_rows = String::new();
    let forecast_status = xml_escape(
        report
            .map(|value| value.forecast.status.as_str())
            .unwrap_or("NO_DATA"),
    );
    if let Some(value) = report {
        for bucket in &value.analytics.by_region {
            region_rows.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td></tr>",
                xml_escape(&bucket.label),
                bucket.tickets,
                bucket
                    .change_pct
                    .map(|change| format!("{change:.1}%"))
                    .unwrap_or_else(|| "—".to_owned()),
            ));
        }
        for bucket in &value.analytics.by_topic {
            topic_rows.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td></tr>",
                xml_escape(&bucket.label),
                bucket.tickets,
                bucket.high_priority,
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
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                xml_escape(&alert.id),
                xml_escape(&alert.status),
                xml_escape(&alert.region_id),
                alert.ticket_count,
            ));
        }
        for point in &value.forecast.points {
            forecast_rows.push_str(&format!(
                "<tr><td>{}</td><td>{}</td></tr>",
                xml_escape(&point.date),
                point.tickets,
            ));
        }
    }
    format!(
        r#"<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Pulse 109 report</title>
<style>body{{font:14px sans-serif;color:#17202a}}table{{border-collapse:collapse;width:100%;margin:8px 0 18px}}th,td{{border:1px solid #ccd3da;padding:5px;text-align:left}}h1,h2{{margin:12px 0 6px}}</style>
</head><body><h1>Pulse 109 report</h1><p>Period: {range}</p>
<h2>Overview</h2><table><tr><th>Metric</th><th>Value</th></tr><tr><td>Total tickets</td><td>{total}</td></tr><tr><td>Open tickets</td><td>{open}</td></tr><tr><td>Resolved tickets</td><td>{resolved}</td></tr></table>
<h2>Regions</h2><table><tr><th>Region</th><th>Tickets</th><th>Change</th></tr>{region_rows}</table>
<h2>Topics</h2><table><tr><th>Topic</th><th>Tickets</th><th>High priority</th></tr>{topic_rows}</table>
<h2>Time series</h2><table><tr><th>Date</th><th>Tickets</th><th>Resolved</th></tr>{series_rows}</table>
<h2>Alerts</h2><table><tr><th>ID</th><th>Status</th><th>Region</th><th>Tickets</th></tr>{alert_rows}</table>
<h2>Forecast</h2><p>Status: {forecast_status}</p><table><tr><th>Date</th><th>Tickets</th></tr>{forecast_rows}</table>
</body></html>"#
    )
}

fn html_to_pdf_lines(html: &str) -> Vec<String> {
    let block_html = html
        .replace("</h1>", "</h1>\n")
        .replace("</h2>", "</h2>\n")
        .replace("</p>", "</p>\n")
        .replace("</tr>", "</tr>\n")
        .replace("</td>", " | </td>");
    let mut plain = String::with_capacity(block_html.len());
    let mut in_tag = false;
    for character in block_html.chars() {
        match character {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                plain.push(' ');
            }
            _ if !in_tag => plain.push(character),
            _ => {}
        }
    }
    plain
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

fn pdf_escape(value: &str) -> String {
    value
        .chars()
        .map(|character| match character {
            '(' => "\\(".to_owned(),
            ')' => "\\)".to_owned(),
            '\\' => "\\\\".to_owned(),
            character if character.is_ascii() && !character.is_control() => character.to_string(),
            _ => "?".to_owned(),
        })
        .collect()
}

fn pdf_report(report: Option<&ReportSlice>) -> Vec<u8> {
    let html = report_html(report);
    let mut content = String::from("BT /F1 10 Tf 48 760 Td ");
    for (index, line) in html_to_pdf_lines(&html).into_iter().take(48).enumerate() {
        if index > 0 {
            content.push_str("0 -14 Td ");
        }
        content.push('(');
        content.push_str(&pdf_escape(&line));
        content.push_str(") Tj ");
    }
    content.push_str("ET\n");
    let bodies = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".to_vec(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        format!("<< /Length {} >>\nstream\n{}endstream", content.len(), content).into_bytes(),
    ];
    let mut pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n".to_vec();
    let mut offsets = Vec::with_capacity(bodies.len());
    for (index, body) in bodies.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n", index + 1).as_bytes());
        pdf.extend_from_slice(body);
        pdf.extend_from_slice(b"\nendobj\n");
    }
    let xref_offset = pdf.len();
    pdf.extend_from_slice(
        format!("xref\n0 {}\n0000000000 65535 f \n", bodies.len() + 1).as_bytes(),
    );
    for offset in offsets {
        pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n",
            bodies.len() + 1
        )
        .as_bytes(),
    );
    pdf
}

fn xlsx_report(report: Option<&ReportSlice>) -> Vec<u8> {
    let content_types = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>"#;
    let root_rels = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#;
    let workbook = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Report" sheetId="1" r:id="rId1"/></sheets></workbook>"#;
    let workbook_rels = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>"#;
    let range = report_range(report);
    let mut rows: Vec<[String; 4]> = vec![
        [
            "period".to_owned(),
            "metric".to_owned(),
            "value".to_owned(),
            "details".to_owned(),
        ],
        [
            range.clone(),
            "total_tickets".to_owned(),
            report_metric(report, "total_tickets"),
            String::new(),
        ],
        [
            range.clone(),
            "open_tickets".to_owned(),
            report_metric(report, "open_tickets"),
            String::new(),
        ],
        [
            range.clone(),
            "resolved_tickets".to_owned(),
            report_metric(report, "resolved_tickets"),
            String::new(),
        ],
    ];
    if let Some(value) = report {
        for point in &value.analytics.time_series {
            rows.push([
                range.clone(),
                "series".to_owned(),
                point.tickets.to_string(),
                format!("{} resolved={}", point.date, point.resolved),
            ]);
        }
        for alert in &value.alerts {
            rows.push([
                range.clone(),
                "alert".to_owned(),
                alert.ticket_count.to_string(),
                format!("{} {} {}", alert.id, alert.status, alert.region_id),
            ]);
        }
        for point in &value.forecast.points {
            rows.push([
                range.clone(),
                "forecast".to_owned(),
                point.tickets.to_string(),
                point.date.clone(),
            ]);
        }
        rows.push([
            range.clone(),
            "forecast_status".to_owned(),
            value.forecast.status.clone(),
            value.forecast.model.clone(),
        ]);
    } else {
        rows.push([
            range.clone(),
            "forecast_status".to_owned(),
            "NO_DATA".to_owned(),
            String::new(),
        ]);
    }
    let cell = |column: &str, row: usize, value: &str| {
        format!(
            r#"<c r="{column}{row}" t="inlineStr"><is><t>{}</t></is></c>"#,
            xml_escape(value)
        )
    };
    let mut sheet_data = String::new();
    for (index, row) in rows.iter().enumerate() {
        let row_number = index + 1;
        sheet_data.push_str(&format!(
            r#"<row r="{row_number}">{}</row>"#,
            [
                cell("A", row_number, &row[0]),
                cell("B", row_number, &row[1]),
                cell("C", row_number, &row[2]),
                cell("D", row_number, &row[3]),
            ]
            .join("")
        ));
    }
    let sheet = format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:D{}"/><sheetData>{sheet_data}</sheetData></worksheet>"#,
        rows.len()
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
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn report_response(format: &str, report: Option<&ReportSlice>) -> Response {
    let (body, content_type, filename) = if format == "pdf" {
        (pdf_report(report), "application/pdf", "pulse109-report.pdf")
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
    response
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
        return Ok(report_response("pdf", Some(&report)));
    }
    Ok(report_response("pdf", None))
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
        return Ok(report_response("xlsx", Some(&report)));
    }
    Ok(report_response("xlsx", None))
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
    Ok(Json(json!({
        "items": [
            {"id": "report-demo-7d", "format": "pdf", "status": "ready", "download": "/api/v1/analytics/export.pdf"},
            {"id": "report-demo-7d", "format": "xlsx", "status": "ready", "download": "/api/v1/analytics/export.xlsx"}
        ],
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

fn metric_bucket(id: String, label: String, tickets: Vec<&Ticket>, store: &Store) -> MetricBucket {
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
            .filter(|ticket| ticket.priority == "high")
            .count(),
        avg_confidence,
        change_abs: None,
        change_pct: None,
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
    let _store = state.read_store()?;
    let pattern = [11_u32, 12, 10, 13, 14, 12, 15];
    let resolved_pattern = [7_u32, 7, 6, 8, 8, 7, 9];
    let start = NaiveDate::from_ymd_opt(2026, 9, 22).expect("fixed demo date is valid");
    let points = (0..horizon_days)
        .map(|index| TimeSeriesPoint {
            date: (start + Duration::days(i64::from(index))).to_string(),
            tickets: pattern[index as usize % pattern.len()],
            resolved: resolved_pattern[index as usize % resolved_pattern.len()],
        })
        .collect::<Vec<_>>();
    let peak_value = pattern.iter().copied().max().unwrap_or_default();
    let expected_peaks = points
        .iter()
        .filter(|point| point.tickets == peak_value)
        .map(|point| point.date.clone())
        .collect();
    Ok(Json(ForecastResponse {
        source: "deterministic-demo".to_owned(),
        model_version: "forecast-statsforecast-seasonal-naive-2026-09-24-001".to_owned(),
        model: "seasonal-naive-demo".to_owned(),
        status: "OK".to_owned(),
        insufficient_history: false,
        horizon_days,
        history: Vec::new(),
        points,
        expected_peaks,
        backtest: json!({"status": "DEMO_ONLY"}),
    }))
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
) -> Result<Json<AlertListResponse>, ApiError> {
    require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    if let Some(repository) = state.repository() {
        let items = repository
            .detect_alerts()
            .await
            .map_err(ApiError::Internal)?;
        let total = items.len();
        if total > 0 {
            state.publish_alerts_changed();
        }
        return Ok(Json(AlertListResponse { items, total }));
    }
    let store = state.read_store()?;
    let items = store.alerts.values().cloned().collect::<Vec<_>>();
    let total = items.len();
    Ok(Json(AlertListResponse { items, total }))
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
        .find(|cycle| !matches!(cycle.state.as_str(), "PROMOTED" | "REJECTED"))
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
            "stages": ["COLLECT", "TRAIN", "EVALUATE", "SHADOW", "PROMOTE", "REJECT"],
            "production_auto_update": false,
            "operator_correction_triggers_training": false,
        }),
    }))
}

#[derive(Debug, Deserialize, Default)]
pub struct CreateLearningCycleRequest {
    pub dataset_version: Option<String>,
    pub candidate_model_version: Option<String>,
}

async fn create_learning_cycle(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<CreateLearningCycleRequest>,
) -> Result<(StatusCode, Json<LearningCycle>), ApiError> {
    require_role(&headers, &state.config, &[Role::MlReviewer, Role::Admin])?;
    if let Some(repository) = state.repository() {
        let cycle = repository
            .create_learning_cycle(&request)
            .await
            .map_err(ApiError::Internal)?;
        return Ok((StatusCode::CREATED, Json(cycle)));
    }
    let mut store = state.write_store()?;
    let cycle = LearningCycle {
        id: format!("cycle-{:03}", store.next_cycle_number),
        state: "COLLECT".to_owned(),
        dataset_version: request
            .dataset_version
            .unwrap_or_else(|| format!("dataset-demo-2026-09-{:03}", store.next_cycle_number)),
        candidate_model_version: request.candidate_model_version.unwrap_or_else(|| {
            format!(
                "classifier-candidate-2026-09-{:03}",
                store.next_cycle_number
            )
        }),
        metrics: LearningMetrics {
            macro_f1: 0.0,
            accuracy: 0.0,
            evaluated_samples: 0,
        },
        feedback_count: 0,
        decision_note: None,
        created_at: DEMO_TIMESTAMP.to_owned(),
        updated_at: DEMO_TIMESTAMP.to_owned(),
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
                } else {
                    ApiError::Internal(error)
                }
            })?;
        return Ok((StatusCode::CREATED, Json(feedback)));
    }
    let mut store = state.write_store()?;
    if !store.learning_cycles.contains_key(&cycle_id) {
        return Err(ApiError::NotFound(format!(
            "learning cycle {cycle_id} not found"
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
        created_at: DEMO_TIMESTAMP.to_owned(),
    };
    store.next_feedback_number += 1;
    store.learning_feedback.push(feedback.clone());
    if let Some(cycle) = store.learning_cycles.get_mut(&cycle_id) {
        cycle.feedback_count += 1;
        cycle.updated_at = DEMO_TIMESTAMP.to_owned();
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
            .close_learning_cycle(&request)
            .await
            .map_err(ApiError::Internal)?;
        let _ = repository
            .audit(
                &actor.user_id,
                "CLOSE_LEARNING_CYCLE",
                "learning_cycle",
                request.cycle_id.as_deref(),
                None,
                None,
                json!({}),
            )
            .await;
        return Ok((StatusCode::ACCEPTED, Json(result)));
    }
    let mut store = state.write_store()?;
    let cycle_id = request.cycle_id.or_else(|| {
        store
            .learning_cycles
            .values()
            .find(|cycle| cycle.state == "COLLECT")
            .map(|cycle| cycle.id.clone())
    });
    let cycle_id =
        cycle_id.ok_or_else(|| ApiError::Conflict("no COLLECT cycle is available".to_owned()))?;
    let cycle = store
        .learning_cycles
        .get_mut(&cycle_id)
        .ok_or_else(|| ApiError::NotFound(format!("learning cycle {cycle_id} not found")))?;
    if cycle.state != "COLLECT" {
        return Err(ApiError::Conflict(format!(
            "cycle {} cannot close from state {}",
            cycle.id, cycle.state
        )));
    }
    cycle.state = "EVALUATE".to_owned();
    cycle.updated_at = DEMO_TIMESTAMP.to_owned();
    Ok((
        StatusCode::ACCEPTED,
        Json(json!({
            "job_id": format!("train-{}", cycle.id),
            "state": cycle.state,
            "cycle": cycle,
            "production_model_unchanged": true,
        })),
    ))
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
    let cycle = store
        .learning_cycles
        .values()
        .find(|cycle| !matches!(cycle.state.as_str(), "PROMOTED" | "REJECTED"))
        .cloned()
        .ok_or_else(|| ApiError::NotFound("no active learning cycle".to_owned()))?;
    Ok(Json(json!({
        "cycle_id": cycle.id,
        "state": cycle.state,
        "offline_metrics": cycle.metrics,
        "shadow_metrics": {"agreement": 0.88, "correction_rate_delta": -0.04, "sample_size": cycle.feedback_count},
        "critical_regressions": [],
        "promotion_policy_version": "policy-demo-v1",
        "decision": "READY_TO_PROMOTE"
    })))
}

fn active_cycle_id(store: &Store) -> Option<String> {
    store
        .learning_cycles
        .values()
        .find(|cycle| !matches!(cycle.state.as_str(), "PROMOTED" | "REJECTED"))
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
    let mut store = state.write_store()?;
    let (candidate_id, metrics) = {
        let cycle = store
            .learning_cycles
            .get(&cycle_id)
            .ok_or_else(|| ApiError::NotFound(format!("learning cycle {cycle_id} not found")))?;
        if cycle.state != "EVALUATE" && cycle.state != "SHADOW" {
            return Err(ApiError::Conflict(format!(
                "cycle {} cannot be promoted from state {}",
                cycle.id, cycle.state
            )));
        }
        (cycle.candidate_model_version.clone(), cycle.metrics.clone())
    };
    if let Some(model) = store.models.get_mut(&candidate_id) {
        model.status = "production".to_owned();
        model.promoted_at = Some(DEMO_TIMESTAMP.to_owned());
    } else {
        let labels = store.topics.iter().map(|topic| topic.id.clone()).collect();
        store.models.insert(
            candidate_id.clone(),
            ModelVersion {
                id: candidate_id.clone(),
                model_family: "candidate".to_owned(),
                base_model: "deterministic-demo".to_owned(),
                dataset_version: "unknown".to_owned(),
                status: "production".to_owned(),
                metrics,
                languages: vec!["ru".to_owned(), "kk".to_owned()],
                labels,
                created_at: DEMO_TIMESTAMP.to_owned(),
                promoted_at: Some(DEMO_TIMESTAMP.to_owned()),
            },
        );
    }
    for model in store.models.values_mut() {
        if model.id != candidate_id && model.status == "production" {
            model.status = "archived".to_owned();
        }
    }
    let cycle = store
        .learning_cycles
        .get_mut(&cycle_id)
        .ok_or_else(|| ApiError::NotFound(format!("learning cycle {cycle_id} not found")))?;
    cycle.state = "PROMOTED".to_owned();
    cycle.decision_note = request
        .note
        .or_else(|| Some(format!("Promoted by {}", actor.user_id)));
    cycle.updated_at = DEMO_TIMESTAMP.to_owned();
    Ok(Json(cycle.clone()))
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
    let cycle = store
        .learning_cycles
        .get_mut(&cycle_id)
        .ok_or_else(|| ApiError::NotFound(format!("learning cycle {cycle_id} not found")))?;
    if cycle.state == "PROMOTED" {
        return Err(ApiError::Conflict(
            "a promoted cycle cannot be rejected".to_owned(),
        ));
    }
    cycle.state = "REJECTED".to_owned();
    cycle.decision_note = request
        .note
        .or_else(|| Some(format!("Rejected by {}", actor.user_id)));
    cycle.updated_at = DEMO_TIMESTAMP.to_owned();
    Ok(Json(cycle.clone()))
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
        user_id = %actor.user_id,
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
            "/readyz": { "get": { "summary": "Readiness" } },
            "/api/v1/openapi.json": { "get": { "summary": "Core OpenAPI route index" } },
            "/api/v1/docs": { "get": { "summary": "Core OpenAPI route index" } },
            "/api/v1/tickets": { "get": { "summary": "List tickets" }, "post": { "summary": "Create ticket" } },
            "/api/v1/tickets/{ticket_id}": { "get": { "summary": "Get ticket and prediction" } },
            "/api/v1/tickets/{ticket_id}/prediction": { "get": { "summary": "Get current ticket prediction" } },
            "/api/v1/tickets/{ticket_id}/vector": { "delete": { "summary": "Delete one Qdrant vector" } },
            "/api/v1/tickets/{ticket_id}/pulse-state": { "put": { "summary": "Store human-confirmed Pulse state" } },
            "/api/v1/import": { "post": { "summary": "Import validated tickets into PostgreSQL and Qdrant" } },
            "/api/v1/datasets/provenance": { "get": { "summary": "Read synthetic, real and unassigned dataset counts" } },
            "/api/v1/assist/preview": { "post": { "summary": "Preview prediction and related tickets" } },
            "/api/v1/assist/{ticket_id}/confirm": { "post": { "summary": "Confirm prediction" } },
            "/api/v1/assist/{ticket_id}/correct": { "post": { "summary": "Correct prediction" } },
            "/api/v1/assist/confirm": { "post": { "summary": "Confirm prediction using ticket_id in body" } },
            "/api/v1/assist/correct": { "post": { "summary": "Correct prediction using ticket_id in body" } },
            "/api/v1/assist/confirm/{ticket_id}": { "post": { "summary": "Confirm prediction alias" } },
            "/api/v1/assist/correct/{ticket_id}": { "post": { "summary": "Correct prediction alias" } },
            "/api/v1/analytics": { "get": { "summary": "Situation center analytics" } },
            "/api/v1/analytics/overview": { "get": { "summary": "Situation center overview" } },
            "/api/v1/analytics/drilldown": { "get": { "summary": "Drill analytics metrics down to source tickets" } },
            "/api/v1/taxonomy": { "get": { "summary": "Current regions, topics, services and available ticket filters" } },
            "/api/v1/analytics/query": { "post": { "summary": "Validated QueryIntent analytics" } },
            "/api/v1/retrieval/reindex": { "post": { "summary": "Queue a PostgreSQL-backed Qdrant reindex" } },
            "/api/v1/analytics/export.pdf": { "get": { "summary": "Export ReportSlice as PDF" } },
            "/api/v1/analytics/export.xlsx": { "get": { "summary": "Export ReportSlice as XLSX" } },
            "/api/v1/reports": { "get": { "summary": "List generated reports" } },
            "/api/v1/events": { "get": { "summary": "SSE notifications" } },
            "/api/v1/forecast": { "get": { "summary": "Forecast baseline" } },
            "/api/v1/alerts": { "get": { "summary": "List alerts" } },
            "/api/v1/alerts/detect": { "post": { "summary": "Detect and persist spike alerts" } },
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
            "/api/v1/learning/candidate/evaluation": { "get": { "summary": "Evaluate candidate" } },
            "/api/v1/learning/candidate/promote": { "post": { "summary": "Promote candidate" } },
            "/api/v1/learning/candidate/reject": { "post": { "summary": "Reject candidate" } },
            "/api/v1/tickets/{ticket_id}/relation-feedback": { "post": { "summary": "Collect relation feedback" } },
            "/api/v1/models": { "get": { "summary": "List model versions" } },
            "/api/v1/models/{model_id}": { "get": { "summary": "Get model version" } },
            "/api/v1/models/{model_id}/promote": { "post": { "summary": "Promote model version after human review" } }
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

    async fn body_json(response: Response) -> Value {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn health_and_readiness_are_available_without_auth() {
        let app = app(AppState::demo());
        let response = app
            .clone()
            .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_json(response).await["status"], "ok");

        let response = app
            .oneshot(Request::get("/readyz").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(body_json(response).await["status"], "ready");
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
        assert!(value["paths"]["/api/v1/assist/preview"].is_object());
        assert!(value["paths"]["/api/v1/learning/{cycle_id}/feedback"].is_object());
        assert!(value["paths"]["/api/v1/models/{model_id}/promote"].is_object());
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
        assert_eq!(body_json(cycle).await["active_cycle"]["feedback_count"], 4);

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
        assert!(to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .starts_with(b"%PDF-1.4"));

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

        let response = app
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
    }

    #[tokio::test]
    async fn forecast_supports_required_horizons() {
        let app = app(AppState::demo());
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
