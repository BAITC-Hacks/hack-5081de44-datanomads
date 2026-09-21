//! Pulse 109 Core API.
//!
//! The default storage is an in-memory deterministic demo store.  The store is
//! deliberately kept behind a small interface (`AppState`) so the HTTP
//! contract can be exercised locally before wiring the same handlers to the
//! PostgreSQL repository used in production.

use axum::{
    extract::{Path, Query, Request, State},
    http::{header, HeaderMap, HeaderValue, Method, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    env,
    sync::{Arc, RwLock},
    time::Instant,
};
use thiserror::Error;
use tower_http::cors::CorsLayer;
use tracing::info;

const SERVICE_NAME: &str = "pulse109-core";
const API_VERSION: &str = "0.1.0";
const DEMO_TIMESTAMP: &str = "2026-09-21T08:00:00Z";

/// Runtime configuration.  `dev_auth` is enabled by default for the local
/// deterministic demo: a request without `x-pulse-role` acts as ADMIN, while
/// an explicitly supplied role is always checked.
#[derive(Clone, Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub dev_auth: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_owned(),
            port: 8080,
            dev_auth: true,
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        let defaults = Self::default();
        Self {
            host: env::var("PULSE_HOST").unwrap_or(defaults.host),
            port: env::var("PULSE_PORT")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(defaults.port),
            dev_auth: env::var("PULSE_DEV_AUTH")
                .map(|value| !matches!(value.to_ascii_lowercase().as_str(), "0" | "false" | "no"))
                .unwrap_or(defaults.dev_auth),
        }
    }
}

/// Shared application state.
#[derive(Clone)]
pub struct AppState {
    store: Arc<RwLock<Store>>,
    pub config: Config,
}

impl AppState {
    pub fn demo() -> Self {
        Self {
            store: Arc::new(RwLock::new(Store::demo())),
            config: Config::default(),
        }
    }

    pub fn from_env() -> Self {
        Self {
            store: Arc::new(RwLock::new(Store::demo())),
            config: Config::from_env(),
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
    pub updated_at: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct AlternativePrediction {
    pub topic_id: String,
    pub topic_label: String,
    pub confidence: f32,
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
    pub alternatives: Vec<AlternativePrediction>,
    pub created_at: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct OperatorDecision {
    pub id: String,
    pub ticket_id: String,
    pub action: String,
    pub predicted_topic_id: String,
    pub confirmed_topic_id: String,
    pub service: String,
    pub priority: String,
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
}

#[derive(Clone, Debug, Serialize)]
pub struct ResponseTemplate {
    pub id: String,
    pub title: String,
    pub body: String,
    pub language: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Alert {
    pub id: String,
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
    let role_header = headers
        .get("x-pulse-role")
        .and_then(|value| value.to_str().ok());
    let user_id = headers
        .get("x-user-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("demo-admin")
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
            "x-pulse-role is required outside development mode".to_owned(),
        )),
    }
}

fn require_role(headers: &HeaderMap, config: &Config, allowed: &[Role]) -> Result<Actor, ApiError> {
    let role_was_supplied = headers.contains_key("x-pulse-role");
    let current = actor(headers, config)?;
    if !role_was_supplied && config.dev_auth {
        // A local browser demo has no auth gateway. Explicit role headers are
        // still checked below, while headerless local requests remain usable.
        Ok(current)
    } else if allowed.contains(&current.role) {
        Ok(current)
    } else {
        Err(ApiError::Forbidden(format!(
            "role {} is not allowed for this operation",
            current.role.as_str()
        )))
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

fn topic_for_text(text: &str) -> (&'static str, f32, &'static str, &'static str) {
    let value = text.to_ascii_lowercase();
    if value.contains("вод") || value.contains("су ") || value.contains("суару") {
        ("TOPIC-WATER", 0.96, "Водоканал", "high")
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
    } else if value.contains("қоқыс") || value.contains("мусор") || value.contains("эколог")
    {
        ("TOPIC-ENVIRONMENT", 0.84, "Управление экологии", "low")
    } else {
        ("TOPIC-OTHER", 0.61, "Единый контакт-центр", "low")
    }
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
    let (topic_id, confidence, service, priority_state) = topic_for_text(&ticket.text);
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
                updated_at: DEMO_TIMESTAMP.to_owned(),
            };
            let prediction = prediction_for_ticket(&ticket, &topics);
            predictions.insert(ticket.id.clone(), prediction);
            tickets.insert(ticket.id.clone(), ticket);
        }

        let alerts = [
            Alert {
                id: "alert-001".to_owned(),
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
            },
            Alert {
                id: "alert-002".to_owned(),
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
                confirmed_topic_id: "TOPIC-ROADS".to_owned(),
                service: "Городская инфраструктура".to_owned(),
                priority: "high".to_owned(),
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
        .route("/api/v1/tickets/{ticket_id}", get(get_ticket))
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
        .route("/api/v1/forecast", get(forecast))
        .route("/api/v1/alerts", get(list_alerts))
        .route("/api/v1/alerts/{alert_id}", get(get_alert))
        .route("/api/v1/alerts/{alert_id}/ack", post(ack_alert))
        .route("/api/v1/alerts/{alert_id}/acknowledge", post(ack_alert))
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
        .route("/api/v1/models", get(list_models))
        .route("/api/v1/models/{model_id}", get(get_model))
        .route("/api/v1/models/{model_id}/promote", post(promote_model))
        .layer(middleware::from_fn(request_context))
        .layer(
            CorsLayer::new()
                .allow_origin(tower_http::cors::Any)
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
                .allow_headers(tower_http::cors::Any)
                .expose_headers([header::HeaderName::from_static("x-request-id")]),
        )
        .with_state(state)
}

async fn request_context(mut request: Request, next: Next) -> Response {
    let request_id = request
        .headers()
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
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
        .filter(|value| !value.is_empty())
        .unwrap_or(&request_id)
        .to_owned();
    let started = Instant::now();
    let method = request.method().clone();
    let path = request.uri().path().to_owned();
    request.extensions_mut().insert(request_id.clone());
    let mut response = next.run(request).await;
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    if let Ok(value) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", value);
    }
    info!(
        service = SERVICE_NAME,
        request_id = %request_id,
        trace_id = %trace_id,
        endpoint = %path,
        method = %method,
        latency_ms,
        status = response.status().as_u16(),
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
    Query(query): Query<TicketQuery>,
) -> Result<Json<TicketListResponse>, ApiError> {
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
    let mut store = state.write_store()?;
    let region_id = request.region_id.unwrap_or_else(|| "R01".to_owned());
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
    let (topic_id, _, _, _) = topic_for_text(text);
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

async fn get_ticket(
    State(state): State<AppState>,
    Path(ticket_id): Path<String>,
) -> Result<Json<TicketDetailResponse>, ApiError> {
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
    Path(ticket_id): Path<String>,
) -> Result<Json<Prediction>, ApiError> {
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
}

fn related_tickets(store: &Store, ticket: &Ticket, prediction: &Prediction) -> Vec<SimilarTicket> {
    store
        .tickets
        .values()
        .filter(|candidate| candidate.id != ticket.id)
        .filter(|candidate| candidate.topic_id == prediction.topic_id)
        .take(3)
        .enumerate()
        .map(|(index, candidate)| SimilarTicket {
            ticket_id: candidate.id.clone(),
            score: (0.91 - index as f32 * 0.11).max(0.5),
            relation: if candidate.region_id == ticket.region_id {
                "similar"
            } else {
                "repeat"
            }
            .to_owned(),
            topic_id: candidate.topic_id.clone(),
            region_id: candidate.region_id.clone(),
        })
        .collect()
}

async fn assist_preview(
    State(state): State<AppState>,
    Json(request): Json<PreviewRequest>,
) -> Result<Json<AssistPreviewResponse>, ApiError> {
    if request.ticket_id.is_none() && request.text.as_deref().is_none_or(str::is_empty) {
        return Err(ApiError::BadRequest(
            "ticket_id or non-empty text is required".to_owned(),
        ));
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
        let (topic_id, _, _, _) = topic_for_text(&text);
        let region_id = request.region_id.unwrap_or_else(|| "R01".to_owned());
        let region = store
            .regions
            .iter()
            .find(|region| region.id == region_id)
            .cloned()
            .ok_or_else(|| ApiError::BadRequest(format!("unknown region_id: {region_id}")))?;
        let language = request
            .language
            .unwrap_or_else(|| detect_language(&text).to_owned());
        if language != "ru" && language != "kk" {
            return Err(ApiError::BadRequest("language must be ru or kk".to_owned()));
        }
        (
            Ticket {
                id: "preview-001".to_owned(),
                external_ref: "PREVIEW-0001".to_owned(),
                text,
                language,
                region_id: region.id,
                region_name: region.name,
                topic_id: topic_id.to_owned(),
                topic_label: topic_label(&store.topics, topic_id),
                priority: priority_for_topic(topic_id).to_owned(),
                status: "preview".to_owned(),
                source: "preview".to_owned(),
                created_at: DEMO_TIMESTAMP.to_owned(),
                updated_at: DEMO_TIMESTAMP.to_owned(),
            },
            "deterministic-demo".to_owned(),
        )
    };
    let prediction = store
        .predictions
        .get(&ticket.id)
        .cloned()
        .unwrap_or_else(|| prediction_for_ticket(&ticket, &store.topics));
    let related = related_tickets(&store, &ticket, &prediction);
    let duplicate_candidates = related
        .iter()
        .filter(|item| item.score >= 0.8)
        .cloned()
        .collect();
    let repeat_candidates = related
        .iter()
        .filter(|item| item.relation == "repeat")
        .cloned()
        .collect();
    Ok(Json(AssistPreviewResponse {
        response_template: response_template(&ticket.language, &prediction.topic_id),
        ticket,
        prediction,
        similar_tickets: related,
        duplicate_candidates,
        repeat_candidates,
        source,
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
    {
        let ticket = store
            .tickets
            .get_mut(&ticket_id)
            .ok_or_else(|| ApiError::NotFound(format!("ticket {ticket_id} not found")))?;
        ticket.status = "triaged".to_owned();
        ticket.updated_at = DEMO_TIMESTAMP.to_owned();
    }
    let service = request
        .service
        .unwrap_or_else(|| service_for_topic(&topic_id).to_owned());
    let priority = request
        .priority
        .unwrap_or_else(|| prediction.predicted_priority.clone());
    let decision_id = format!("decision-{:03}", store.next_decision_number);
    store.next_decision_number += 1;
    let decision = OperatorDecision {
        id: decision_id,
        ticket_id: ticket_id.clone(),
        action: action.to_owned(),
        predicted_topic_id: prediction.topic_id.clone(),
        confirmed_topic_id: topic_id,
        service,
        priority,
        note: request.note,
        user_id: actor.user_id,
        created_at: DEMO_TIMESTAMP.to_owned(),
    };
    store.decisions.push(decision.clone());
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
        result_state = "recorded",
        "operator_decision_recorded"
    );
    Ok(Json(DecisionResponse {
        ticket,
        prediction,
        decision,
    }))
}

#[derive(Debug, Deserialize)]
pub struct AnalyticsQuery {
    pub region_id: Option<String>,
    pub topic_id: Option<String>,
    pub range: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct MetricBucket {
    pub id: String,
    pub label: String,
    pub tickets: usize,
    pub high_priority: usize,
    pub avg_confidence: f32,
}

#[derive(Debug, Serialize)]
pub struct TimeSeriesPoint {
    pub date: String,
    pub tickets: u32,
    pub resolved: u32,
}

#[derive(Debug, Serialize)]
pub struct AnalyticsResponse {
    pub generated_at: String,
    pub source: String,
    pub range: String,
    pub overview: Value,
    pub by_region: Vec<MetricBucket>,
    pub by_topic: Vec<MetricBucket>,
    pub time_series: Vec<TimeSeriesPoint>,
}

async fn analytics(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<AnalyticsQuery>,
) -> Result<Json<AnalyticsResponse>, ApiError> {
    require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
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
    let overview = json!({
        "total_tickets": filtered.len(),
        "open_tickets": open_tickets,
        "resolved_tickets": resolved,
        "high_priority_tickets": high_priority,
        "operator_decisions": store.decisions.iter().filter(|decision| filtered.iter().any(|ticket| ticket.id == decision.ticket_id)).count(),
        "confirmed_decisions": store.decisions.iter().filter(|decision| decision.action == "confirm" && filtered.iter().any(|ticket| ticket.id == decision.ticket_id)).count(),
        "corrected_decisions": store.decisions.iter().filter(|decision| decision.action == "correct" && filtered.iter().any(|ticket| ticket.id == decision.ticket_id)).count(),
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
    }
}

#[derive(Debug, Serialize)]
pub struct ForecastResponse {
    pub source: String,
    pub model: String,
    pub horizon_days: u32,
    pub points: Vec<TimeSeriesPoint>,
}

async fn forecast(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<ForecastResponse>, ApiError> {
    require_role(&headers, &state.config, &[Role::Manager, Role::Admin])?;
    let _store = state.read_store()?;
    Ok(Json(ForecastResponse {
        source: "deterministic-demo".to_owned(),
        model: "seasonal-naive-demo".to_owned(),
        horizon_days: 7,
        points: vec![
            TimeSeriesPoint {
                date: "2026-09-22".to_owned(),
                tickets: 11,
                resolved: 7,
            },
            TimeSeriesPoint {
                date: "2026-09-23".to_owned(),
                tickets: 12,
                resolved: 7,
            },
            TimeSeriesPoint {
                date: "2026-09-24".to_owned(),
                tickets: 10,
                resolved: 6,
            },
            TimeSeriesPoint {
                date: "2026-09-25".to_owned(),
                tickets: 13,
                resolved: 8,
            },
            TimeSeriesPoint {
                date: "2026-09-26".to_owned(),
                tickets: 14,
                resolved: 8,
            },
            TimeSeriesPoint {
                date: "2026-09-27".to_owned(),
                tickets: 12,
                resolved: 7,
            },
            TimeSeriesPoint {
                date: "2026-09-28".to_owned(),
                tickets: 15,
                resolved: 9,
            },
        ],
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
    let mut store = state.write_store()?;
    let alert = store
        .alerts
        .get_mut(&alert_id)
        .ok_or_else(|| ApiError::NotFound(format!("alert {alert_id} not found")))?;
    alert.status = "acknowledged".to_owned();
    alert.acknowledged_by = Some(actor.user_id);
    alert.acknowledged_at = Some(DEMO_TIMESTAMP.to_owned());
    Ok(Json(alert.clone()))
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
    pub feedback_type: String,
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
        &[Role::Operator, Role::Manager, Role::Admin],
    )?;
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
    let feedback = LearningFeedback {
        id: format!("feedback-{:03}", store.next_feedback_number),
        ticket_id: request.ticket_id,
        cycle_id: Some(cycle_id.clone()),
        feedback_type: request.feedback_type,
        comment: request.comment,
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

#[derive(Debug, Deserialize, Default)]
pub struct CycleDecisionRequest {
    pub note: Option<String>,
}

async fn promote_learning_cycle(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(cycle_id): Path<String>,
    Json(request): Json<CycleDecisionRequest>,
) -> Result<Json<LearningCycle>, ApiError> {
    let actor = require_role(&headers, &state.config, &[Role::MlReviewer, Role::Admin])?;
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
            "description": "P0 operator workflow and situation center API. Demo responses are deterministic and marked with source=deterministic-demo."
        },
        "servers": [{ "url": "/" }],
        "security": [{ "PulseRole": [] }],
        "components": {
            "securitySchemes": {
                "PulseRole": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "x-pulse-role",
                    "description": "Development role stub: OPERATOR, MANAGER, ML_REVIEWER or ADMIN."
                }
            }
        },
        "paths": {
            "/healthz": { "get": { "summary": "Liveness" } },
            "/readyz": { "get": { "summary": "Readiness" } },
            "/api/v1/tickets": { "get": { "summary": "List tickets" }, "post": { "summary": "Create ticket" } },
            "/api/v1/tickets/{ticket_id}": { "get": { "summary": "Get ticket and prediction" } },
            "/api/v1/assist/preview": { "post": { "summary": "Preview prediction and related tickets" } },
            "/api/v1/assist/{ticket_id}/confirm": { "post": { "summary": "Confirm prediction" } },
            "/api/v1/assist/{ticket_id}/correct": { "post": { "summary": "Correct prediction" } },
            "/api/v1/analytics": { "get": { "summary": "Situation center analytics" } },
            "/api/v1/forecast": { "get": { "summary": "Forecast baseline" } },
            "/api/v1/alerts": { "get": { "summary": "List alerts" } },
            "/api/v1/learning": { "get": { "summary": "Learning loop status" }, "post": { "summary": "Start candidate cycle" } },
            "/api/v1/models": { "get": { "summary": "List model versions" } }
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
    }
}
