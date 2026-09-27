//! PostgreSQL/Qdrant/ML integration for the real Core runtime.
//!
//! The demo store in `lib.rs` is intentionally kept for unit tests and for
//! an explicitly selected `PULSE_STORAGE=memory` run.  Compose uses this
//! repository instead.  PostgreSQL is the source of truth; Qdrant only keeps
//! the vector index and the ML service owns classification/embedding.

use crate::{
    Alert, AlertQuery, AlternativePrediction, AnalyticsDrilldownQuery, AnalyticsQuery,
    AnalyticsResponse, AssistPreviewResponse, CloseLearningCycleRequest,
    CreateLearningCycleRequest, DecisionRequest, DecisionResponse, ForecastQuery, ForecastResponse,
    ImportRequest, ImportResponse, LearningCycle, LearningFeedback, LearningFeedbackRequest,
    LearningMetrics, LearningOverview, MetricBucket, ModelQuery, ModelVersion, OperatorDecision,
    Prediction, QueryIntentRequest, ResponseTemplate, SimilarTicket, Ticket, TicketDetailResponse,
    TicketListResponse, TicketQuery, TimeSeriesPoint, Topic,
};
use chrono::{DateTime, NaiveDate, Utc};
use reqwest::{Client, StatusCode as HttpStatus};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sqlx::{postgres::PgPoolOptions, FromRow, PgPool, Postgres, QueryBuilder, Row};
use std::env;

const DEFAULT_EMBEDDING_DIMENSION: usize = 32;
const DEFAULT_QDRANT_COLLECTION: &str = "pulse109_tickets_v1";
const DEFAULT_EMBEDDER_VERSION: &str = "embedder-demo-2026-09-21-001";
const DEFAULT_FORECAST_MODEL_VERSION: &str = "forecast-statsforecast-seasonal-naive-2026-09-24-001";

async fn enqueue_classifier_shadow_job(
    tx: &mut sqlx::Transaction<'_, Postgres>,
    ticket_id: i64,
    production_prediction_id: i64,
    production_model_version: &str,
) -> Result<(), String> {
    sqlx::query(
        "INSERT INTO background_jobs (job_type, payload, state) SELECT 'SHADOW_CLASSIFIER', jsonb_build_object('cycle_id', lc.id::text, 'ticket_id', $1::text, 'production_prediction_id', $2::text, 'production_model_version', $3, 'candidate_model_version', lc.candidate_model_version, 'candidate_artifact_checksum', mv.artifact_checksum), 'QUEUED' FROM learning_cycles lc JOIN tickets t ON t.id = $1 JOIN model_versions mv ON mv.model_version = lc.candidate_model_version JOIN model_evaluations me ON me.model_version = mv.model_version WHERE lc.state = 'EVALUATE' AND lc.production_model_version = $3 AND lc.evaluation_started_at IS NOT NULL AND t.created_at >= lc.evaluation_started_at AND (lc.evaluation_ends_at IS NULL OR t.created_at < lc.evaluation_ends_at) AND mv.status IN ('CANDIDATE', 'SHADOW') AND mv.manifest_uri IS NOT NULL AND mv.artifact_checksum LIKE 'sha256:%' AND me.metrics_json->>'report_version' = 'classifier-pair-evaluation.v1' ORDER BY lc.updated_at DESC, lc.id DESC LIMIT 1",
    )
    .bind(ticket_id)
    .bind(production_prediction_id)
    .bind(production_model_version)
    .execute(&mut **tx)
    .await
    .map_err(|error| format!("queue classifier shadow: {error}"))?;
    Ok(())
}

#[derive(Debug)]
pub enum ImportError {
    Invalid(String),
    Conflict(String),
    Internal(String),
}

impl From<String> for ImportError {
    fn from(message: String) -> Self {
        Self::Internal(message)
    }
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

#[derive(Clone)]
pub struct PgRepository {
    pub pool: PgPool,
    pub qdrant_url: String,
    pub ml_service_url: String,
    pub qdrant_collection: String,
    pub embedding_dimension: usize,
    pub embedder_version: String,
    pub forecast_model_version: String,
    client: Client,
}

#[derive(Debug, Deserialize)]
struct MlClassifyResponse {
    model_version: String,
    predictions: Vec<MlClassification>,
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
    updated_at: DateTime<Utc>,
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
    confirmed_priority: String,
    service: Option<String>,
    user_id: String,
    note: Option<String>,
    created_at: DateTime<Utc>,
}

#[derive(Debug)]
struct QdrantHit {
    id: i64,
    score: f32,
    topic_id: String,
    region_id: String,
    created_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
struct RoutingDecision {
    service_id: String,
    service_name: String,
    priority: String,
    reason: String,
}

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
            embedder_version: embedder_version.into(),
            forecast_model_version: env::var("FORECAST_MODEL_VERSION")
                .unwrap_or_else(|_| DEFAULT_FORECAST_MODEL_VERSION.to_owned()),
            client: Client::builder()
                .build()
                .map_err(|error| format!("build HTTP client: {error}"))?,
        })
    }

    pub async fn initialize(&self) -> Result<(), String> {
        sqlx::migrate!("../migrations")
            .run(&self.pool)
            .await
            .map_err(|error| format!("apply migrations: {error}"))?;
        self.ensure_qdrant_collection().await
    }

    pub async fn readiness(&self) -> Result<Value, String> {
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await
            .map_err(|error| format!("postgres: {error}"))?;
        let qdrant = self
            .client
            .get(format!("{}/readyz", self.qdrant_url))
            .send()
            .await
            .map_err(|error| format!("qdrant: {error}"))?
            .error_for_status()
            .map_err(|error| format!("qdrant: {error}"))?;
        let ml = self
            .client
            .get(format!("{}/readyz", self.ml_service_url))
            .send()
            .await
            .map_err(|error| format!("ml service: {error}"))?
            .error_for_status()
            .map_err(|error| format!("ml service: {error}"))?;
        let ml_health: Value = ml
            .json()
            .await
            .map_err(|error| format!("ml service readiness response: {error}"))?;
        let production_version: String = sqlx::query_scalar("SELECT model_version FROM model_versions WHERE status = 'PRODUCTION' ORDER BY promoted_at DESC NULLS LAST, id DESC LIMIT 1")
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("production model registry: {error}"))?
            .ok_or_else(|| "production classifier pointer is missing".to_owned())?;
        let served_version = ml_health["model_versions"]["classifier"]
            .as_str()
            .ok_or_else(|| "ML classifier version is missing from readiness response".to_owned())?;
        if production_version != served_version {
            return Err(
                "ML classifier version differs from PostgreSQL production pointer".to_owned(),
            );
        }
        Ok(json!({
            "postgres": true,
            "qdrant": qdrant.status().is_success(),
            "ml_service": true,
            "collection": self.qdrant_collection,
            "embedding_dimension": self.embedding_dimension,
            "embedder_version": self.embedder_version,
            "forecast_model_version": self.forecast_model_version,
        }))
    }

    async fn ensure_qdrant_collection(&self) -> Result<(), String> {
        let url = format!("{}/collections/{}", self.qdrant_url, self.qdrant_collection);
        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|error| format!("qdrant collection check: {error}"))?;
        if response.status().is_success() {
            return Ok(());
        }
        if response.status() != HttpStatus::NOT_FOUND {
            return Err(format!(
                "qdrant collection check returned {}",
                response.status()
            ));
        }
        self.client
            .put(&url)
            .json(&json!({
                "vectors": {
                    "size": self.embedding_dimension,
                    "distance": "Cosine"
                }
            }))
            .send()
            .await
            .map_err(|error| format!("qdrant collection create: {error}"))?
            .error_for_status()
            .map_err(|error| format!("qdrant collection create: {error}"))?;
        Ok(())
    }

    async fn classify(
        &self,
        text: &str,
        language: Option<&str>,
        request_id: &str,
    ) -> Result<MlClassificationWithModel, String> {
        let response = self
            .client
            .post(format!("{}/internal/v1/classify", self.ml_service_url))
            .header("x-request-id", request_id)
            .json(&json!({
                "text": text,
                "language": language,
                "top_k": 3,
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

    async fn embed(&self, text: &str, request_id: &str) -> Result<(String, Vec<f32>), String> {
        let response = self
            .client
            .post(format!("{}/internal/v1/embed", self.ml_service_url))
            .header("x-request-id", request_id)
            .json(&json!({
                "text": text,
                "dimension": self.embedding_dimension,
                "normalize": true,
            }))
            .send()
            .await
            .map_err(|error| format!("ML embed request: {error}"))?
            .error_for_status()
            .map_err(|error| format!("ML embed response: {error}"))?
            .json::<MlEmbedResponse>()
            .await
            .map_err(|error| format!("ML embed JSON: {error}"))?;
        let vector = response
            .embedding
            .or_else(|| response.embeddings.into_iter().next())
            .ok_or_else(|| "ML embed returned no vector".to_owned())?;
        if vector.len() != self.embedding_dimension {
            return Err(format!(
                "ML embed returned dimension {}, expected {}",
                vector.len(),
                self.embedding_dimension
            ));
        }
        Ok((self.embedder_version.clone(), vector))
    }

    async fn qdrant_upsert(
        &self,
        ticket_id: i64,
        vector: &[f32],
        topic_id: &str,
        region_id: &str,
        created_at: &str,
    ) -> Result<(), String> {
        self.ensure_qdrant_collection().await?;
        self.client
            .put(format!(
                "{}/collections/{}/points?wait=true",
                self.qdrant_url, self.qdrant_collection
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
        ticket_id: i64,
        topic_id: &str,
        region_id: &str,
    ) -> Result<(), String> {
        self.ensure_qdrant_collection().await?;
        self.client
            .post(format!(
                "{}/collections/{}/points/payload?wait=true",
                self.qdrant_url, self.qdrant_collection
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

    async fn qdrant_delete(&self, ticket_id: i64) -> Result<(), String> {
        self.ensure_qdrant_collection().await?;
        self.client
            .post(format!(
                "{}/collections/{}/points/delete?wait=true",
                self.qdrant_url, self.qdrant_collection
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
        vector: &[f32],
        exclude_id: Option<i64>,
    ) -> Result<Vec<QdrantHit>, String> {
        self.ensure_qdrant_collection().await?;
        let result = self
            .client
            .post(format!(
                "{}/collections/{}/points/search",
                self.qdrant_url, self.qdrant_collection
            ))
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

    async fn resolve_routing(
        &self,
        topic_id: &str,
        region_id: &str,
        service_label: Option<&str>,
        requested_priority: Option<&str>,
    ) -> Result<RoutingDecision, String> {
        let service_row = sqlx::query(
            "SELECT s.id, s.name_ru, rr.reason FROM routing_rules rr JOIN services s ON s.id = rr.service_id WHERE rr.active AND (rr.topic_id = $1 OR rr.topic_id IS NULL) AND (rr.region_id = $2 OR rr.region_id IS NULL) ORDER BY CASE rr.source WHEN 'OFFICIAL' THEN 0 WHEN 'LABEL_HISTORY' THEN 1 WHEN 'MANUAL' THEN 2 ELSE 3 END, CASE WHEN rr.region_id IS NULL THEN 1 ELSE 0 END, rr.precedence ASC, rr.id ASC LIMIT 1",
        )
        .bind(topic_id)
        .bind(region_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("resolve routing rule: {error}"))?;
        let (service_id, service_name, service_reason) = if let Some(row) = service_row {
            (
                row.try_get::<String, _>("id")
                    .map_err(|error| format!("routing service id: {error}"))?,
                row.try_get::<String, _>("name_ru")
                    .map_err(|error| format!("routing service name: {error}"))?,
                row.try_get::<String, _>("reason")
                    .map_err(|error| format!("routing reason: {error}"))?,
            )
        } else if let Some(label) = service_label.filter(|value| !value.trim().is_empty()) {
            let row = sqlx::query(
                "SELECT id, name_ru FROM services WHERE active AND (id = $1 OR lower(name_ru) = lower($1)) LIMIT 1",
            )
            .bind(label.trim())
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("resolve service label: {error}"))?;
            if let Some(row) = row {
                (
                    row.try_get::<String, _>("id")
                        .map_err(|error| format!("service id: {error}"))?,
                    row.try_get::<String, _>("name_ru")
                        .map_err(|error| format!("service name: {error}"))?,
                    "Source service label/history".to_owned(),
                )
            } else {
                (
                    "service_other".to_owned(),
                    "Другая служба".to_owned(),
                    "No authoritative or mapped service rule".to_owned(),
                )
            }
        } else {
            (
                "service_other".to_owned(),
                "Другая служба".to_owned(),
                "No authoritative or mapped service rule".to_owned(),
            )
        };

        let priority_row = sqlx::query(
            "SELECT priority, reason FROM priority_rules WHERE active AND (topic_id = $1 OR topic_id IS NULL) AND (region_id = $2 OR region_id IS NULL) ORDER BY CASE source WHEN 'OFFICIAL' THEN 0 WHEN 'LABEL_HISTORY' THEN 1 WHEN 'MANUAL' THEN 2 ELSE 3 END, CASE WHEN region_id IS NULL THEN 1 ELSE 0 END, precedence ASC, id ASC LIMIT 1",
        )
        .bind(topic_id)
        .bind(region_id)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("resolve priority rule: {error}"))?;
        let (priority, priority_reason) = if let Some(row) = priority_row {
            (
                row.try_get::<String, _>("priority")
                    .map_err(|error| format!("priority value: {error}"))?,
                row.try_get::<String, _>("reason")
                    .map_err(|error| format!("priority reason: {error}"))?,
            )
        } else if let Some(value) = requested_priority.filter(|value| !value.trim().is_empty()) {
            (
                normalize_priority(value),
                "Explicit request priority".to_owned(),
            )
        } else {
            (
                "medium".to_owned(),
                "No priority rule; medium fallback".to_owned(),
            )
        };
        Ok(RoutingDecision {
            service_id,
            service_name,
            priority,
            reason: format!("{service_reason}; {priority_reason}"),
        })
    }

    async fn response_template(
        &self,
        language: &str,
        topic_id: &str,
        service_name: &str,
    ) -> Result<ResponseTemplate, String> {
        let row = sqlx::query(
            "SELECT rt.id, rt.body, rt.language, rt.approved, COALESCE(tp.name_ru, tp.name_kk, tp.id) AS topic_label, (rt.service_id IS NULL OR lower(COALESCE(s.name_ru, '')) = lower($3)) AS service_match FROM response_templates rt LEFT JOIN topics tp ON tp.id = rt.topic_id LEFT JOIN services s ON s.id = rt.service_id WHERE upper(rt.language) = upper($1) AND rt.topic_id = $2 ORDER BY service_match DESC, rt.approved DESC, rt.version DESC, rt.id DESC LIMIT 1",
        )
        .bind(normalize_language(language))
        .bind(topic_id)
        .bind(service_name)
        .fetch_optional(&self.pool)
        .await
        .map_err(|error| format!("fetch response template: {error}"))?;
        let Some(row) = row else {
            return Ok(unavailable_response_template(language));
        };
        if !row
            .try_get::<bool, _>("service_match")
            .map_err(|error| format!("template service match: {error}"))?
        {
            return Ok(unavailable_response_template(language));
        }
        let approved = row.try_get::<bool, _>("approved").unwrap_or(false);
        Ok(ResponseTemplate {
            id: row
                .try_get::<i64, _>("id")
                .map_err(|error| format!("template id: {error}"))?
                .to_string(),
            title: format!(
                "Обращение: {}",
                row.try_get::<String, _>("topic_label")
                    .map_err(|error| format!("template topic: {error}"))?
            ),
            body: row
                .try_get("body")
                .map_err(|error| format!("template body: {error}"))?,
            language: row
                .try_get::<String, _>("language")
                .map_err(|error| format!("template language: {error}"))?
                .to_ascii_lowercase(),
            approved,
            source: if approved {
                "AUTHORITATIVE".to_owned()
            } else {
                "MANUAL_DEMO".to_owned()
            },
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
        let classification = self.classify(text, language, request_id).await?;
        let db_topic = normalize_topic_id(&classification.prediction.topic_id);
        let db_region = normalize_region_id(region_id.unwrap_or("KZ-ASTANA"));
        let db_language =
            normalize_language(language.unwrap_or(&classification.prediction.language));
        let routing = self
            .resolve_routing(&db_topic, &db_region, None, priority)
            .await?;
        let db_priority = routing.priority.clone();
        let external_id = format!(
            "api-{}",
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        );
        let now = Utc::now();
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
        }))
        .bind(json!({"classifier": classification.model_version}))
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("insert ticket: {error}"))?;

        let prediction = prediction_for_db(
            ticket_id,
            &classification,
            &db_priority,
            &routing.service_name,
            &routing.reason,
        );
        let production_prediction_id: i64 = sqlx::query_scalar(
            "INSERT INTO ticket_predictions (ticket_id, model_version, topic_id, service_id, priority, confidence, alternatives, prediction, needs_review) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id",
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
            "routing_reason": routing.reason,
        }))
        .bind(classification.prediction.needs_review)
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("insert prediction: {error}"))?;
        enqueue_classifier_shadow_job(
            &mut tx,
            ticket_id,
            production_prediction_id,
            &classification.model_version,
        )
        .await?;
        tx.commit()
            .await
            .map_err(|error| format!("commit ticket transaction: {error}"))?;

        let (_, vector) = self.embed(text, request_id).await?;
        self.qdrant_upsert(ticket_id, &vector, &db_topic, &db_region, &now.to_rfc3339())
            .await?;
        sqlx::query("UPDATE tickets SET embedding_ref = $2, model_versions = model_versions || $3::jsonb, updated_in_pulse_at = now() WHERE id = $1")
            .bind(ticket_id)
            .bind(format!("qdrant:{}:{ticket_id}", self.qdrant_collection))
            .bind(json!({"embedder": self.embedder_version}))
            .execute(&self.pool)
            .await
            .map_err(|error| format!("store embedding reference: {error}"))?;

        let ticket = self.fetch_ticket_by_id(ticket_id).await?;
        Ok(TicketDetailResponse {
            ticket,
            prediction,
            latest_decision: None,
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
        .bind(request.is_synthetic.unwrap_or(false))
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
            || registered_synthetic != request.is_synthetic.unwrap_or(false)
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
        for item in &request.tickets {
            let external_id = json_text(item, "external_ticket_id")
                .ok_or_else(|| "ticket external_ticket_id is required".to_owned())?;
            let text = json_text(item, "original_text")
                .ok_or_else(|| format!("ticket {external_id} original_text is required"))?;
            let region_id = normalize_region_id(
                &json_text(item, "region_id").unwrap_or_else(|| "KZ-ASTANA".to_owned()),
            );
            let language =
                normalize_language(json_text(item, "language").as_deref().unwrap_or("UNKNOWN"));
            let classification = self.classify(&text, Some(&language), request_id).await?;
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
                )
                .await?;
            let created_at = json_datetime(item, "created_at")?;
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
            .bind(json_datetime_optional(item, "closed_at")?)
            .bind(json_datetime_optional(item, "deadline_at")?)
            .bind(json_text(item, "resolution_text"))
            .bind(json_text(item, "official_response"))
            .bind(json!({
                "topic_id": topic_id,
                "confidence": classification.prediction.confidence,
                "model_version": classification.model_version,
                "routing_reason": routing.reason,
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
                let production_prediction_id: i64 = sqlx::query_scalar(
                    "INSERT INTO ticket_predictions (ticket_id, model_version, topic_id, service_id, priority, confidence, alternatives, prediction, needs_review) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) RETURNING id",
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
                    "routing_reason": routing.reason,
                }))
                .bind(classification.prediction.needs_review)
                .fetch_one(&mut *tx)
                .await
                .map_err(|error| format!("insert imported prediction {external_id}: {error}"))?;
                enqueue_classifier_shadow_job(
                    &mut tx,
                    id,
                    production_prediction_id,
                    &classification.model_version,
                )
                .await?;
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
            let (_, vector) = self.embed(&text, request_id).await?;
            self.qdrant_upsert(ticket_id, &vector, &topic_id, &region_id, &created_at)
                .await?;
            sqlx::query("UPDATE tickets SET embedding_ref = $2, model_versions = model_versions || $3::jsonb, updated_in_pulse_at = now() WHERE id = $1")
                .bind(ticket_id)
                .bind(format!("qdrant:{}:{ticket_id}", self.qdrant_collection))
                .bind(json!({"embedder": self.embedder_version}))
                .execute(&self.pool)
                .await
                .map_err(|error| format!("store imported embedding reference: {error}"))?;
            indexed_rows += 1;
        }
        Ok(ImportResponse {
            import_run_id: import_run_id.to_string(),
            source_system,
            dataset_version,
            total_rows: request.tickets.len() + request.quarantine.len(),
            imported_rows,
            duplicate_rows,
            quarantined_rows: request.quarantine.len(),
            indexed_rows,
            source: "postgres+ml+qdrant".to_owned(),
        })
    }

    pub async fn queue_reindex(&self) -> Result<Value, String> {
        let payload = json!({
            "kind": "reindex_qdrant",
            "collection": self.qdrant_collection,
            "embedding_dimension": self.embedding_dimension,
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
        self.qdrant_delete(numeric_id).await?;
        sqlx::query(
            "UPDATE tickets SET embedding_ref = NULL, updated_in_pulse_at = now() WHERE id = $1",
        )
        .bind(numeric_id)
        .execute(&self.pool)
        .await
        .map_err(|error| format!("clear embedding reference: {error}"))?;
        Ok(json!({
            "ticket_id": ticket.id,
            "collection": self.qdrant_collection,
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
            .push(")::bigint AS previous_total_tickets, COALESCE(AVG(EXTRACT(EPOCH FROM (d.created_at - t.created_at)) / 60.0) FILTER (WHERE t.created_at >= ")
            .push_bind(current_since)
            .push(" AND d.created_at >= t.created_at), 0)::float8 AS avg_decision_minutes, COALESCE(AVG(p.confidence) FILTER (WHERE t.created_at >= ")
            .push_bind(current_since)
            .push("), 0)::float8 AS average_confidence FROM tickets t LEFT JOIN LATERAL (SELECT confidence FROM ticket_predictions WHERE ticket_id = t.id ORDER BY created_at DESC LIMIT 1) p ON TRUE LEFT JOIN LATERAL (SELECT decision, created_at FROM operator_decisions WHERE ticket_id = t.id ORDER BY created_at DESC LIMIT 1) d ON TRUE WHERE t.created_at >= ")
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
        let avg_decision_minutes: f64 = overview_row.try_get("avg_decision_minutes").unwrap_or(0.0);
        let previous_total: i64 = overview_row.try_get("previous_total_tickets").unwrap_or(0);
        let average_confidence: f64 = overview_row.try_get("average_confidence").unwrap_or(0.0);
        let total_change = total_tickets - previous_total;
        let total_change_pct = percent_change(total_tickets, previous_total);

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
        region_query.push(" GROUP BY r.id, r.name_ru, r.name_en ORDER BY tickets DESC, r.id");
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
            "SELECT to_char(date_trunc('day', t.created_at), 'YYYY-MM-DD') AS date, COUNT(*)::bigint AS tickets, COUNT(*) FILTER (WHERE t.status IN ('RESOLVED', 'CLOSED'))::bigint AS resolved FROM tickets t WHERE t.created_at >= ",
        );
        series_query
            .push_bind(current_since)
            .push(" AND t.created_at <= now()");
        push_analytics_filters(&mut series_query, query, "t");
        series_query.push(
            " GROUP BY date_trunc('day', t.created_at) ORDER BY date_trunc('day', t.created_at)",
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
        let time_series = (0..days)
            .map(|offset| {
                let date =
                    (current_since.date_naive() + chrono::Duration::days(offset + 1)).to_string();
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
                "avg_decision_minutes": avg_decision_minutes.max(0.0),
                "average_confidence": average_confidence,
                "previous_total_tickets": previous_total.max(0),
                "change_abs": total_change,
                "change_pct": total_change_pct,
            }),
            by_region,
            by_topic,
            time_series,
        })
    }

    pub async fn analytics_drilldown(
        &self,
        query: &AnalyticsDrilldownQuery,
        dimension: &str,
    ) -> Result<TicketListResponse, String> {
        let mut builder = QueryBuilder::<Postgres>::new(TICKET_SELECT);
        push_drilldown_predicates(&mut builder, query, dimension)?;

        let limit = query.limit.unwrap_or(100).clamp(1, 100);
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
        Ok(TicketListResponse {
            items: rows.into_iter().map(ticket_from_db).collect(),
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
        if values.is_empty() {
            return Ok(ForecastResponse {
                source: "postgres".to_owned(),
                model_version: self.forecast_model_version.clone(),
                model: "seasonal-naive-baseline".to_owned(),
                status: "INSUFFICIENT_HISTORY".to_owned(),
                insufficient_history: true,
                horizon_days,
                history: Vec::new(),
                points: Vec::new(),
                expected_peaks: Vec::new(),
                backtest: json!({"sample_count": 0}),
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
            .unwrap_or(values.len() < 7);
        let forecast_values = payload
            .get("forecast")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
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
        let history = dates
            .iter()
            .zip(values.iter())
            .map(|(date, value)| TimeSeriesPoint {
                date: date.clone(),
                tickets: value.max(0.0).round() as u32,
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
        Ok(ForecastResponse {
            source: "postgres+ml".to_owned(),
            model_version,
            model,
            status,
            insufficient_history,
            horizon_days,
            history,
            points,
            expected_peaks,
            backtest: payload
                .get("backtest")
                .cloned()
                .unwrap_or_else(|| json!({})),
        })
    }

    pub async fn analytics_query(&self, query: &QueryIntentRequest) -> Result<Value, String> {
        let mut intent = query
            .intent
            .as_deref()
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        if intent == "compare" {
            intent = "compare_regions".to_owned();
        }
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
        let service_id = query
            .filters
            .as_ref()
            .and_then(|filters| filters.service_id.clone());
        let status = query
            .filters
            .as_ref()
            .and_then(|filters| filters.status.clone());
        let district = query
            .filters
            .as_ref()
            .and_then(|filters| filters.district.clone());
        let channel = query
            .filters
            .as_ref()
            .and_then(|filters| filters.channel.clone());
        let range = query.range.clone().or_else(|| {
            query
                .filters
                .as_ref()
                .and_then(|filters| filters.range.clone())
        });
        let filters = AnalyticsQuery {
            region_id: region_id.clone(),
            topic_id: topic_id.clone(),
            service_id: service_id.clone(),
            status: status.clone(),
            district: district.clone(),
            channel: channel.clone(),
            range: range.clone().or_else(|| Some("30d".to_owned())),
        };
        let report = self.analytics(&filters).await?;
        let mut rows = match intent.as_str() {
            "count" => vec![json!({
                "period": range.clone().unwrap_or_else(|| "30d".to_owned()),
                "count": report.overview.get("total_tickets").cloned().unwrap_or(json!(0)),
            })],
            "trend" => report
                .time_series
                .iter()
                .map(|point| json!({"period": point.date, "count": point.tickets, "resolved": point.resolved}))
                .collect(),
            "compare_regions" => report
                .by_region
                .iter()
                .map(|bucket| json!({"key": bucket.id, "label": bucket.label, "count": bucket.tickets, "change_abs": bucket.change_abs, "change_pct": bucket.change_pct}))
                .collect(),
            "top_topics" => report
                .by_topic
                .iter()
                .map(|bucket| json!({"key": bucket.id, "label": bucket.label, "count": bucket.tickets, "change_abs": bucket.change_abs, "change_pct": bucket.change_pct}))
                .collect(),
            "spikes" => report
                .time_series
                .windows(2)
                .filter(|window| window[1].tickets >= 3 && window[1].tickets > window[0].tickets.saturating_mul(3) / 2)
                .map(|window| json!({"period": window[1].date, "count": window[1].tickets, "baseline": window[0].tickets, "is_spike": true}))
                .collect(),
            "forecast" => {
                let forecast = self
                    .forecast(&ForecastQuery {
                        horizon: Some(30),
                        horizon_days: None,
                        region_id,
                        topic_id,
                        service_id,
                        status,
                        district,
                        channel,
                    })
                    .await?;
                return serde_json::to_value(forecast)
                    .map(|value| json!({"intent": "forecast", "filters": {"range": range}, "result": value, "source": "postgres+ml"}))
                    .map_err(|error| format!("serialize forecast query: {error}"));
            }
            _ => return Err(format!("unsupported QueryIntent: {intent}")),
        };
        if let Some(limit) = query.limit {
            rows.truncate(limit.clamp(1, 100));
        }
        let total = report
            .overview
            .get("total_tickets")
            .cloned()
            .unwrap_or(json!(0));
        Ok(json!({
            "intent": intent,
            "filters": {"region_id": region_id, "topic_id": topic_id, "range": range.unwrap_or_else(|| "30d".to_owned())},
            "number": total,
            "rows": rows,
            "source": "postgres",
        }))
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

    pub async fn detect_alerts(&self) -> Result<Vec<Alert>, String> {
        let rows = sqlx::query(
            "SELECT region_id, topic_id, COUNT(*) FILTER (WHERE created_at >= now() - interval '7 days' AND created_at <= now())::bigint AS current_count, COUNT(*) FILTER (WHERE created_at < now() - interval '7 days' AND created_at >= now() - interval '35 days')::bigint AS previous_count FROM tickets WHERE created_at >= now() - interval '35 days' AND created_at <= now() GROUP BY region_id, topic_id HAVING COUNT(*) FILTER (WHERE created_at >= now() - interval '7 days' AND created_at <= now()) >= 3",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|error| format!("detect alert series: {error}"))?;
        let day = Utc::now().date_naive().to_string();
        let mut alert_ids = Vec::new();
        for row in rows {
            let region_id: String = row
                .try_get("region_id")
                .map_err(|error| format!("alert region: {error}"))?;
            let topic_id: String = row
                .try_get("topic_id")
                .map_err(|error| format!("alert topic: {error}"))?;
            let current: i64 = row.try_get("current_count").unwrap_or(0);
            let previous: i64 = row.try_get("previous_count").unwrap_or(0);
            let baseline = previous as f64 / 4.0;
            let deviation = if baseline > 0.0 {
                current as f64 / baseline
            } else {
                current as f64
            };
            if deviation < 1.5 {
                continue;
            }
            let severity = if deviation >= 3.0 {
                "CRITICAL"
            } else if deviation >= 2.0 {
                "HIGH"
            } else {
                "MEDIUM"
            };
            let incident_key = format!("topic_spike:{region_id}:{topic_id}:{day}");
            let detail = json!({
                "detector": "robust_baseline",
                "window_days": 7,
                "baseline_days": 28,
                "current_count": current,
                "baseline": baseline,
                "deviation": deviation,
            });
            let id = sqlx::query_scalar::<_, i64>("INSERT INTO alerts (incident_key, alert_type, severity, status, region_id, topic_id, period_start, period_end, current_count, baseline, deviation, detail) VALUES ($1, 'TOPIC_SPIKE', $2, 'OPEN', $3, $4, now() - interval '7 days', now(), $5, $6, $7, $8) ON CONFLICT (incident_key) DO UPDATE SET current_count = EXCLUDED.current_count, baseline = EXCLUDED.baseline, deviation = EXCLUDED.deviation, detail = EXCLUDED.detail WHERE alerts.status <> 'CLOSED' RETURNING id")
                .bind(&incident_key)
                .bind(severity)
                .bind(&region_id)
                .bind(&topic_id)
                .bind(current as i32)
                .bind(baseline)
                .bind(deviation)
                .bind(detail)
                .fetch_optional(&self.pool)
                .await
                .map_err(|error| format!("persist alert: {error}"))?;
            let Some(id) = id else {
                // A closed incident is intentionally not reopened by a
                // repeated detector run during the same cooldown window.
                continue;
            };
            sqlx::query("INSERT INTO alert_ticket_links (alert_id, ticket_id) SELECT $1, id FROM tickets WHERE region_id = $2 AND topic_id = $3 AND created_at >= now() - interval '7 days' AND created_at <= now() ON CONFLICT DO NOTHING")
                .bind(id)
                .bind(&region_id)
                .bind(&topic_id)
                .execute(&self.pool)
                .await
                .map_err(|error| format!("link alert tickets: {error}"))?;
            alert_ids.push(id);
        }
        let mut alerts = Vec::with_capacity(alert_ids.len());
        for id in alert_ids {
            alerts.push(self.alert_from_id(id).await?);
        }
        Ok(alerts)
    }

    async fn alert_from_id(&self, id: i64) -> Result<Alert, String> {
        let row = sqlx::query("SELECT a.id, a.incident_key, a.alert_type, a.severity, a.status, a.region_id, a.topic_id, a.current_count, a.created_at, a.acknowledged_at, a.acknowledged_by, a.closed_at, a.closed_by, COALESCE(a.detail, '{}'::jsonb) AS detail, COALESCE(r.name_ru, a.region_id, 'Все регионы') AS region_name, COALESCE(tp.name_ru, tp.name_kk, a.topic_id, 'Все темы') AS topic_name FROM alerts a LEFT JOIN regions r ON r.id = a.region_id LEFT JOIN topics tp ON tp.id = a.topic_id WHERE a.id = $1")
            .bind(id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("fetch alert: {error}"))?
            .ok_or_else(|| format!("alert {id} not found"))?;
        let links = sqlx::query(
            "SELECT ticket_id::text FROM alert_ticket_links WHERE alert_id = $1 ORDER BY ticket_id",
        )
        .bind(id)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| format!("fetch alert links: {error}"))?
        .into_iter()
        .filter_map(|item| item.try_get::<String, _>("ticket_id").ok())
        .collect::<Vec<_>>();
        let detail: Value = row.try_get("detail").unwrap_or_else(|_| json!({}));
        let region_name: String = row
            .try_get("region_name")
            .unwrap_or_else(|_| "Все регионы".to_owned());
        let topic_name: String = row
            .try_get("topic_name")
            .unwrap_or_else(|_| "Все темы".to_owned());
        let current_count: i32 = row.try_get("current_count").unwrap_or(0);
        let title = detail
            .get("title")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("Всплеск обращений: {topic_name}"));
        let description = detail
            .get("description")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| {
                format!("{current_count} обращений за последние 7 дней; регион: {region_name}.")
            });
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
            linked_ticket_ids: links,
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
                !matches!(
                    cycle.state.as_str(),
                    "PROMOTED" | "REJECTED" | "INSUFFICIENT_FEEDBACK"
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
                "stages": ["COLLECT", "TRAINING", "EVALUATE", "DECISION", "PROMOTE", "REJECT"],
                "production_auto_update": false,
                "trainer": "OFFLINE_TRAINER_REQUIRES_REVIEWED_INPUTS",
            }),
        })
    }

    pub async fn create_learning_cycle(
        &self,
        request: &CreateLearningCycleRequest,
    ) -> Result<LearningCycle, String> {
        let suffix = Utc::now().timestamp_nanos_opt().unwrap_or_default();
        let dataset_version = request
            .dataset_version
            .clone()
            .unwrap_or_else(|| format!("operator-feedback-{suffix}"));
        sqlx::query("INSERT INTO dataset_versions (dataset_version, schema_version, manifest_uri, manifest_sha256, record_count) VALUES ($1, 'unified-ticket.v1', 'postgres://operator-feedback', 'pending', 0)")
            .bind(&dataset_version)
            .execute(&self.pool)
            .await
            .map_err(|error| format!("ensure learning dataset: {error}"))?;
        let cycle_id = format!("cycle-{suffix}");
        let candidate_model_version = request
            .candidate_model_version
            .clone()
            .unwrap_or_else(|| format!("classifier-candidate-{suffix}"));
        let id: i64 = sqlx::query_scalar("INSERT INTO learning_cycles (cycle_id, state, candidate_dataset_version, candidate_model_version, min_feedback_count) VALUES ($1, 'COLLECT', $2, $3, 1) RETURNING id")
            .bind(&cycle_id)
            .bind(&dataset_version)
            .bind(&candidate_model_version)
            .fetch_one(&self.pool)
            .await
            .map_err(|error| format!("create learning cycle: {error}"))?;
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
        let cycle = self.learning_cycle_from_id_lookup(cycle_id).await?;
        let ticket_id = request
            .ticket_id
            .parse::<i64>()
            .map_err(|_| format!("ticket {} not found", request.ticket_id))?;
        let ticket_exists: bool =
            sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM tickets WHERE id = $1)")
                .bind(ticket_id)
                .fetch_one(&self.pool)
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
        let id: i64 = sqlx::query_scalar("INSERT INTO learning_feedback (cycle_id, ticket_id, production_model_version, production_prediction, operator_confirmed_decision, accepted_or_corrected, validation_status) SELECT lc.id, $2, NULL, '{}'::jsonb, $3, $4, 'UNVERIFIED' FROM learning_cycles lc WHERE lc.id = $1 AND lc.state = 'COLLECT' FOR UPDATE OF lc RETURNING id")
            .bind(cycle.id.parse::<i64>().map_err(|_| "invalid learning cycle id".to_owned())?)
            .bind(ticket_id)
            .bind(json!({"feedback_type": feedback_type, "comment": request.comment, "user_id": user_id}))
            .bind(accepted_or_corrected)
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("insert learning feedback: {error}"))?
            .ok_or_else(|| "learning cycle is not collecting feedback".to_owned())?;
        self.learning_feedback_from_id(id, &cycle.id).await
    }

    pub async fn close_learning_cycle(
        &self,
        request: &CloseLearningCycleRequest,
    ) -> Result<Value, String> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin collect close: {error}"))?;
        let cycle = if let Some(cycle_id) = request.cycle_id.as_deref() {
            sqlx::query("SELECT id, state, production_model_version, candidate_dataset_version, candidate_model_version, min_feedback_count FROM learning_cycles WHERE cycle_id = $1 OR id::text = $1 ORDER BY id DESC LIMIT 1 FOR UPDATE")
                .bind(cycle_id)
                .fetch_optional(&mut *tx)
                .await
                .map_err(|error| format!("find collect cycle: {error}"))?
                .ok_or_else(|| format!("learning cycle {cycle_id} not found"))?
        } else {
            sqlx::query("SELECT id, state, production_model_version, candidate_dataset_version, candidate_model_version, min_feedback_count FROM learning_cycles WHERE state = 'COLLECT' ORDER BY created_at DESC, id DESC LIMIT 1 FOR UPDATE")
                .fetch_optional(&mut *tx)
                .await
                .map_err(|error| format!("find collect cycle: {error}"))?
                .ok_or_else(|| "no COLLECT cycle is available".to_owned())?
        };
        let cycle_db_id: i64 = cycle
            .try_get("id")
            .map_err(|error| format!("cycle id: {error}"))?;
        let state: String = cycle
            .try_get("state")
            .map_err(|error| format!("cycle state: {error}"))?;
        if state != "COLLECT" {
            return Err(format!(
                "cycle {} cannot close from state {}",
                cycle_db_id, state
            ));
        }
        let production_version: Option<String> = cycle
            .try_get("production_model_version")
            .map_err(|error| format!("cycle production version: {error}"))?;
        let feedback_count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM learning_feedback WHERE cycle_id = $1 AND validation_status = 'VALID' AND production_model_version = $2 AND production_prediction ? 'topic_id' AND operator_confirmed_decision ? 'decision_id'")
            .bind(cycle_db_id)
            .bind(&production_version)
            .fetch_one(&mut *tx)
            .await
            .map_err(|error| format!("count verified feedback: {error}"))?;
        let minimum: i32 = cycle
            .try_get("min_feedback_count")
            .map_err(|error| format!("minimum feedback count: {error}"))?;
        if feedback_count < i64::from(minimum.max(1)) {
            sqlx::query("UPDATE learning_cycles SET state = 'INSUFFICIENT_FEEDBACK', updated_at = now(), decision_note = 'INSUFFICIENT_FEEDBACK' WHERE id = $1")
                .bind(cycle_db_id)
                .execute(&mut *tx)
                .await
                .map_err(|error| format!("close learning cycle: {error}"))?;
            tx.commit()
                .await
                .map_err(|error| format!("commit insufficient cycle: {error}"))?;
            return Ok(json!({
                "job_id": Value::Null,
                "state": "INSUFFICIENT_FEEDBACK",
                "cycle": self.learning_cycle_from_id(cycle_db_id).await?,
                "production_model_unchanged": true,
            }));
        }
        sqlx::query(
            "UPDATE learning_cycles SET state = 'TRAINING', updated_at = now() WHERE id = $1",
        )
        .bind(cycle_db_id)
        .execute(&mut *tx)
        .await
        .map_err(|error| format!("start learning cycle: {error}"))?;
        let payload = json!({
            "kind": "training",
            "cycle_id": cycle_db_id.to_string(),
            "candidate_model_version": cycle.try_get::<Option<String>, _>("candidate_model_version").map_err(|error| format!("candidate version: {error}"))?,
            "model_type": "classifier",
            "dataset_version": cycle.try_get::<Option<String>, _>("candidate_dataset_version").map_err(|error| format!("candidate dataset: {error}"))?,
            "production_model_version": production_version,
            "samples": [],
            "min_samples": minimum.max(1),
        });
        let job_id: i64 = sqlx::query_scalar("INSERT INTO background_jobs (job_type, payload, state) VALUES ('TRAIN_CLASSIFIER', $1, 'QUEUED') RETURNING id")
            .bind(payload)
            .fetch_one(&mut *tx)
            .await
            .map_err(|error| format!("queue training job: {error}"))?;
        tx.commit()
            .await
            .map_err(|error| format!("commit collect close: {error}"))?;
        Ok(json!({
            "job_id": job_id.to_string(),
            "state": "TRAINING",
            "cycle": self.learning_cycle_from_id(cycle_db_id).await?,
            "production_model_unchanged": true,
        }))
    }

    pub async fn candidate_evaluation(&self) -> Result<Value, String> {
        let cycle = self
            .learning_overview()
            .await?
            .active_cycle
            .ok_or_else(|| "no active learning cycle".to_owned())?;
        let candidate_row = sqlx::query("SELECT artifact_checksum FROM model_versions WHERE model_version = $1 AND status IN ('CANDIDATE', 'SHADOW')")
            .bind(&cycle.candidate_model_version)
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("check candidate evaluation: {error}"))?;
        let candidate_exists = candidate_row.is_some();
        let candidate_checksum = candidate_row
            .map(|row| row.try_get::<Option<String>, _>("artifact_checksum"))
            .transpose()
            .map_err(|error| format!("candidate checksum: {error}"))?
            .flatten();
        let production_row = sqlx::query("SELECT model_version, artifact_checksum FROM model_versions WHERE status = 'PRODUCTION' ORDER BY promoted_at DESC NULLS LAST, id DESC LIMIT 1")
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("check production evaluation: {error}"))?;
        let production = production_row
            .map(|row| {
                Ok::<_, sqlx::Error>((
                    row.try_get::<String, _>("model_version")?,
                    row.try_get::<Option<String>, _>("artifact_checksum")?,
                ))
            })
            .transpose()
            .map_err(|error| format!("production evaluation: {error}"))?;
        let policy_version: String = sqlx::query_scalar(
            "SELECT promotion_policy_version FROM learning_cycles WHERE id = $1",
        )
        .bind(
            cycle
                .id
                .parse::<i64>()
                .map_err(|_| "invalid learning cycle id".to_owned())?,
        )
        .fetch_one(&self.pool)
        .await
        .map_err(|error| format!("fetch promotion policy version: {error}"))?;
        let evaluation = sqlx::query("SELECT evaluation_id, model_version, metrics_json, shadow_metrics_json, critical_regressions, sample_size, decision FROM model_evaluations WHERE model_version = $1 ORDER BY created_at DESC LIMIT 1")
            .bind(&cycle.candidate_model_version)
            .fetch_optional(&self.pool)
            .await
            .map_err(|error| format!("fetch candidate evaluation: {error}"))?;
        let (offline_metrics, shadow_metrics, regressions, sample_size, decision) = if let Some(
            row,
        ) =
            evaluation
        {
            (
                row.try_get::<Value, _>("metrics_json")
                    .map_err(|error| format!("offline metrics: {error}"))?,
                row.try_get::<Value, _>("shadow_metrics_json")
                    .map_err(|error| format!("shadow metrics: {error}"))?,
                row.try_get::<Value, _>("critical_regressions")
                    .map_err(|error| format!("critical regressions: {error}"))?,
                row.try_get::<i32, _>("sample_size")
                    .map_err(|error| format!("sample size: {error}"))?,
                row.try_get::<Option<String>, _>("decision")
                    .map_err(|error| format!("evaluation decision: {error}"))?
                    .unwrap_or_else(|| "PENDING".to_owned()),
            )
        } else {
            (
                json!({"status": if candidate_exists { "EVALUATION_PENDING" } else { "TRAINER_NOT_CONFIGURED" }}),
                json!({}),
                json!([]),
                0,
                "NOT_READY".to_owned(),
            )
        };
        let evidence = json!({
            "decision": decision,
            "offline_metrics": offline_metrics,
            "shadow_metrics": shadow_metrics,
            "critical_regressions": regressions,
            "sample_size": sample_size,
        });
        let ready = production
            .as_ref()
            .is_some_and(|(production_version, production_checksum)| {
                promotion_evidence_ready(
                    &evidence,
                    &cycle.candidate_model_version,
                    production_version,
                    &policy_version,
                    candidate_checksum.as_deref().unwrap_or(""),
                    production_checksum.as_deref().unwrap_or(""),
                )
            });
        Ok(json!({
            "cycle_id": cycle.id,
            "state": cycle.state,
            "offline_metrics": evidence["offline_metrics"],
            "shadow_metrics": evidence["shadow_metrics"],
            "critical_regressions": evidence["critical_regressions"],
            "sample_size": evidence["sample_size"],
            "promotion_policy_version": policy_version,
            "decision": if ready { evidence["decision"].as_str().unwrap_or("NOT_READY") } else { "NOT_READY" },
        }))
    }

    pub async fn promote_learning_cycle(
        &self,
        cycle_id: &str,
        note: Option<&str>,
        user_id: &str,
    ) -> Result<LearningCycle, String> {
        let cycle = self.learning_cycle_from_id_lookup(cycle_id).await?;
        if !matches!(cycle.state.as_str(), "EVALUATE" | "DECISION") {
            return Err(format!(
                "cycle {} cannot be promoted from state {}",
                cycle.id, cycle.state
            ));
        }
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin promotion: {error}"))?;
        let cycle_db_id = cycle
            .id
            .parse::<i64>()
            .map_err(|_| "invalid learning cycle id".to_owned())?;
        let locked_cycle = sqlx::query(
            "SELECT state, promotion_policy_version FROM learning_cycles WHERE id = $1 FOR UPDATE",
        )
        .bind(cycle_db_id)
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("lock learning cycle: {error}"))?;
        let state: String = locked_cycle
            .try_get("state")
            .map_err(|error| format!("cycle state: {error}"))?;
        if !matches!(state.as_str(), "EVALUATE" | "DECISION") {
            return Err("learning cycle is no longer awaiting promotion".to_owned());
        }
        let policy_version: String = locked_cycle
            .try_get("promotion_policy_version")
            .map_err(|error| format!("promotion policy version: {error}"))?;
        let candidate = sqlx::query("SELECT status, artifact_checksum, manifest_uri FROM model_versions WHERE model_version = $1 FOR UPDATE")
            .bind(&cycle.candidate_model_version)
            .fetch_optional(&mut *tx)
            .await
            .map_err(|error| format!("lock candidate model: {error}"))?
            .ok_or_else(|| "candidate model artifact is not available".to_owned())?;
        let candidate_status: String = candidate
            .try_get("status")
            .map_err(|error| format!("candidate status: {error}"))?;
        let candidate_checksum: Option<String> = candidate
            .try_get("artifact_checksum")
            .map_err(|error| format!("candidate checksum: {error}"))?;
        let candidate_uri: Option<String> = candidate
            .try_get("manifest_uri")
            .map_err(|error| format!("candidate manifest URI: {error}"))?;
        if !matches!(candidate_status.as_str(), "CANDIDATE" | "SHADOW")
            || candidate_uri.as_deref().is_none_or(str::is_empty)
        {
            return Err("candidate model artifact is not available".to_owned());
        }
        let production = sqlx::query("SELECT model_version, artifact_checksum FROM model_versions WHERE status = 'PRODUCTION' ORDER BY promoted_at DESC NULLS LAST, id DESC LIMIT 1 FOR UPDATE")
            .fetch_optional(&mut *tx)
            .await
            .map_err(|error| format!("lock production model: {error}"))?
            .ok_or_else(|| "production model artifact is not available".to_owned())?;
        let production_version: String = production
            .try_get("model_version")
            .map_err(|error| format!("production model version: {error}"))?;
        let production_checksum: Option<String> = production
            .try_get("artifact_checksum")
            .map_err(|error| format!("production checksum: {error}"))?;
        let evaluation = sqlx::query("SELECT decision, metrics_json, shadow_metrics_json, critical_regressions, sample_size FROM model_evaluations WHERE model_version = $1 ORDER BY created_at DESC, id DESC LIMIT 1 FOR UPDATE")
            .bind(&cycle.candidate_model_version)
            .fetch_optional(&mut *tx)
            .await
            .map_err(|error| format!("lock candidate evaluation: {error}"))?
            .ok_or_else(|| "candidate evaluation evidence is missing".to_owned())?;
        let evidence = json!({
            "decision": evaluation.try_get::<Option<String>, _>("decision").map_err(|error| format!("evaluation decision: {error}"))?,
            "offline_metrics": evaluation.try_get::<Value, _>("metrics_json").map_err(|error| format!("offline metrics: {error}"))?,
            "shadow_metrics": evaluation.try_get::<Value, _>("shadow_metrics_json").map_err(|error| format!("shadow metrics: {error}"))?,
            "critical_regressions": evaluation.try_get::<Value, _>("critical_regressions").map_err(|error| format!("critical regressions: {error}"))?,
            "sample_size": evaluation.try_get::<i32, _>("sample_size").map_err(|error| format!("evaluation sample size: {error}"))?,
        });
        if !promotion_evidence_ready(
            &evidence,
            &cycle.candidate_model_version,
            &production_version,
            &policy_version,
            candidate_checksum.as_deref().unwrap_or(""),
            production_checksum.as_deref().unwrap_or(""),
        ) {
            return Err(
                "candidate evaluation evidence does not satisfy promotion policy".to_owned(),
            );
        }
        sqlx::query("UPDATE model_versions SET status = 'ARCHIVED' WHERE status = 'PRODUCTION'")
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("archive production model: {error}"))?;
        sqlx::query("UPDATE model_versions SET status = 'PRODUCTION', promoted_at = now() WHERE model_version = $1").bind(&cycle.candidate_model_version).execute(&mut *tx).await.map_err(|error| format!("promote candidate: {error}"))?;
        sqlx::query("UPDATE learning_cycles SET state = 'PROMOTED', decision_note = $2, updated_at = now() WHERE id = $1").bind(cycle_db_id).bind(note.or(Some("promoted"))).execute(&mut *tx).await.map_err(|error| format!("promote learning cycle: {error}"))?;
        sqlx::query("INSERT INTO audit_log (actor_id, action, entity_type, entity_id, reason, metadata) VALUES ($1, 'PROMOTE_MODEL', 'learning_cycle', $2, $3, $4)").bind(user_id).bind(cycle.id.clone()).bind(note).bind(json!({"candidate_model_version": cycle.candidate_model_version})).execute(&mut *tx).await.map_err(|error| format!("audit promotion: {error}"))?;
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
        if cycle.state == "PROMOTED" {
            return Err("a promoted cycle cannot be rejected".to_owned());
        }
        sqlx::query("UPDATE learning_cycles SET state = 'REJECTED', decision_note = $2, updated_at = now() WHERE id = $1")
            .bind(cycle.id.parse::<i64>().unwrap_or_default())
            .bind(note.or(Some("rejected")))
            .execute(&self.pool)
            .await
            .map_err(|error| format!("reject learning cycle: {error}"))?;
        sqlx::query("INSERT INTO audit_log (actor_id, action, entity_type, entity_id, reason) VALUES ($1, 'REJECT_MODEL', 'learning_cycle', $2, $3)")
            .bind(user_id)
            .bind(cycle.id.clone())
            .bind(note)
            .execute(&self.pool)
            .await
            .map_err(|error| format!("audit rejection: {error}"))?;
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
        sqlx::query("UPDATE model_versions SET status = 'REJECTED' WHERE status = 'PRODUCTION' AND model_version <> $1").bind(model_id).execute(&mut *tx).await.map_err(|error| format!("archive model: {error}"))?;
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
        let row = sqlx::query("SELECT lc.id, lc.cycle_id, lc.state, COALESCE(lc.candidate_dataset_version, 'unknown') AS dataset_version, COALESCE(lc.candidate_model_version, 'pending') AS candidate_model_version, lc.created_at, lc.updated_at, COALESCE(lc.decision_note, NULL) AS decision_note, (SELECT COUNT(*)::int FROM learning_feedback lf WHERE lf.cycle_id = lc.id AND lf.validation_status = 'VALID' AND lf.production_model_version = lc.production_model_version AND lf.production_prediction ? 'topic_id' AND lf.operator_confirmed_decision ? 'decision_id') AS feedback_count, COALESCE((SELECT me.metrics_json FROM model_evaluations me WHERE me.model_version = lc.candidate_model_version ORDER BY me.created_at DESC LIMIT 1), '{}'::jsonb) AS metrics_json FROM learning_cycles lc WHERE lc.id = $1")
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
        Ok(LearningCycle {
            id: row.try_get::<i64, _>("id").unwrap_or(id).to_string(),
            state: row.try_get("state").unwrap_or_default(),
            dataset_version: row
                .try_get("dataset_version")
                .unwrap_or_else(|_| "unknown".to_owned()),
            candidate_model_version: row
                .try_get("candidate_model_version")
                .unwrap_or_else(|_| "pending".to_owned()),
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
    ) -> Result<AssistPreviewResponse, String> {
        let (ticket, source, exclude_id, preview_prediction, latest_decision) =
            if let Some(ticket_id) = ticket_id {
                let detail = self.get_ticket(ticket_id).await?;
                let id = detail.ticket.id.parse::<i64>().ok();
                (
                    detail.ticket,
                    "postgres-ticket+ml+qdrant".to_owned(),
                    id,
                    None,
                    detail.latest_decision,
                )
            } else {
                let text = text.ok_or_else(|| "text is required".to_owned())?.trim();
                if text.is_empty() {
                    return Err("text must not be empty".to_owned());
                }
                let classification = self.classify(text, language, request_id).await?;
                let db_topic = normalize_topic_id(&classification.prediction.topic_id);
                let db_region = normalize_region_id(region_id.unwrap_or("KZ-ASTANA"));
                let routing = self
                    .resolve_routing(&db_topic, &db_region, None, None)
                    .await?;
                let prediction = prediction_for_db(
                    0,
                    &classification,
                    &routing.priority,
                    &routing.service_name,
                    &routing.reason,
                );
                let ticket = Ticket {
                    id: "preview".to_owned(),
                    external_ref: "preview".to_owned(),
                    text: text.to_owned(),
                    language: normalize_language(
                        language.unwrap_or(&classification.prediction.language),
                    )
                    .to_ascii_lowercase(),
                    region_id: db_region.clone(),
                    region_name: db_region,
                    topic_id: db_topic,
                    topic_label: classification.prediction.topic.clone(),
                    priority: routing.priority,
                    status: "preview".to_owned(),
                    source: "preview".to_owned(),
                    created_at: Utc::now().to_rfc3339(),
                    updated_at: Utc::now().to_rfc3339(),
                };
                (ticket, "ml+qdrant".to_owned(), None, Some(prediction), None)
            };
        let (_, vector) = self.embed(&ticket.text, request_id).await?;
        let hits = self.qdrant_search(&vector, exclude_id).await?;
        let current_created_at = DateTime::parse_from_rfc3339(&ticket.created_at)
            .ok()
            .map(|value| value.with_timezone(&Utc));
        let current_topic_id = preview_prediction
            .as_ref()
            .map(|value| value.topic_id.clone())
            .unwrap_or_else(|| ticket.topic_id.clone());
        let similar = hits
            .into_iter()
            .map(|hit| SimilarTicket {
                ticket_id: hit.id.to_string(),
                score: hit.score,
                relation: {
                    let same_topic = hit.topic_id == current_topic_id;
                    let same_region = hit.region_id == ticket.region_id;
                    let within_time_window = match (current_created_at, hit.created_at) {
                        (Some(current), Some(candidate)) => {
                            (current - candidate).num_days().unsigned_abs() <= 30
                        }
                        _ => false,
                    };
                    if hit.score >= 0.90 && same_topic && same_region && within_time_window {
                        "duplicate"
                    } else if hit.score >= 0.78 && same_topic && within_time_window {
                        "repeat"
                    } else {
                        "similar"
                    }
                    .to_owned()
                },
                topic_id: hit.topic_id,
                region_id: hit.region_id,
            })
            .collect::<Vec<_>>();
        let prediction = if let Some(prediction) = preview_prediction {
            prediction
        } else {
            self.fetch_prediction(
                ticket
                    .id
                    .parse::<i64>()
                    .map_err(|_| "stored ticket has invalid database id".to_owned())?,
            )
            .await?
        };
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
        let (template_topic, template_service) = latest_decision
            .as_ref()
            .map(|decision| {
                (
                    decision.confirmed_topic_id.as_str(),
                    decision.service.as_str(),
                )
            })
            .unwrap_or((
                prediction.topic_id.as_str(),
                prediction.recommended_service.as_str(),
            ));
        Ok(AssistPreviewResponse {
            response_template: self
                .response_template(&ticket.language, template_topic, template_service)
                .await?,
            ticket,
            prediction,
            similar_tickets: similar,
            duplicate_candidates,
            repeat_candidates,
            source,
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
        let topic_exists: bool =
            sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM topics WHERE id = $1)")
                .bind(&confirmed_topic)
                .fetch_one(&self.pool)
                .await
                .map_err(|error| format!("check topic: {error}"))?;
        if !topic_exists {
            return Err(format!("unknown topic_id: {confirmed_topic}"));
        }
        let routing = self
            .resolve_routing(
                &confirmed_topic,
                &ticket.region_id,
                None,
                request.priority.as_deref(),
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
        let decision_value = if action == "correct" {
            "CORRECTED"
        } else {
            "CONFIRMED"
        };
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|error| format!("begin decision transaction: {error}"))?;
        let decision_id: i64 = sqlx::query_scalar(
            "INSERT INTO operator_decisions (ticket_id, user_id, confirmed_topic_id, confirmed_service_id, confirmed_priority, decision, feedback) VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING id",
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
            "routing_reason": routing.reason,
        }))
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("insert operator decision: {error}"))?;
        let confirmed_payload = json!({
            "action": action,
            "topic_id": confirmed_topic,
            "service": service,
            "priority": priority,
            "decision_id": decision_id.to_string(),
        });
        sqlx::query("UPDATE tickets SET topic_id = $2, service_id = $3, priority = $4, status = 'TRIAGED', operator_confirmed_decision = $5, needs_review = false, updated_in_pulse_at = now() WHERE id = $1")
            .bind(ticket.id.parse::<i64>().map_err(|_| "invalid database ticket id".to_owned())?)
            .bind(&confirmed_topic)
            .bind(&service_id)
            .bind(&priority)
            .bind(&confirmed_payload)
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("update ticket decision: {error}"))?;
        let collect_cycle = sqlx::query(
            "SELECT lc.id, lc.production_model_version FROM learning_cycles lc WHERE lc.state = 'COLLECT' AND (lc.production_model_version = $1 OR (lc.production_model_version IS NULL AND NOT EXISTS (SELECT 1 FROM learning_feedback lf WHERE lf.cycle_id = lc.id AND lf.validation_status = 'VALID'))) ORDER BY lc.created_at DESC, lc.id DESC LIMIT 1 FOR UPDATE OF lc",
        )
        .bind(&prediction.model_version)
        .fetch_optional(&mut *tx)
        .await
        .map_err(|error| format!("find collecting learning cycle: {error}"))?;
        if let Some(cycle) = collect_cycle {
            let cycle_id: i64 = cycle
                .try_get("id")
                .map_err(|error| format!("collect cycle id: {error}"))?;
            if cycle
                .try_get::<Option<String>, _>("production_model_version")
                .map_err(|error| format!("collect production version: {error}"))?
                .is_none()
            {
                sqlx::query("UPDATE learning_cycles SET production_model_version = $2, updated_at = now() WHERE id = $1")
                    .bind(cycle_id)
                    .bind(&prediction.model_version)
                    .execute(&mut *tx)
                    .await
                    .map_err(|error| format!("pin collect production version: {error}"))?;
            }
            sqlx::query("INSERT INTO learning_feedback (cycle_id, ticket_id, production_model_version, production_prediction, operator_confirmed_decision, accepted_or_corrected) VALUES ($1, $2, $3, $4, $5, $6)")
                .bind(cycle_id)
                .bind(ticket.id.parse::<i64>().map_err(|_| "invalid database ticket id".to_owned())?)
                .bind(&prediction.model_version)
                .bind(json!({"topic_id": prediction.topic_id, "confidence": prediction.confidence}))
                .bind(&confirmed_payload)
                .bind(if action == "correct" { "CORRECTED" } else { "ACCEPTED" })
                .execute(&mut *tx)
                .await
                .map_err(|error| format!("insert learning feedback: {error}"))?;
        }
        tx.commit()
            .await
            .map_err(|error| format!("commit decision transaction: {error}"))?;
        // Operator corrections change the deterministic retrieval metadata.
        // Keep the vector itself stable, but update its Qdrant payload so the
        // next similarity decision sees the confirmed topic and region.
        self.qdrant_update_payload(
            ticket.id.parse::<i64>().unwrap_or_default(),
            &confirmed_topic,
            &ticket.region_id,
        )
        .await?;
        let decision = OperatorDecision {
            id: format!("decision-{decision_id}"),
            ticket_id: ticket.id.clone(),
            action: action.to_owned(),
            predicted_topic_id: prediction.topic_id.clone(),
            confirmed_topic_id: confirmed_topic,
            service,
            priority,
            note: request.note.clone(),
            user_id: user_id.to_owned(),
            created_at: Utc::now().to_rfc3339(),
        };
        let mut ticket = self
            .fetch_ticket_by_id(ticket.id.parse::<i64>().unwrap_or_default())
            .await?;
        ticket.status = "triaged".to_owned();
        Ok(DecisionResponse {
            ticket,
            prediction,
            decision,
        })
    }

    pub async fn relation_feedback(
        &self,
        ticket_id: &str,
        related_ticket_id: Option<&str>,
        relation: &str,
        decision: &str,
        user_id: &str,
    ) -> Result<(), String> {
        let ticket = self.fetch_ticket(ticket_id).await?;
        let related = match related_ticket_id {
            Some(value) => Some(self.fetch_ticket(value).await?),
            None => None,
        };
        sqlx::query("INSERT INTO relation_feedback (ticket_id, related_ticket_id, relation, decision, user_id) VALUES ($1, $2, $3, $4, $5)")
            .bind(ticket.id.parse::<i64>().map_err(|_| "invalid ticket id".to_owned())?)
            .bind(related.map(|item| item.id.parse::<i64>().unwrap_or_default()))
            .bind(relation.to_ascii_uppercase())
            .bind(decision.to_ascii_uppercase())
            .bind(user_id)
            .execute(&self.pool)
            .await
            .map_err(|error| format!("insert relation feedback: {error}"))?;
        Ok(())
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
            "SELECT d.id, d.ticket_id, d.decision, COALESCE(d.feedback->>'predicted_topic_id', d.confirmed_topic_id) AS predicted_topic_id, d.confirmed_topic_id, COALESCE(d.confirmed_priority, 'normal') AS confirmed_priority, d.feedback->>'service' AS service, COALESCE(d.user_id, 'unknown') AS user_id, d.feedback->>'note' AS note, d.created_at FROM operator_decisions d WHERE d.ticket_id = $1 ORDER BY d.created_at DESC, d.id DESC LIMIT 1",
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
    let confidence_state = core_confidence_state(
        row.prediction
            .get("confidence_state")
            .and_then(Value::as_str),
        confidence,
        row.needs_review,
    );
    Prediction {
        ticket_id: row.ticket_id.to_string(),
        model_version: row.model_version,
        topic_id: row.topic_id.unwrap_or_else(|| "unknown".to_owned()),
        topic_label: row.topic_label,
        confidence,
        confidence_state: confidence_state.to_owned(),
        recommended_service: row.service_name,
        predicted_priority: row.priority,
        routing_reason: row
            .prediction
            .get("routing_reason")
            .and_then(Value::as_str)
            .unwrap_or("persisted routing rule")
            .to_owned(),
        alternatives,
        created_at: row.created_at.to_rfc3339(),
    }
}

fn core_confidence_state(
    source: Option<&str>,
    confidence: f32,
    needs_review: bool,
) -> &'static str {
    match source {
        Some("CONFIDENT" | "confident" | "high") => "high",
        Some("UNCERTAIN" | "uncertain" | "medium") => "medium",
        Some("LOW_CONFIDENCE" | "low_confidence" | "low") => "low",
        Some(_) => "low",
        None if needs_review => "low",
        None if confidence >= 0.85 => "high",
        None => "medium",
    }
}

fn prediction_for_db(
    ticket_id: i64,
    classification: &MlClassificationWithModel,
    priority: &str,
    service_name: &str,
    routing_reason: &str,
) -> Prediction {
    let topic_id = normalize_topic_id(&classification.prediction.topic_id);
    let confidence = classification.prediction.confidence;
    Prediction {
        ticket_id: ticket_id.to_string(),
        model_version: classification.model_version.clone(),
        topic_id,
        topic_label: classification.prediction.topic.clone(),
        confidence,
        confidence_state: core_confidence_state(
            Some(&classification.prediction.confidence_state),
            confidence,
            classification.prediction.needs_review,
        )
        .to_owned(),
        recommended_service: service_name.to_owned(),
        predicted_priority: priority.to_owned(),
        routing_reason: routing_reason.to_owned(),
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
        confirmed_topic_id: row
            .confirmed_topic_id
            .unwrap_or_else(|| "unknown".to_owned()),
        service: row.service.unwrap_or_else(|| "Другая служба".to_owned()),
        priority: row.confirmed_priority,
        note: row.note,
        user_id: row.user_id,
        created_at: row.created_at.to_rfc3339(),
    }
}

fn response_template_for(language: &str, topic: &str) -> ResponseTemplate {
    ResponseTemplate {
        id: format!("db-template-{}", language.to_ascii_lowercase()),
        title: format!("Обращение: {topic}"),
        body: if language.eq_ignore_ascii_case("KZ") || language.eq_ignore_ascii_case("kk") {
            "Өтінішіңіз тіркелді және жауапты қызметке жіберілді.".to_owned()
        } else {
            "Обращение зарегистрировано и направлено в ответственную службу.".to_owned()
        },
        language: language.to_ascii_lowercase(),
        approved: false,
        source: "MANUAL_DEMO".to_owned(),
    }
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
        "ecology" => "environment".to_owned(),
        "waste" => "waste_management".to_owned(),
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

fn promotion_evidence_ready(
    evidence: &Value,
    candidate_model: &str,
    production_model: &str,
    promotion_policy_version: &str,
    candidate_checksum: &str,
    production_checksum: &str,
) -> bool {
    let valid_checksum = |value: &str| {
        value.strip_prefix("sha256:").is_some_and(|hex| {
            hex.len() == 64
                && hex
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        })
    };
    let offline = &evidence["offline_metrics"];
    let shadow = &evidence["shadow_metrics"];
    let offline_count = offline["sample_count"].as_u64().unwrap_or(0);
    let offline_minimum = offline["policy"]["min_total_samples"].as_u64().unwrap_or(0);
    let shadow_count = shadow["sample_count"].as_u64().unwrap_or(0);
    let shadow_minimum = shadow["policy"]["min_samples"].as_u64().unwrap_or(0);
    let real_count = shadow["origin_counts"]["real"].as_u64().unwrap_or(0);
    let real_minimum = shadow["policy"]["min_real_samples"].as_u64().unwrap_or(0);
    let real_metrics_valid = match (
        shadow["real_production_agreement"].as_f64(),
        shadow["real_candidate_agreement"].as_f64(),
        shadow["real_correction_rate_delta"].as_f64(),
        shadow["policy"]["max_correction_rate_increase"].as_f64(),
    ) {
        (Some(production), Some(candidate), Some(delta), Some(limit)) => {
            (0.0..=1.0).contains(&production)
                && (0.0..=1.0).contains(&candidate)
                && (0.0..=1.0).contains(&limit)
                && (delta - (production - candidate)).abs() <= 0.000002
                && delta <= limit
        }
        _ => false,
    };
    evidence["decision"] == "READY_TO_REVIEW"
        && offline["report_version"] == "classifier-pair-evaluation.v1"
        && offline["decision"] == "PENDING_HUMAN_REVIEW"
        && offline["candidate"]["model_version"] == candidate_model
        && offline["production"]["model_version"] == production_model
        && offline["candidate"]["artifact_checksum"] == candidate_checksum
        && offline["production"]["artifact_checksum"] == production_checksum
        && offline["policy"]["policy_version"] == "classifier-critical-regression.v1"
        && valid_checksum(candidate_checksum)
        && valid_checksum(production_checksum)
        && offline_count > 0
        && offline_minimum > 0
        && offline_count >= offline_minimum
        && offline["regressed_critical_topics"] == json!([])
        && shadow["report_version"] == "classifier-shadow-evaluation.v1"
        && shadow["gate_population"] == "real_only.v1"
        && shadow["status"] == "VALID"
        && shadow["decision"] == "PENDING_HUMAN_REVIEW"
        && shadow["candidate_model_version"] == candidate_model
        && shadow["production_model_version"] == production_model
        && shadow["promotion_policy_version"] == promotion_policy_version
        && shadow["policy"]["policy_version"] == "classifier-shadow-policy.v1"
        && shadow["policy"]["promotion_policy_version"] == promotion_policy_version
        && shadow["blind_ab_enabled"].is_boolean()
        && shadow_count > 0
        && shadow_minimum > 0
        && shadow_count >= shadow_minimum
        && real_minimum > 0
        && real_count >= real_minimum
        && real_metrics_valid
        && shadow["global_regression"] == false
        && evidence["sample_size"] == shadow_count
        && shadow["critical_regressions"] == json!([])
        && evidence["critical_regressions"] == json!([])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored_prediction(state: Option<&str>) -> DbPrediction {
        DbPrediction {
            ticket_id: 1,
            model_version: "classifier-test".to_owned(),
            topic_id: Some("roads".to_owned()),
            topic_label: "Дороги".to_owned(),
            service_name: "Дорожная служба".to_owned(),
            priority: "normal".to_owned(),
            confidence: Some(0.94),
            alternatives: json!([]),
            prediction: state.map_or_else(|| json!({}), |value| json!({"confidence_state": value})),
            needs_review: true,
            created_at: Utc::now(),
        }
    }

    #[test]
    fn persisted_prediction_preserves_model_confidence_state() {
        assert_eq!(
            prediction_from_db(stored_prediction(Some("CONFIDENT"))).confidence_state,
            "high"
        );
        assert_eq!(
            prediction_from_db(stored_prediction(Some("UNCERTAIN"))).confidence_state,
            "medium"
        );
        assert_eq!(
            prediction_from_db(stored_prediction(Some("LOW_CONFIDENCE"))).confidence_state,
            "low"
        );
        assert_eq!(
            prediction_from_db(stored_prediction(None)).confidence_state,
            "low"
        );
    }

    #[test]
    fn fresh_prediction_uses_the_same_core_confidence_state() {
        let classification = MlClassificationWithModel {
            model_version: "classifier-test".to_owned(),
            prediction: MlClassification {
                language: "RU".to_owned(),
                topic_id: "roads".to_owned(),
                topic: "Дороги".to_owned(),
                confidence: 0.94,
                confidence_state: "UNCERTAIN".to_owned(),
                needs_review: true,
                alternatives: vec![],
            },
        };
        assert_eq!(
            prediction_for_db(1, &classification, "normal", "Дорожная служба", "test")
                .confidence_state,
            "medium"
        );
    }

    #[test]
    fn promotion_requires_real_offline_and_shadow_evidence() {
        let candidate_checksum = format!("sha256:{}", "a".repeat(64));
        let production_checksum = format!("sha256:{}", "b".repeat(64));
        let mut evidence = json!({
            "decision": "READY_TO_REVIEW",
            "offline_metrics": {
                "report_version": "classifier-pair-evaluation.v1",
                "decision": "PENDING_HUMAN_REVIEW",
                "sample_count": 48,
                "policy": {"policy_version": "classifier-critical-regression.v1", "min_total_samples": 40},
                "candidate": {"model_version": "candidate-v1", "artifact_checksum": candidate_checksum},
                "production": {"model_version": "production-v1", "artifact_checksum": production_checksum},
                "regressed_critical_topics": []
            },
            "shadow_metrics": {
                "report_version": "classifier-shadow-evaluation.v1",
                "gate_population": "real_only.v1",
                "status": "VALID",
                "candidate_model_version": "candidate-v1",
                "production_model_version": "production-v1",
                "promotion_policy_version": "policy-v1",
                "blind_ab_enabled": false,
                "sample_count": 30,
                "origin_counts": {"real": 30},
                "policy": {"policy_version": "classifier-shadow-policy.v1", "promotion_policy_version": "policy-v1", "min_samples": 30, "min_real_samples": 30, "max_correction_rate_increase": 0.05},
                "decision": "PENDING_HUMAN_REVIEW",
                "real_production_agreement": 0.8,
                "real_candidate_agreement": 0.825,
                "real_correction_rate_delta": -0.025,
                "global_regression": false,
                "critical_regressions": []
            },
            "critical_regressions": [],
            "sample_size": 30
        });
        let ready = |value: &Value| {
            promotion_evidence_ready(
                value,
                "candidate-v1",
                "production-v1",
                "policy-v1",
                &candidate_checksum,
                &production_checksum,
            )
        };
        assert!(ready(&evidence));
        evidence["shadow_metrics"]["gate_population"] = Value::Null;
        assert!(!ready(&evidence));
        evidence["shadow_metrics"]["gate_population"] = json!("real_only.v1");
        evidence["shadow_metrics"]["real_correction_rate_delta"] = json!(0.1);
        assert!(!ready(&evidence));
        evidence["shadow_metrics"]["real_correction_rate_delta"] = json!(-0.025);
        assert!(ready(&evidence));
        evidence["shadow_metrics"]["real_candidate_agreement"] = json!(0.74);
        evidence["shadow_metrics"]["real_correction_rate_delta"] = json!(0.06);
        assert!(!ready(&evidence));
        evidence["shadow_metrics"]["real_candidate_agreement"] = json!(0.825);
        evidence["shadow_metrics"]["real_correction_rate_delta"] = json!(-0.025);
        assert!(ready(&evidence));
        assert!(!promotion_evidence_ready(
            &evidence,
            "candidate-v1",
            "production-v1",
            "policy-v1",
            "sha256:wrong",
            &production_checksum,
        ));
        evidence["shadow_metrics"]["sample_count"] = json!(29);
        assert!(!ready(&evidence));
        evidence["shadow_metrics"]["sample_count"] = json!(30);
        evidence["shadow_metrics"]["origin_counts"]["real"] = json!(29);
        assert!(!ready(&evidence));
        evidence["shadow_metrics"]["origin_counts"]["real"] = json!(30);
        evidence["offline_metrics"] = json!({"status": "FAKE_TRAINER_NO_METRICS"});
        assert!(!ready(&evidence));
        evidence["offline_metrics"] = json!({
            "report_version": "classifier-pair-evaluation.v1",
            "decision": "CRITICAL_REGRESSION",
            "sample_count": 48,
            "policy": {"policy_version": "classifier-critical-regression.v1", "min_total_samples": 40},
            "candidate": {"model_version": "candidate-v1", "artifact_checksum": candidate_checksum},
            "production": {"model_version": "production-v1", "artifact_checksum": production_checksum},
            "regressed_critical_topics": ["roads"]
        });
        assert!(!ready(&evidence));
        evidence["offline_metrics"]["decision"] = json!("PENDING_HUMAN_REVIEW");
        evidence["offline_metrics"]["regressed_critical_topics"] = json!([]);
        evidence["shadow_metrics"]["status"] = json!("INSUFFICIENT_EVIDENCE");
        assert!(!ready(&evidence));
    }
}
