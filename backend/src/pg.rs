//! PostgreSQL/Qdrant/ML integration for the real Core runtime.
//!
//! The demo store in `lib.rs` is intentionally kept for unit tests and for
//! an explicitly selected `PULSE_STORAGE=memory` run.  Compose uses this
//! repository instead.  PostgreSQL is the source of truth; Qdrant only keeps
//! the vector index and the ML service owns classification/embedding.

use crate::{
    AlternativePrediction, AnalyticsQuery, AnalyticsResponse, AssistPreviewResponse,
    DecisionRequest, DecisionResponse, MetricBucket, OperatorDecision, Prediction,
    ResponseTemplate, SimilarTicket, Ticket, TicketDetailResponse, TicketListResponse, TicketQuery,
    TimeSeriesPoint, Topic,
};
use chrono::{DateTime, Utc};
use reqwest::{Client, StatusCode as HttpStatus};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{postgres::PgPoolOptions, FromRow, PgPool, Postgres, QueryBuilder, Row};

const EMBEDDING_DIMENSION: usize = 32;
const QDRANT_COLLECTION: &str = "pulse109_tickets";

#[derive(Clone)]
pub struct PgRepository {
    pub pool: PgPool,
    pub qdrant_url: String,
    pub ml_service_url: String,
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
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .min_connections(1)
            .connect_lazy(database_url)
            .map_err(|error| format!("invalid DATABASE_URL: {error}"))?;
        Ok(Self {
            pool,
            qdrant_url: qdrant_url.into().trim_end_matches('/').to_owned(),
            ml_service_url: ml_service_url.into().trim_end_matches('/').to_owned(),
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
        Ok(json!({
            "postgres": true,
            "qdrant": qdrant.status().is_success(),
            "ml_service": ml.status().is_success(),
            "collection": QDRANT_COLLECTION,
        }))
    }

    async fn ensure_qdrant_collection(&self) -> Result<(), String> {
        let url = format!("{}/collections/{}", self.qdrant_url, QDRANT_COLLECTION);
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
                    "size": EMBEDDING_DIMENSION,
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
                "dimension": EMBEDDING_DIMENSION,
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
        if vector.len() != EMBEDDING_DIMENSION {
            return Err(format!(
                "ML embed returned dimension {}, expected {}",
                vector.len(),
                EMBEDDING_DIMENSION
            ));
        }
        Ok(("embedder-demo-2026-09-21-001".to_owned(), vector))
    }

    async fn qdrant_upsert(
        &self,
        ticket_id: i64,
        vector: &[f32],
        topic_id: &str,
        region_id: &str,
        external_ticket_id: &str,
    ) -> Result<(), String> {
        self.ensure_qdrant_collection().await?;
        self.client
            .put(format!(
                "{}/collections/{}/points?wait=true",
                self.qdrant_url, QDRANT_COLLECTION
            ))
            .json(&json!({
                "points": [{
                    "id": ticket_id,
                    "vector": vector,
                    "payload": {
                        "ticket_id": ticket_id.to_string(),
                        "external_ticket_id": external_ticket_id,
                        "topic_id": topic_id,
                        "region_id": region_id,
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
                self.qdrant_url, QDRANT_COLLECTION
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
            });
        }
        Ok(hits)
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
        let db_priority = priority.unwrap_or("normal").to_owned();
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
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'service_other', $10, 'OPEN', $11, $12)
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
        .bind(&db_priority)
        .bind(json!({
            "topic_id": db_topic,
            "confidence": classification.prediction.confidence,
            "model_version": classification.model_version,
        }))
        .bind(json!({"classifier": classification.model_version}))
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("insert ticket: {error}"))?;

        let prediction = prediction_for_db(ticket_id, &classification, &db_priority);
        sqlx::query(
            "INSERT INTO ticket_predictions (ticket_id, model_version, topic_id, service_id, priority, confidence, alternatives, prediction, needs_review) VALUES ($1, $2, $3, 'service_other', $4, $5, $6, $7, $8)",
        )
        .bind(ticket_id)
        .bind(&classification.model_version)
        .bind(&db_topic)
        .bind(&db_priority)
        .bind(classification.prediction.confidence as f64)
        .bind(alternatives_json(&classification.prediction))
        .bind(json!({
            "topic_id": db_topic,
            "topic": classification.prediction.topic,
            "confidence": classification.prediction.confidence,
            "confidence_state": classification.prediction.confidence_state,
            "needs_review": classification.prediction.needs_review,
        }))
        .bind(classification.prediction.needs_review)
        .execute(&mut *tx)
        .await
        .map_err(|error| format!("insert prediction: {error}"))?;
        tx.commit()
            .await
            .map_err(|error| format!("commit ticket transaction: {error}"))?;

        let (_, vector) = self.embed(text, request_id).await?;
        self.qdrant_upsert(ticket_id, &vector, &db_topic, &db_region, &external_id)
            .await?;
        sqlx::query("UPDATE tickets SET embedding_ref = $2, model_versions = model_versions || $3::jsonb, updated_in_pulse_at = now() WHERE id = $1")
            .bind(ticket_id)
            .bind(format!("qdrant:{QDRANT_COLLECTION}:{ticket_id}"))
            .bind(json!({"embedder": "embedder-demo-2026-09-21-001"}))
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
        let mut overview_query = QueryBuilder::<Postgres>::new(
            "SELECT COUNT(*)::bigint AS total_tickets, COUNT(*) FILTER (WHERE t.status IN ('OPEN', 'TRIAGED'))::bigint AS open_tickets, COUNT(*) FILTER (WHERE t.status IN ('RESOLVED', 'CLOSED'))::bigint AS resolved_tickets, COUNT(*) FILTER (WHERE lower(COALESCE(t.priority, '')) = 'high')::bigint AS high_priority_tickets, COUNT(*) FILTER (WHERE t.operator_confirmed_decision IS NOT NULL)::bigint AS operator_decisions FROM tickets t WHERE TRUE",
        );
        if let Some(region_id) = &query.region_id {
            overview_query
                .push(" AND t.region_id = ")
                .push_bind(normalize_region_id(region_id));
        }
        if let Some(topic_id) = &query.topic_id {
            overview_query
                .push(" AND t.topic_id = ")
                .push_bind(normalize_topic_id(topic_id));
        }
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

        let region_rows = sqlx::query(
            "SELECT r.id, COALESCE(r.name_ru, r.name_en, r.id) AS label, COUNT(t.id)::bigint AS tickets, COUNT(t.id) FILTER (WHERE lower(COALESCE(t.priority, '')) = 'high')::bigint AS high_priority, COALESCE(AVG(p.confidence::float8), 0)::float8 AS avg_confidence FROM regions r LEFT JOIN tickets t ON t.region_id = r.id LEFT JOIN LATERAL (SELECT confidence FROM ticket_predictions WHERE ticket_id = t.id ORDER BY created_at DESC LIMIT 1) p ON TRUE GROUP BY r.id, r.name_ru, r.name_en ORDER BY tickets DESC, r.id",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|error| format!("analytics regions: {error}"))?;
        let by_region = region_rows
            .into_iter()
            .map(|row| MetricBucket {
                id: row.try_get("id").unwrap_or_default(),
                label: row.try_get("label").unwrap_or_default(),
                tickets: row.try_get::<i64, _>("tickets").unwrap_or(0).max(0) as usize,
                high_priority: row.try_get::<i64, _>("high_priority").unwrap_or(0).max(0) as usize,
                avg_confidence: row.try_get::<f64, _>("avg_confidence").unwrap_or(0.0) as f32,
            })
            .collect::<Vec<_>>();
        let topic_rows = sqlx::query(
            "SELECT tp.id, COALESCE(tp.name_ru, tp.name_kk, tp.id) AS label, COUNT(t.id)::bigint AS tickets, COUNT(t.id) FILTER (WHERE lower(COALESCE(t.priority, '')) = 'high')::bigint AS high_priority, COALESCE(AVG(p.confidence::float8), 0)::float8 AS avg_confidence FROM topics tp LEFT JOIN tickets t ON t.topic_id = tp.id LEFT JOIN LATERAL (SELECT confidence FROM ticket_predictions WHERE ticket_id = t.id ORDER BY created_at DESC LIMIT 1) p ON TRUE GROUP BY tp.id, tp.name_ru, tp.name_kk ORDER BY tickets DESC, tp.id",
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|error| format!("analytics topics: {error}"))?;
        let by_topic = topic_rows
            .into_iter()
            .map(|row| MetricBucket {
                id: row.try_get("id").unwrap_or_default(),
                label: row.try_get("label").unwrap_or_default(),
                tickets: row.try_get::<i64, _>("tickets").unwrap_or(0).max(0) as usize,
                high_priority: row.try_get::<i64, _>("high_priority").unwrap_or(0).max(0) as usize,
                avg_confidence: row.try_get::<f64, _>("avg_confidence").unwrap_or(0.0) as f32,
            })
            .collect::<Vec<_>>();
        let days = query
            .range
            .as_deref()
            .and_then(|value| value.strip_suffix('d'))
            .and_then(|value| value.parse::<i64>().ok())
            .unwrap_or(7)
            .clamp(1, 366);
        let since = Utc::now() - chrono::Duration::days(days);
        let series_rows = sqlx::query(
            "SELECT to_char(date_trunc('day', t.created_at), 'YYYY-MM-DD') AS date, COUNT(*)::bigint AS tickets, COUNT(*) FILTER (WHERE t.status IN ('RESOLVED', 'CLOSED'))::bigint AS resolved FROM tickets t WHERE t.created_at >= $1 GROUP BY date_trunc('day', t.created_at) ORDER BY date_trunc('day', t.created_at)",
        )
        .bind(since)
        .fetch_all(&self.pool)
        .await
        .map_err(|error| format!("analytics time series: {error}"))?;
        let time_series = series_rows
            .into_iter()
            .map(|row| TimeSeriesPoint {
                date: row.try_get("date").unwrap_or_default(),
                tickets: row.try_get::<i64, _>("tickets").unwrap_or(0).max(0) as u32,
                resolved: row.try_get::<i64, _>("resolved").unwrap_or(0).max(0) as u32,
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
                "confirmed_decisions": 0,
                "corrected_decisions": 0,
                "average_confidence": 0.0,
            }),
            by_region,
            by_topic,
            time_series,
        })
    }

    pub async fn assist_preview(
        &self,
        ticket_id: Option<&str>,
        text: Option<&str>,
        language: Option<&str>,
        region_id: Option<&str>,
        request_id: &str,
    ) -> Result<AssistPreviewResponse, String> {
        let (ticket, source, exclude_id, preview_prediction) = if let Some(ticket_id) = ticket_id {
            let detail = self.get_ticket(ticket_id).await?;
            let id = detail.ticket.id.parse::<i64>().ok();
            (
                detail.ticket,
                "postgres-ticket+ml+qdrant".to_owned(),
                id,
                None,
            )
        } else {
            let text = text.ok_or_else(|| "text is required".to_owned())?.trim();
            if text.is_empty() {
                return Err("text must not be empty".to_owned());
            }
            let classification = self.classify(text, language, request_id).await?;
            let db_topic = normalize_topic_id(&classification.prediction.topic_id);
            let db_region = normalize_region_id(region_id.unwrap_or("KZ-ASTANA"));
            let prediction = prediction_for_db(0, &classification, "normal");
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
                priority: "normal".to_owned(),
                status: "preview".to_owned(),
                source: "preview".to_owned(),
                created_at: Utc::now().to_rfc3339(),
                updated_at: Utc::now().to_rfc3339(),
            };
            (ticket, "ml+qdrant".to_owned(), None, Some(prediction))
        };
        let (_, vector) = self.embed(&ticket.text, request_id).await?;
        let hits = self.qdrant_search(&vector, exclude_id).await?;
        let similar = hits
            .into_iter()
            .map(|hit| SimilarTicket {
                ticket_id: hit.id.to_string(),
                score: hit.score,
                relation: if hit.score >= 0.90 {
                    "duplicate"
                } else {
                    "similar"
                }
                .to_owned(),
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
            .filter(|item| item.relation == "similar")
            .cloned()
            .collect();
        Ok(AssistPreviewResponse {
            response_template: response_template_for(&ticket.language, &prediction.topic_label),
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
        let priority = request
            .priority
            .clone()
            .unwrap_or_else(|| prediction.predicted_priority.clone());
        let service = request
            .service
            .clone()
            .unwrap_or_else(|| prediction.recommended_service.clone());
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
            "INSERT INTO operator_decisions (ticket_id, user_id, confirmed_topic_id, confirmed_service_id, confirmed_priority, decision, feedback) VALUES ($1, $2, $3, 'service_other', $4, $5, $6) RETURNING id",
        )
        .bind(ticket.id.parse::<i64>().map_err(|_| "invalid database ticket id".to_owned())?)
        .bind(user_id)
        .bind(&confirmed_topic)
        .bind(&priority)
        .bind(decision_value)
        .bind(json!({
            "note": request.note,
            "service": service,
            "action": action,
            "predicted_topic_id": prediction.topic_id,
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
        sqlx::query("UPDATE tickets SET status = 'TRIAGED', operator_confirmed_decision = $2, needs_review = false, updated_in_pulse_at = now() WHERE id = $1")
            .bind(ticket.id.parse::<i64>().map_err(|_| "invalid database ticket id".to_owned())?)
            .bind(&confirmed_payload)
            .execute(&mut *tx)
            .await
            .map_err(|error| format!("update ticket decision: {error}"))?;
        let cycle_id: i64 = sqlx::query_scalar(
            "INSERT INTO learning_cycles (cycle_id, state, candidate_dataset_version, candidate_model_version, min_feedback_count) VALUES ('vertical-slice', 'COLLECT', 'operator-feedback', 'pending', 1) ON CONFLICT (cycle_id) DO UPDATE SET updated_at = now() RETURNING id",
        )
        .fetch_one(&mut *tx)
        .await
        .map_err(|error| format!("ensure learning cycle: {error}"))?;
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
        tx.commit()
            .await
            .map_err(|error| format!("commit decision transaction: {error}"))?;
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
    Prediction {
        ticket_id: row.ticket_id.to_string(),
        model_version: row.model_version,
        topic_id: row.topic_id.unwrap_or_else(|| "unknown".to_owned()),
        topic_label: row.topic_label,
        confidence,
        confidence_state: if row.needs_review {
            "low"
        } else if confidence >= 0.85 {
            "high"
        } else {
            "medium"
        }
        .to_owned(),
        recommended_service: row.service_name,
        predicted_priority: row.priority,
        alternatives,
        created_at: row.created_at.to_rfc3339(),
    }
}

fn prediction_for_db(
    ticket_id: i64,
    classification: &MlClassificationWithModel,
    priority: &str,
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
        recommended_service: "Другая служба".to_owned(),
        predicted_priority: priority.to_owned(),
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
        "other" | "topic-other" => "unknown".to_owned(),
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
