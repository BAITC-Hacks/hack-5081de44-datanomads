use axum::{
    body::{to_bytes, Body},
    http::Request,
};
use pulse109_core::{app, AppState, ImportRequest};
use std::time::Duration;
use tokio_stream::StreamExt;
use tower::ServiceExt;

#[tokio::test]
async fn demo_api_supports_preview_and_manager_analytics() {
    let application = app(AppState::demo());
    let preview = application
        .clone()
        .oneshot(
            Request::post("/api/v1/assist/preview")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"text":"В городе нет воды"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(preview.status(), 200);

    let analytics = application
        .oneshot(
            Request::get("/api/v1/analytics")
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(analytics.status(), 200);
    let analytics: serde_json::Value =
        serde_json::from_slice(&to_bytes(analytics.into_body(), usize::MAX).await.unwrap())
            .unwrap();
    assert_eq!(
        analytics["runtime_metrics"]["operator_decision_time_samples"],
        1
    );
    assert!(
        analytics["runtime_metrics"]["operator_decision_time_minutes"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(analytics["runtime_metrics"]["similarity_usefulness"].is_null());
    assert!(analytics["runtime_metrics"]["duplicate_precision"].is_null());
}

#[tokio::test]
async fn learning_feedback_rejects_a_cycle_outside_collect() {
    let application = app(AppState::demo());
    let response = application
        .oneshot(
            Request::post("/api/v1/learning/cycle-001/feedback")
                .header("content-type", "application/json")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::from(
                    r#"{"ticket_id":"ticket-001","decision":"confirm"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 409);
}

#[tokio::test]
async fn operator_decision_reports_when_no_collect_cycle_is_open() {
    let application = app(AppState::demo());
    let response = application
        .oneshot(
            Request::post("/api/v1/assist/ticket-001/confirm")
                .header("content-type", "application/json")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 200);
    let body: serde_json::Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["learning_feedback_status"], "NO_ACTIVE_COLLECT_CYCLE");
    assert!(body["learning_feedback_cycle_id"].is_null());
}

#[tokio::test]
async fn assist_preview_preserves_language_states_and_correlated_context() {
    let application = app(AppState::demo());
    for (language, request_id, trace_id, partial, template_source, ticket_language) in [
        (
            "RU",
            Some("preview-ru"),
            Some("trace-ru"),
            false,
            "MANUAL_REQUIRED",
            "ru",
        ),
        (
            "KZ",
            Some("preview-kz"),
            Some("trace-kz"),
            false,
            "MANUAL_REQUIRED",
            "kk",
        ),
        (
            "MIXED",
            Some("preview-mixed"),
            Some("trace-mixed"),
            true,
            "UNAVAILABLE",
            "mixed",
        ),
        ("UNKNOWN", None, None, true, "UNAVAILABLE", "unknown"),
    ] {
        let mut request = Request::post("/api/v1/assist/preview")
            .header("content-type", "application/json")
            .header("x-pulse-role", "OPERATOR");
        if let Some(request_id) = request_id {
            request = request.header("x-request-id", request_id);
        }
        if let Some(trace_id) = trace_id {
            request = request.header("x-trace-id", trace_id);
        }
        let response = application
            .clone()
            .oneshot(
                request
                    .body(Body::from(format!(
                        r#"{{"text":"Проверить обращение вручную","language":"{language}"}}"#
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let response_request_id = response.headers()["x-request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let response_trace_id = response.headers()["x-trace-id"]
            .to_str()
            .unwrap()
            .to_owned();
        let preview: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();

        assert_eq!(preview["orchestration"]["language"], language);
        assert_eq!(preview["ticket"]["language"], ticket_language);
        assert_eq!(preview["orchestration"]["request_id"], response_request_id);
        assert_eq!(preview["orchestration"]["trace_id"], response_trace_id);
        assert_eq!(
            preview["orchestration"]["status"],
            if partial { "partial" } else { "complete" }
        );
        assert_eq!(preview["orchestration"]["needs_review"], true);
        assert_eq!(preview["response_template"]["source"], template_source);
        assert_eq!(preview["response_template"]["approved"], false);
        if partial {
            assert_eq!(preview["prediction"]["topic_id"], "UNKNOWN");
            assert_eq!(preview["prediction"]["confidence"].as_f64(), Some(0.0));
            assert_eq!(preview["ticket"]["priority"], "UNKNOWN");
        }
        assert!(preview["orchestration"]["latency_ms"].as_f64().unwrap() >= 0.0);
        if let Some(request_id) = request_id {
            assert_eq!(response_request_id, request_id);
        }
        if let Some(trace_id) = trace_id {
            assert_eq!(response_trace_id, trace_id);
        } else {
            assert_eq!(response_request_id, response_trace_id);
        }
    }

    let unsupported_language = application
        .clone()
        .oneshot(
            Request::post("/api/v1/assist/preview")
                .header("content-type", "application/json")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::from(r#"{"text":"Нет воды","language":"EN"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unsupported_language.status(), 400);

    let official_before = application
        .clone()
        .oneshot(
            Request::get("/api/v1/tickets/ticket-001")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let official_before: serde_json::Value = serde_json::from_slice(
        &to_bytes(official_before.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    let preview = application
        .clone()
        .oneshot(
            Request::post("/api/v1/assist/preview")
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"text":"Проверить обращение вручную","language":"UNKNOWN"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(preview.status(), 200);
    let official_after = application
        .oneshot(
            Request::get("/api/v1/tickets/ticket-001")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let official_after: serde_json::Value = serde_json::from_slice(
        &to_bytes(official_after.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(official_before["ticket"], official_after["ticket"]);
    assert_eq!(official_before["prediction"], official_after["prediction"]);
    assert_eq!(
        official_before["latest_decision"],
        official_after["latest_decision"]
    );
}

#[tokio::test]
async fn dataset_provenance_marks_demo_records_as_synthetic() {
    let response = app(AppState::demo())
        .oneshot(
            Request::get("/api/v1/datasets/provenance")
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let provenance: serde_json::Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert!(provenance["synthetic_ticket_count"].as_i64().unwrap() > 0);
    assert_eq!(provenance["real_ticket_count"], 0);
}

#[test]
fn import_request_requires_explicit_synthetic_provenance() {
    let request = serde_json::from_value::<ImportRequest>(serde_json::json!({
        "source_system": "example",
        "tickets": [],
        "quarantine": []
    }));
    assert!(request.is_err());
}

#[tokio::test]
async fn related_ticket_detail_returns_permitted_context_and_latest_operator_action() {
    let application = app(AppState::demo());
    let ticket = application
        .clone()
        .oneshot(
            Request::get("/api/v1/tickets/ticket-001")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ticket.status(), 200);
    let ticket: serde_json::Value =
        serde_json::from_slice(&to_bytes(ticket.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert!(!ticket["ticket"]["text"]
        .as_str()
        .unwrap_or_default()
        .is_empty());
    assert!(!ticket["ticket"]["topic_label"]
        .as_str()
        .unwrap_or_default()
        .is_empty());
    assert!(!ticket["ticket"]["region_name"]
        .as_str()
        .unwrap_or_default()
        .is_empty());
    assert!(!ticket["ticket"]["created_at"]
        .as_str()
        .unwrap_or_default()
        .is_empty());
    assert!(ticket["ticket"]["closed_at"].is_null());

    let corrected = application
        .clone()
        .oneshot(
            Request::post("/api/v1/assist/ticket-001/correct")
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"topic_id":"TOPIC-ROADS","service":"service_other","priority":"critical"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(corrected.status(), 200);

    let ticket = application
        .clone()
        .oneshot(
            Request::get("/api/v1/tickets/ticket-001")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(ticket.status(), 200);
    let ticket: serde_json::Value =
        serde_json::from_slice(&to_bytes(ticket.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(ticket["latest_decision"]["action"], "correct");
    assert_eq!(
        ticket["latest_decision"]["confirmed_topic_id"],
        "TOPIC-ROADS"
    );
    assert!(ticket["latest_decision"]["created_at"].as_str().is_some());

    let denied = application
        .oneshot(
            Request::get("/api/v1/tickets/ticket-001")
                .header("x-pulse-role", "ML_REVIEWER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied.status(), 403);
}

#[tokio::test]
async fn relation_feedback_keeps_rejected_relation_and_suggestion_snapshot() {
    let application = app(AppState::demo());
    let response = application
        .oneshot(
            Request::post("/api/v1/tickets/ticket-001/relation-feedback")
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"related_ticket_id":"ticket-002","relation":"DUPLICATE","decision":"REJECTED","suggestion":{"score":0.95,"threshold":0.9,"rule_version":"related-ticket-rules.v1","model_version":"embedder-test-v1","distance_metric":"Cosine"}}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 201);
    let feedback: serde_json::Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(feedback["feedback_type"], "relation:DUPLICATE:REJECTED");
    assert_eq!(feedback["suggestion"]["score"], 0.95);
    assert_eq!(feedback["suggestion"]["threshold"], 0.9);
    assert_eq!(
        feedback["suggestion"]["rule_version"],
        "related-ticket-rules.v1"
    );
    assert_eq!(feedback["suggestion"]["model_version"], "embedder-test-v1");
    assert_eq!(feedback["suggestion"]["distance_metric"], "Cosine");
}

#[tokio::test]
async fn corrected_decision_updates_template_without_rewriting_prediction() {
    let application = app(AppState::demo());
    let corrected = application
        .clone()
        .oneshot(
            Request::post("/api/v1/assist/ticket-001/correct")
                .header("x-pulse-role", "OPERATOR")
                .header("x-user-id", "operator-task-010")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"topic_id":"TOPIC-ROADS","service":"service_other","priority":"critical"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(corrected.status(), 200);
    let corrected: serde_json::Value =
        serde_json::from_slice(&to_bytes(corrected.into_body(), usize::MAX).await.unwrap())
            .unwrap();
    assert_eq!(corrected["prediction"]["topic_id"], "TOPIC-WATER");
    assert_eq!(corrected["decision"]["predicted_topic_id"], "TOPIC-WATER");
    assert_eq!(
        corrected["decision"]["model_version"],
        corrected["prediction"]["model_version"]
    );
    assert!(!corrected["decision"]["predicted_service"]
        .as_str()
        .unwrap_or_default()
        .is_empty());
    assert!(!corrected["decision"]["predicted_priority"]
        .as_str()
        .unwrap_or_default()
        .is_empty());
    assert_eq!(corrected["decision"]["user_id"], "operator-task-010");
    assert_eq!(
        corrected["ticket"]["updated_at"],
        corrected["decision"]["created_at"]
    );
    assert_eq!(corrected["decision"]["priority"], "critical");
    assert_eq!(corrected["decision"]["service"], "Другая служба");
    assert_eq!(
        corrected["decision"]["service_provenance"]["source"],
        "MANUAL"
    );
    assert_eq!(
        corrected["decision"]["priority_provenance"]["source"],
        "MANUAL"
    );
    assert_eq!(
        corrected["decision"]["priority_provenance"]["reason"],
        "Приоритет переопределён оператором"
    );

    let detail = application
        .clone()
        .oneshot(
            Request::get("/api/v1/tickets/ticket-001")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let detail: serde_json::Value =
        serde_json::from_slice(&to_bytes(detail.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(detail["ticket"]["topic_id"], "TOPIC-WATER");
    assert_eq!(detail["ticket"]["priority"], "high");
    assert_eq!(detail["ticket"]["status"], "open");
    assert_eq!(detail["prediction"]["topic_id"], "TOPIC-WATER");
    assert_eq!(
        detail["prediction"]["service_provenance"]["source"],
        "MANUAL"
    );
    assert_eq!(
        detail["prediction"]["priority_provenance"]["source"],
        "MANUAL"
    );
    assert_eq!(detail["latest_decision"]["priority"], "critical");
    assert_eq!(
        detail["latest_decision"]["confirmed_topic_id"],
        "TOPIC-ROADS"
    );
    assert_eq!(
        detail["latest_decision"]["confirmed_topic_label"],
        "Дороги и благоустройство"
    );
    assert_eq!(
        detail["latest_decision"]["model_version"],
        detail["prediction"]["model_version"]
    );
    assert_eq!(detail["latest_decision"]["service"], "Другая служба");
    assert_eq!(
        detail["latest_decision"]["priority_provenance"]["source"],
        "MANUAL"
    );

    let preview = application
        .oneshot(
            Request::post("/api/v1/assist/preview")
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"ticket_id":"ticket-001"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(preview.status(), 200);
    let preview: serde_json::Value =
        serde_json::from_slice(&to_bytes(preview.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(preview["response_template"]["id"], "manual");
    assert_eq!(preview["response_template"]["source"], "MANUAL_REQUIRED");
    assert_eq!(preview["response_template"]["body"], "");
    assert_eq!(preview["response_template"]["approved"], false);
    assert_eq!(preview["ticket"]["topic_id"], "TOPIC-WATER");
    assert_eq!(preview["ticket"]["priority"], "high");
    assert_eq!(preview["ticket"]["status"], "open");
    assert_eq!(preview["prediction"]["topic_id"], "TOPIC-WATER");
    let related_tickets = preview["similar_tickets"].as_array().unwrap();
    assert!(!related_tickets.is_empty());
    assert!(related_tickets
        .iter()
        .all(|candidate| candidate["topic_id"] == "TOPIC-ROADS"));
}

#[tokio::test]
async fn analytics_aggregates_operator_corrections_and_human_relation_feedback() {
    let application = app(AppState::demo());
    let corrected = application
        .clone()
        .oneshot(
            Request::post("/api/v1/assist/ticket-001/correct")
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"topic_id":"TOPIC-ROADS","service":"service_other","priority":"critical"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(corrected.status(), 200);

    for (related_ticket_id, decision) in [("ticket-002", "CONFIRMED"), ("ticket-003", "REJECTED")] {
        let body = serde_json::json!({
            "related_ticket_id": related_ticket_id,
            "relation": "DUPLICATE",
            "decision": decision,
        })
        .to_string();
        let feedback = application
            .clone()
            .oneshot(
                Request::post("/api/v1/tickets/ticket-001/relation-feedback")
                    .header("x-pulse-role", "OPERATOR")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(feedback.status(), 201);
    }

    let response = application
        .oneshot(
            Request::get("/api/v1/analytics")
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let analytics: serde_json::Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    let metrics = &analytics["runtime_metrics"];

    assert_eq!(metrics["operator_decision_time_samples"], 2);
    assert!(metrics["operator_decision_time_minutes"].as_f64().unwrap() > 0.0);
    assert_eq!(metrics["classification_decisions"], 2);
    assert_eq!(metrics["classification_corrections"], 1);
    assert_eq!(metrics["classification_correction_rate"], 0.5);
    assert_eq!(metrics["routing_decisions"], 2);
    assert_eq!(metrics["routing_corrections"], 1);
    assert_eq!(metrics["routing_correction_rate"], 0.5);
    assert_eq!(metrics["priority_decisions"], 2);
    assert_eq!(metrics["priority_corrections"], 1);
    assert_eq!(metrics["priority_correction_rate"], 0.5);
    assert_eq!(metrics["similarity_feedback_count"], 2);
    assert_eq!(metrics["similarity_usefulness"], 0.5);
    assert_eq!(metrics["duplicate_feedback_count"], 2);
    assert_eq!(metrics["duplicate_precision"], 0.5);
}

#[tokio::test]
async fn topic_correction_recomputes_confirmed_routing_without_mutating_source_ticket() {
    let response = app(AppState::demo())
        .oneshot(
            Request::post("/api/v1/assist/ticket-001/correct")
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"topic_id":"TOPIC-ROADS"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let response: serde_json::Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();

    assert_eq!(response["prediction"]["topic_id"], "TOPIC-WATER");
    assert_eq!(response["prediction"]["recommended_service"], "Водоканал");
    assert_eq!(response["prediction"]["predicted_priority"], "high");
    assert_eq!(response["decision"]["confirmed_topic_id"], "TOPIC-ROADS");
    assert_eq!(response["decision"]["service"], "Городская инфраструктура");
    assert_eq!(response["decision"]["priority"], "medium");
    assert_eq!(response["ticket"]["topic_id"], "TOPIC-WATER");
    assert_eq!(response["ticket"]["priority"], "high");
    assert_eq!(response["ticket"]["status"], "open");
}

#[tokio::test]
async fn operator_cannot_read_manager_analytics() {
    let response = app(AppState::demo())
        .oneshot(
            Request::get("/api/v1/analytics")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 403);
}

#[tokio::test]
async fn alert_events_stream_snapshot_and_changes() {
    let application = app(AppState::demo());
    let response = application
        .clone()
        .oneshot(
            Request::get("/api/v1/events")
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let mut stream = response.into_body().into_data_stream();
    let snapshot = tokio::time::timeout(Duration::from_secs(2), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(String::from_utf8_lossy(&snapshot).contains("event: alerts.snapshot"));

    let acknowledged = application
        .oneshot(
            Request::post("/api/v1/alerts/alert-001/ack")
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(acknowledged.status(), 200);
    let changed = tokio::time::timeout(Duration::from_secs(2), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(String::from_utf8_lossy(&changed).contains("event: alerts.changed"));
}

#[tokio::test]
async fn response_templates_require_explicit_approval_and_exact_confirmed_selectors() {
    let application = app(AppState::demo());
    let preview_request = || {
        Request::post("/api/v1/assist/preview")
            .header("x-pulse-role", "OPERATOR")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"ticket_id":"ticket-002"}"#))
            .unwrap()
    };
    let preview = application
        .clone()
        .oneshot(preview_request())
        .await
        .unwrap();
    let preview: serde_json::Value =
        serde_json::from_slice(&to_bytes(preview.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(preview["response_template"]["source"], "MANUAL_REQUIRED");
    assert_eq!(preview["response_template"]["body"], "");

    let create = |service_id: &str, body: &str| {
        Request::post("/api/v1/response-templates")
            .header("x-pulse-role", "MANAGER")
            .header("content-type", "application/json")
            .body(Body::from(format!(
                r#"{{"template_key":"roads-reply","language":"KZ","topic_id":"TOPIC-ROADS","service_id":"{service_id}","body":"{body}"}}"#
            )))
            .unwrap()
    };

    let wrong_service = application
        .clone()
        .oneshot(create(
            "Водоканал",
            "Для {{topic}} в {{region}} ответит {{service}}.",
        ))
        .await
        .unwrap();
    assert_eq!(wrong_service.status(), 200);
    let wrong_service: serde_json::Value = serde_json::from_slice(
        &to_bytes(wrong_service.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(wrong_service["approved"], false);
    assert_eq!(wrong_service["version"], 1);

    let operator_approval = application
        .clone()
        .oneshot(
            Request::post(format!(
                "/api/v1/response-templates/{}/approve",
                wrong_service["id"].as_str().unwrap()
            ))
            .header("x-pulse-role", "OPERATOR")
            .body(Body::empty())
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(operator_approval.status(), 403);

    let approve_wrong_service = application
        .clone()
        .oneshot(
            Request::post(format!(
                "/api/v1/response-templates/{}/approve",
                wrong_service["id"].as_str().unwrap()
            ))
            .header("x-pulse-role", "MANAGER")
            .body(Body::empty())
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(approve_wrong_service.status(), 200);
    let preview = application
        .clone()
        .oneshot(preview_request())
        .await
        .unwrap();
    let preview: serde_json::Value =
        serde_json::from_slice(&to_bytes(preview.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(preview["response_template"]["source"], "MANUAL_REQUIRED");

    let matching_service = application
        .clone()
        .oneshot(create(
            "Городская инфраструктура",
            "Для {{topic}} в {{region}} ответит {{service}}.",
        ))
        .await
        .unwrap();
    assert_eq!(matching_service.status(), 200);
    let matching_service: serde_json::Value = serde_json::from_slice(
        &to_bytes(matching_service.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(matching_service["approved"], false);
    assert_eq!(matching_service["version"], 2);

    let preview = application
        .clone()
        .oneshot(preview_request())
        .await
        .unwrap();
    let preview: serde_json::Value =
        serde_json::from_slice(&to_bytes(preview.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(preview["response_template"]["source"], "MANUAL_REQUIRED");

    let approve_matching = application
        .clone()
        .oneshot(
            Request::post(format!(
                "/api/v1/response-templates/{}/approve",
                matching_service["id"].as_str().unwrap()
            ))
            .header("x-pulse-role", "MANAGER")
            .header("x-user-id", "template-reviewer")
            .body(Body::empty())
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(approve_matching.status(), 200);
    let approved: serde_json::Value = serde_json::from_slice(
        &to_bytes(approve_matching.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(approved["approved"], true);
    assert_eq!(approved["approved_by"], "template-reviewer");
    assert!(approved["approved_at"].as_str().is_some());

    let preview = application
        .clone()
        .oneshot(preview_request())
        .await
        .unwrap();
    let preview: serde_json::Value =
        serde_json::from_slice(&to_bytes(preview.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(preview["response_template"]["source"], "APPROVED_TEMPLATE");
    assert_eq!(preview["response_template"]["approved"], true);
    assert_eq!(preview["response_template"]["version"], 2);
    let body = preview["response_template"]["body"].as_str().unwrap();
    assert!(body.contains("Дороги и благоустройство"));
    assert!(body.contains("Городская инфраструктура"));
    assert!(!body.contains("{{"));

    let edit_approved = application
        .clone()
        .oneshot(
            Request::put(format!(
                "/api/v1/response-templates/{}",
                matching_service["id"].as_str().unwrap()
            ))
            .header("x-pulse-role", "MANAGER")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"body":"new text"}"#))
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(edit_approved.status(), 409);

    let delete_approved = application
        .oneshot(
            Request::delete(format!(
                "/api/v1/response-templates/{}",
                matching_service["id"].as_str().unwrap()
            ))
            .header("x-pulse-role", "MANAGER")
            .body(Body::empty())
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(delete_approved.status(), 409);
}

#[tokio::test]
async fn response_template_import_and_pending_crud_are_manager_only() {
    let application = app(AppState::demo());
    let operator_list = application
        .clone()
        .oneshot(
            Request::get("/api/v1/response-templates")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(operator_list.status(), 403);

    let imported = application
        .clone()
        .oneshot(
            Request::post("/api/v1/response-templates/import")
                .header("x-pulse-role", "MANAGER")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"items":[{"template_key":"water-review","language":"KZ","topic_id":"TOPIC-WATER","service_id":"Водоканал","body":"{{topic}} · {{service}} · {{region}}"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(imported.status(), 200);
    let imported: serde_json::Value =
        serde_json::from_slice(&to_bytes(imported.into_body(), usize::MAX).await.unwrap()).unwrap();
    let record = &imported["items"][0];
    assert_eq!(record["approved"], false);
    assert_eq!(record["version"], 1);
    let template_id = record["id"].as_str().unwrap();

    let read = application
        .clone()
        .oneshot(
            Request::get(format!("/api/v1/response-templates/{template_id}"))
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(read.status(), 200);

    let updated = application
        .clone()
        .oneshot(
            Request::put(format!("/api/v1/response-templates/{template_id}"))
                .header("x-pulse-role", "MANAGER")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"body":"Updated {{topic}} in {{region}}."}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(updated.status(), 200);
    let updated: serde_json::Value =
        serde_json::from_slice(&to_bytes(updated.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(updated["body"], "Updated {{topic}} in {{region}}.");
    assert_eq!(updated["approved"], false);

    let deleted = application
        .clone()
        .oneshot(
            Request::delete(format!("/api/v1/response-templates/{template_id}"))
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(deleted.status(), 204);

    let invalid_variable = application
        .clone()
        .oneshot(
            Request::post("/api/v1/response-templates")
                .header("x-pulse-role", "MANAGER")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"template_key":"bad-variable","language":"RU","topic_id":"TOPIC-ROADS","service_id":"Городская инфраструктура","body":"{{ticket_text}}"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(invalid_variable.status(), 400);

    let approval_cannot_be_injected = application
        .oneshot(
            Request::post("/api/v1/response-templates")
                .header("x-pulse-role", "MANAGER")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"template_key":"fake-approved","language":"RU","topic_id":"TOPIC-ROADS","service_id":"Городская инфраструктура","body":"Reply","approved":true}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(approval_cannot_be_injected.status(), 422);
}
