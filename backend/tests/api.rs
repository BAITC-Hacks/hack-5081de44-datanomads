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
            "MANUAL_DEMO",
            "ru",
        ),
        (
            "KZ",
            Some("preview-kz"),
            Some("trace-kz"),
            false,
            "MANUAL_DEMO",
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
async fn corrected_decision_updates_template_without_rewriting_prediction() {
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
    let corrected: serde_json::Value =
        serde_json::from_slice(&to_bytes(corrected.into_body(), usize::MAX).await.unwrap())
            .unwrap();
    assert_eq!(corrected["prediction"]["topic_id"], "TOPIC-WATER");
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
    assert_eq!(detail["ticket"]["topic_id"], "TOPIC-ROADS");
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
    assert_eq!(
        preview["response_template"]["id"],
        "template-ru-topic-roads"
    );
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
