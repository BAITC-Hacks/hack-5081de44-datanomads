use axum::{
    body::{to_bytes, Body},
    http::Request,
};
use pulse109_core::{app, AppState};
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
async fn corrected_decision_updates_template_without_rewriting_prediction() {
    let application = app(AppState::demo());
    let corrected = application
        .clone()
        .oneshot(
            Request::post("/api/v1/assist/ticket-001/correct")
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"topic_id":"TOPIC-ROADS","service":"Городская инфраструктура","priority":"medium"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(corrected.status(), 200);

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
async fn learning_cycle_counts_operator_decisions_but_not_freeform_feedback() {
    let application = app(AppState::demo());
    let created = application
        .clone()
        .oneshot(
            Request::post("/api/v1/learning")
                .header("x-pulse-role", "ML_REVIEWER")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(created.status(), 201);
    let created: serde_json::Value =
        serde_json::from_slice(&to_bytes(created.into_body(), usize::MAX).await.unwrap()).unwrap();
    let first_cycle = created["id"].as_str().unwrap();

    let note = application
        .clone()
        .oneshot(
            Request::post(format!("/api/v1/learning/{first_cycle}/feedback"))
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ticket_id":"ticket-001","decision":"confirm"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(note.status(), 201);
    let cycle = application
        .clone()
        .oneshot(
            Request::get(format!("/api/v1/learning/{first_cycle}"))
                .header("x-pulse-role", "ML_REVIEWER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let cycle: serde_json::Value =
        serde_json::from_slice(&to_bytes(cycle.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(cycle["feedback_count"], 0);

    let closed = application
        .clone()
        .oneshot(
            Request::post("/api/v1/learning/cycle/close")
                .header("x-pulse-role", "ML_REVIEWER")
                .header("content-type", "application/json")
                .body(Body::from(format!(r#"{{"cycle_id":"{first_cycle}"}}"#)))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(closed.status(), 202);
    let closed: serde_json::Value =
        serde_json::from_slice(&to_bytes(closed.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(closed["state"], "INSUFFICIENT_FEEDBACK");
    assert!(closed["job_id"].is_null());

    let late_note = application
        .clone()
        .oneshot(
            Request::post(format!("/api/v1/learning/{first_cycle}/feedback"))
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"ticket_id":"ticket-001","decision":"confirm"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(late_note.status(), 409);

    let created = application
        .clone()
        .oneshot(
            Request::post("/api/v1/learning")
                .header("x-pulse-role", "ML_REVIEWER")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(created.status(), 201);
    let created: serde_json::Value =
        serde_json::from_slice(&to_bytes(created.into_body(), usize::MAX).await.unwrap()).unwrap();
    let second_cycle = created["id"].as_str().unwrap();
    let decision = application
        .clone()
        .oneshot(
            Request::post("/api/v1/assist/ticket-001/confirm")
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(decision.status(), 200);
    let cycle = application
        .oneshot(
            Request::get(format!("/api/v1/learning/{second_cycle}"))
                .header("x-pulse-role", "ML_REVIEWER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let cycle: serde_json::Value =
        serde_json::from_slice(&to_bytes(cycle.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(cycle["feedback_count"], 1);
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
