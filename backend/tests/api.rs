use axum::{body::Body, http::Request};
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
