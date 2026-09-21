use axum::{body::Body, http::Request};
use pulse109_core::{app, AppState};
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
