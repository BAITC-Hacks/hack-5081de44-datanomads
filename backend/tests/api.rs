use axum::{
    body::{to_bytes, Body},
    http::Request,
};
use pulse109_core::{app, AppState, ImportRequest};
use sha2::{Digest, Sha256};
use std::time::Duration;
use tokio_stream::StreamExt;
use tower::ServiceExt;

#[tokio::test]
async fn cors_only_allows_configured_origins() {
    let application = app(AppState::demo());
    let preflight = application
        .clone()
        .oneshot(
            Request::options("/api/v1/tickets")
                .header("origin", "http://localhost:8080")
                .header("access-control-request-method", "GET")
                .header("access-control-request-headers", "x-pulse-role")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(preflight.status(), 200);
    assert_eq!(
        preflight
            .headers()
            .get("access-control-allow-origin")
            .unwrap(),
        "http://localhost:8080"
    );

    let denied = application
        .oneshot(
            Request::get("/api/v1/tickets")
                .header("origin", "https://evil.example")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert!(denied
        .headers()
        .get("access-control-allow-origin")
        .is_none());

    let mut cors_disabled_state = AppState::demo();
    cors_disabled_state.config.cors_allowed_origins.clear();
    let cors_disabled = app(cors_disabled_state)
        .oneshot(
            Request::get("/api/v1/tickets")
                .header("origin", "http://localhost:8080")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(cors_disabled
        .headers()
        .get("access-control-allow-origin")
        .is_none());
}

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
async fn error_messages_do_not_echo_untrusted_values() {
    let response = app(AppState::demo())
        .oneshot(
            Request::post("/api/v1/tickets")
                .header("content-type", "application/json")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::from(
                    r#"{"text":"safe request","region_id":"PII_SENTINEL_029"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), 400);
    let body: serde_json::Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["message"], "invalid request");
    assert!(!body.to_string().contains("PII_SENTINEL_029"));
}

#[tokio::test]
async fn drift_evidence_is_deduplicated_and_human_reviewed() {
    let application = app(AppState::demo());
    let evidence = serde_json::json!({
        "schema_version": "drift-evidence.v1",
        "evidence_id": "drift-test-001",
        "model_version": "classifier-demo-2026-09-001",
        "detector_version": "detector-test.v1",
        "metric_name": "topic_distribution_js_divergence",
        "baseline_window_start": "2026-09-01T00:00:00Z",
        "baseline_window_end": "2026-09-08T00:00:00Z",
        "observed_window_start": "2026-09-08T00:00:00Z",
        "observed_window_end": "2026-09-15T00:00:00Z",
        "baseline_value": 0.04,
        "observed_value": 0.12,
        "drift_score": 0.08,
        "threshold": 0.05,
        "sample_count": 180,
        "minimum_sample_count": 100,
        "synthetic": false
    });
    let invalid = serde_json::json!({
        "schema_version": "drift-evidence.v1",
        "evidence_id": "drift-test-invalid",
        "model_version": "classifier-demo-2026-09-001",
        "detector_version": "detector-test.v1",
        "metric_name": "topic_distribution_js_divergence",
        "baseline_window_start": "2026-09-01T00:00:00Z",
        "baseline_window_end": "2026-09-08T00:00:00Z",
        "observed_window_start": "2026-09-08T00:00:00Z",
        "observed_window_end": "2026-09-15T00:00:00Z",
        "baseline_value": 0.04,
        "observed_value": 0.12,
        "drift_score": 0.04,
        "threshold": 0.05,
        "sample_count": 180,
        "minimum_sample_count": 100,
        "synthetic": false
    });
    let invalid_response = application
        .clone()
        .oneshot(
            Request::post("/api/v1/learning/drift-triggers")
                .header("x-pulse-role", "ML_SERVICE")
                .header("content-type", "application/json")
                .body(Body::from(invalid.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(invalid_response.status(), 400);

    let created = application
        .clone()
        .oneshot(
            Request::post("/api/v1/learning/drift-triggers")
                .header("x-pulse-role", "ML_SERVICE")
                .header("x-user-id", "data-ml")
                .header("content-type", "application/json")
                .body(Body::from(evidence.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(created.status(), 201);
    let created: serde_json::Value =
        serde_json::from_slice(&to_bytes(created.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(created["state"], "PENDING_REVIEW");
    assert_eq!(created["created_by"], "data-ml");

    let duplicate = application
        .clone()
        .oneshot(
            Request::post("/api/v1/learning/drift-triggers")
                .header("x-pulse-role", "ML_SERVICE")
                .header("content-type", "application/json")
                .body(Body::from(evidence.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(duplicate.status(), 200);

    let mut conflicting_evidence = evidence.clone();
    conflicting_evidence["observed_value"] = serde_json::json!(0.13);
    let conflicting_duplicate = application
        .clone()
        .oneshot(
            Request::post("/api/v1/learning/drift-triggers")
                .header("x-pulse-role", "ML_SERVICE")
                .header("content-type", "application/json")
                .body(Body::from(conflicting_evidence.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(conflicting_duplicate.status(), 409);

    let manager_list = application
        .clone()
        .oneshot(
            Request::get("/api/v1/learning/drift-triggers")
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(manager_list.status(), 200);
    let page: serde_json::Value = serde_json::from_slice(
        &to_bytes(manager_list.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(page["items"].as_array().unwrap().len(), 1);
    assert_eq!(page["can_review"], false);

    let forbidden = application
        .clone()
        .oneshot(
            Request::post("/api/v1/learning/drift-triggers/drift-test-001/review")
                .header("x-pulse-role", "MANAGER")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"decision":"DISMISS"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(forbidden.status(), 403);

    let dismissed = application
        .oneshot(
            Request::post("/api/v1/learning/drift-triggers/drift-test-001/review")
                .header("x-pulse-role", "ML_REVIEWER")
                .header("x-user-id", "reviewer-1")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"decision":"DISMISS"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(dismissed.status(), 200);
    let dismissed: serde_json::Value =
        serde_json::from_slice(&to_bytes(dismissed.into_body(), usize::MAX).await.unwrap())
            .unwrap();
    assert_eq!(dismissed["state"], "DISMISSED");
    assert_eq!(dismissed["reviewed_by"], "reviewer-1");
    assert_eq!(dismissed["learning_cycle_id"], serde_json::Value::Null);
}

#[tokio::test]
async fn manager_analytics_drilldown_omits_source_text() {
    let application = app(AppState::demo());
    let created = application
        .clone()
        .oneshot(
            Request::post("/api/v1/tickets")
                .header("content-type", "application/json")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::from(
                    r#"{"text":"PII_SENTINEL_029 source text","region_id":"R01"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(created.status(), 201);

    let response = application
        .oneshot(
            Request::get("/api/v1/analytics/drilldown?dimension=overview&value=all&range=30d")
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: serde_json::Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    let encoded = body.to_string();
    assert!(!encoded.contains("PII_SENTINEL_029"));
    assert!(body["items"].as_array().unwrap().iter().all(|item| {
        item.get("text").is_none()
            && item.get("external_ref").is_none()
            && item.get("address").is_none()
            && item.get("attachments").is_none()
    }));
}

#[tokio::test]
async fn demo_analytics_uses_one_filtered_slice_for_comparison_and_drilldown() {
    let application = app(AppState::demo());
    let analytics = application
        .clone()
        .oneshot(
            Request::get("/api/v1/analytics?range=7d")
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
    assert_eq!(analytics["source"], "deterministic-demo");
    assert_eq!(analytics["overview"]["total_tickets"], 4);
    assert_eq!(analytics["overview"]["previous_total_tickets"], 8);
    assert_eq!(analytics["overview"]["change_abs"], -4);
    assert_eq!(analytics["overview"]["change_pct"], -50.0);
    assert_eq!(analytics["by_region"].as_array().unwrap().len(), 20);
    for dimension in ["by_region", "by_topic", "time_series"] {
        let total = analytics[dimension]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["tickets"].as_u64().unwrap())
            .sum::<u64>();
        assert_eq!(total, 4, "{dimension} should match the overview slice");
    }

    let service = "service_id=%D0%A6%D0%B8%D1%84%D1%80%D0%BE%D0%B2%D0%BE%D0%B9+%D0%B0%D0%BA%D0%B8%D0%BC%D0%B0%D1%82";
    let service_only = application
        .clone()
        .oneshot(
            Request::get(format!("/api/v1/analytics?range=7d&{service}"))
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(service_only.status(), 200);
    let service_only: serde_json::Value = serde_json::from_slice(
        &to_bytes(service_only.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(service_only["overview"]["total_tickets"], 1);

    let rejected_status = application
        .clone()
        .oneshot(
            Request::get(format!("/api/v1/analytics?range=7d&{service}&status=OPEN"))
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rejected_status.status(), 200);
    let rejected_status: serde_json::Value = serde_json::from_slice(
        &to_bytes(rejected_status.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(rejected_status["overview"]["total_tickets"], 0);

    let filters = format!("range=7d&region_id=R10&topic_id=TOPIC-DIGITAL&{service}&status=TRIAGED");
    let selected = application
        .clone()
        .oneshot(
            Request::get(format!("/api/v1/analytics?{filters}"))
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(selected.status(), 200);
    let selected: serde_json::Value =
        serde_json::from_slice(&to_bytes(selected.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(selected["overview"]["total_tickets"], 1);
    assert_eq!(selected["by_region"][0]["id"], "R10");
    assert_eq!(selected["by_region"][0]["tickets"], 1);
    assert_eq!(selected["by_topic"][0]["id"], "TOPIC-DIGITAL");
    assert_eq!(selected["by_topic"][0]["tickets"], 1);

    let drilldown = application
        .oneshot(
            Request::get(format!(
                "/api/v1/analytics/drilldown?{filters}&dimension=region&value=R10"
            ))
            .header("x-pulse-role", "MANAGER")
            .body(Body::empty())
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(drilldown.status(), 200);
    let drilldown: serde_json::Value =
        serde_json::from_slice(&to_bytes(drilldown.into_body(), usize::MAX).await.unwrap())
            .unwrap();
    assert_eq!(drilldown["total"], 1);
    assert_eq!(drilldown["items"][0]["id"], "ticket-010");
}

#[tokio::test]
async fn reports_exports_and_forecast_use_the_selected_filter_slice() {
    let application = app(AppState::demo());
    let service = "service_id=%D0%A6%D0%B8%D1%84%D1%80%D0%BE%D0%B2%D0%BE%D0%B9+%D0%B0%D0%BA%D0%B8%D0%BC%D0%B0%D1%82";
    let filters = format!("range=7d&region_id=R10&topic_id=TOPIC-DIGITAL&{service}&status=TRIAGED");

    let reports = application
        .clone()
        .oneshot(
            Request::get(format!("/api/v1/reports?{filters}"))
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(reports.status(), 200);
    let reports: serde_json::Value =
        serde_json::from_slice(&to_bytes(reports.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(reports["source"], "deterministic-demo");
    assert_eq!(reports["slice"]["filters"]["range"], "7d");
    assert_eq!(reports["slice"]["filters"]["region_id"], "R10");
    assert_eq!(reports["slice"]["filters"]["topic_id"], "TOPIC-DIGITAL");
    assert_eq!(reports["slice"]["filters"]["status"], "TRIAGED");
    assert_eq!(
        reports["slice"]["analytics"]["overview"]["total_tickets"],
        1
    );
    assert_eq!(
        reports["slice"]["analytics"]["overview"]["previous_total_tickets"],
        0
    );
    assert_eq!(reports["slice"]["analytics"]["by_topic"][0]["tickets"], 1);
    assert_eq!(reports["slice"]["forecast"]["status"], "DEMO_ONLY");
    assert_eq!(
        reports["slice"]["forecast"]["capacity_assessment"]["status"],
        "DATA_UNAVAILABLE"
    );

    let pdf = application
        .clone()
        .oneshot(
            Request::get(format!("/api/v1/analytics/export.pdf?{filters}"))
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(pdf.status(), 200);
    let pdf = to_bytes(pdf.into_body(), usize::MAX).await.unwrap();
    assert!(pdf.starts_with(b"%PDF-"));
    assert!(pdf.len() > 1_000);
    assert!(pdf[pdf.len().saturating_sub(1_024)..]
        .windows(5)
        .any(|window| window == b"%%EOF"));

    let xlsx = application
        .clone()
        .oneshot(
            Request::get(format!("/api/v1/analytics/export.xlsx?{filters}"))
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(xlsx.status(), 200);
    let xlsx = to_bytes(xlsx.into_body(), usize::MAX).await.unwrap();
    let xlsx = String::from_utf8_lossy(&xlsx);
    assert!(xlsx.contains("topic=TOPIC-DIGITAL"));
    assert!(xlsx.contains("<t>period</t>"));
    assert!(xlsx.contains("<t>region_id</t>"));
    assert!(xlsx.contains("<t>topic_id</t>"));
    assert!(xlsx.contains("<t>share_pct</t>"));
    assert!(xlsx.contains("<t>change_abs</t>"));
    assert!(xlsx.contains("<t>generated_at</t>"));
    assert!(xlsx.contains("<t>alert_type</t>"));
    assert!(xlsx.contains("<t>TOPIC-DIGITAL</t>"));
    assert!(xlsx.contains("<v>100</v>"));
    assert!(!xlsx.contains("ticket-010"));

    let forecast_filters = format!("region_id=R10&topic_id=TOPIC-DIGITAL&{service}&status=TRIAGED");
    let forecast = application
        .clone()
        .oneshot(
            Request::get(format!("/api/v1/forecast?horizon=30&{forecast_filters}"))
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(forecast.status(), 200);
    let forecast: serde_json::Value =
        serde_json::from_slice(&to_bytes(forecast.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(forecast["source"], "deterministic-demo");
    assert_eq!(forecast["status"], "INSUFFICIENT_HISTORY");
    assert_eq!(forecast["insufficient_history"], true);
    assert_eq!(
        forecast["capacity_assessment"]["status"],
        "DATA_UNAVAILABLE"
    );
    assert_eq!(
        forecast["capacity_assessment"]["missing_inputs"],
        serde_json::json!([
            "STAFFING",
            "HANDLING_TIME_OR_THROUGHPUT",
            "SCHEDULE",
            "SERVICE_LEVEL_TARGET_OR_SLA"
        ])
    );
    assert!(forecast["capacity_assessment"]
        .get("capacity_risk")
        .is_none());
    assert_eq!(forecast["history"].as_array().unwrap().len(), 367);
    assert_eq!(
        forecast["history"]
            .as_array()
            .unwrap()
            .iter()
            .map(|point| point["tickets"].as_u64().unwrap())
            .sum::<u64>(),
        1
    );
    assert_eq!(forecast["forecast_start"], serde_json::Value::Null);
    assert!(forecast["points"].as_array().unwrap().is_empty());
    assert!(forecast["expected_peaks"].as_array().unwrap().is_empty());

    let no_forecast_filters = format!("region_id=R10&topic_id=TOPIC-DIGITAL&{service}&status=OPEN");
    let no_forecast = application
        .oneshot(
            Request::get(format!("/api/v1/forecast?horizon=30&{no_forecast_filters}"))
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(no_forecast.status(), 200);
    let no_forecast: serde_json::Value =
        serde_json::from_slice(&to_bytes(no_forecast.into_body(), usize::MAX).await.unwrap())
            .unwrap();
    assert_eq!(no_forecast["status"], "INSUFFICIENT_HISTORY");
    assert!(no_forecast["points"].as_array().unwrap().is_empty());
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
async fn viewing_candidate_evaluation_does_not_close_its_window() {
    let application = app(AppState::demo());
    let evaluation = application
        .clone()
        .oneshot(
            Request::get("/api/v1/learning/candidate/evaluation")
                .header("x-pulse-role", "ML_REVIEWER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(evaluation.status(), 200);
    let evaluation: serde_json::Value =
        serde_json::from_slice(&to_bytes(evaluation.into_body(), usize::MAX).await.unwrap())
            .unwrap();
    assert_eq!(evaluation["status"], "PENDING");
    assert_eq!(evaluation["cycle_id"], "cycle-001");
    assert_eq!(evaluation["decision"], "INSUFFICIENT_EVIDENCE");
    assert_eq!(evaluation["synthetic"], true);
    assert_eq!(evaluation["shadow_evaluation"]["blind_ab"], "DISABLED");
    assert_eq!(
        evaluation["candidate_comparisons"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        evaluation["candidate_comparisons"][0]["candidate_model_version"],
        evaluation["candidate_model_version"]
    );
    assert_eq!(evaluation["evaluation_set"]["cycle_id"], "cycle-001");

    let generic_promotion = application
        .clone()
        .oneshot(
            Request::post("/api/v1/models/classifier-candidate-2026-09-001/promote")
                .header("x-pulse-role", "ML_REVIEWER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(generic_promotion.status(), 409);

    let promotion = application
        .clone()
        .oneshot(
            Request::post("/api/v1/learning/candidate/promote")
                .header("content-type", "application/json")
                .header("x-pulse-role", "ML_REVIEWER")
                .body(Body::from(r#"{"note":"must wait for evaluation close"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(promotion.status(), 409);

    let close = application
        .clone()
        .oneshot(
            Request::post("/api/v1/learning/cycle/close")
                .header("content-type", "application/json")
                .header("x-pulse-role", "ML_REVIEWER")
                .body(Body::from(r#"{"cycle_id":"cycle-001"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(close.status(), 202);
    let close: serde_json::Value =
        serde_json::from_slice(&to_bytes(close.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(close["state"], "DECISION");
    assert_eq!(close["production_model_unchanged"], true);
    assert_eq!(close["cycle"]["state"], "DECISION");

    let completed_evaluation = application
        .clone()
        .oneshot(
            Request::get("/api/v1/learning/candidate/evaluation")
                .header("x-pulse-role", "ML_REVIEWER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(completed_evaluation.status(), 200);
    let completed_evaluation: serde_json::Value = serde_json::from_slice(
        &to_bytes(completed_evaluation.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(completed_evaluation["status"], "FAILED");
    assert_eq!(completed_evaluation["decision"], "INSUFFICIENT_EVIDENCE");
    assert_eq!(completed_evaluation["synthetic"], true);

    let promotion_after_close = application
        .oneshot(
            Request::post("/api/v1/learning/candidate/promote")
                .header("content-type", "application/json")
                .header("x-pulse-role", "ML_REVIEWER")
                .body(Body::from(
                    r#"{"note":"in-memory evaluation must fail closed"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(promotion_after_close.status(), 409);
}

#[tokio::test]
async fn learning_cycle_rejection_requires_a_reviewer_and_keeps_production_unchanged() {
    let application = app(AppState::demo());
    let forbidden = application
        .clone()
        .oneshot(
            Request::post("/api/v1/learning/candidate/reject")
                .header("content-type", "application/json")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::from(r#"{"note":"not authorized"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(forbidden.status(), 403);

    let before = application
        .clone()
        .oneshot(
            Request::get("/api/v1/learning")
                .header("x-pulse-role", "ML_REVIEWER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(before.status(), 200);
    let before: serde_json::Value =
        serde_json::from_slice(&to_bytes(before.into_body(), usize::MAX).await.unwrap()).unwrap();
    let production_before = before["production_model"]["id"].as_str().unwrap();

    let close = application
        .clone()
        .oneshot(
            Request::post("/api/v1/learning/cycle/close")
                .header("content-type", "application/json")
                .header("x-pulse-role", "ML_REVIEWER")
                .body(Body::from(r#"{"cycle_id":"cycle-001"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(close.status(), 202);

    let reject = application
        .clone()
        .oneshot(
            Request::post("/api/v1/learning/candidate/reject")
                .header("content-type", "application/json")
                .header("x-pulse-role", "ML_REVIEWER")
                .body(Body::from(r#"{"note":"reviewed candidate evidence"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(reject.status(), 200);
    let rejected: serde_json::Value =
        serde_json::from_slice(&to_bytes(reject.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(rejected["state"], "REJECTED");
    assert_eq!(rejected["decision_note"], "reviewed candidate evidence");

    let candidate_id = rejected["candidate_model_version"].as_str().unwrap();
    let candidate = application
        .clone()
        .oneshot(
            Request::get(format!("/api/v1/models/{candidate_id}"))
                .header("x-pulse-role", "ML_REVIEWER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(candidate.status(), 200);
    let candidate: serde_json::Value =
        serde_json::from_slice(&to_bytes(candidate.into_body(), usize::MAX).await.unwrap())
            .unwrap();
    assert_eq!(candidate["status"], "rejected");

    let after = application
        .oneshot(
            Request::get("/api/v1/learning")
                .header("x-pulse-role", "ML_REVIEWER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(after.status(), 200);
    let after: serde_json::Value =
        serde_json::from_slice(&to_bytes(after.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(after["production_model"]["id"], production_before);
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
        let trace_source = trace_id.unwrap_or(response_request_id.as_str());
        let expected_trace_id = format!("trace-{:x}", Sha256::digest(trace_source.as_bytes()));
        assert_eq!(response_trace_id, expected_trace_id);
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
async fn routing_feedback_is_labeled_simulation_and_preserves_operator_decision() {
    let application = app(AppState::demo());
    let before = application
        .clone()
        .oneshot(
            Request::get("/api/v1/tickets/ticket-002")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(before.status(), 200);
    let before: serde_json::Value =
        serde_json::from_slice(&to_bytes(before.into_body(), usize::MAX).await.unwrap()).unwrap();

    let accepted = application
        .clone()
        .oneshot(
            Request::post("/api/v1/tickets/ticket-002/routing-feedback")
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"service_feedback":"ACCEPTED"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(accepted.status(), 201);
    let accepted: serde_json::Value =
        serde_json::from_slice(&to_bytes(accepted.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(accepted["service_feedback"], "ACCEPTED");
    assert!(accepted["corrected_target_service"].is_null());
    assert_eq!(accepted["source_system"], "DEMO_SIMULATION");

    let response = application
        .clone()
        .oneshot(
            Request::post("/api/v1/tickets/ticket-002/routing-feedback")
                .header("x-pulse-role", "OPERATOR")
                .header("x-user-id", "routing-feedback-reviewer")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"service_feedback":"CORRECTED","corrected_target_service":"Управление транспорта"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 201);
    let record: serde_json::Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(record["service_feedback"], "CORRECTED");
    assert_eq!(
        record["original_route_recommendation"],
        before["latest_decision"]["predicted_service"]
    );
    assert_eq!(
        record["operator_confirmed_route"],
        before["latest_decision"]["service"]
    );
    assert_eq!(
        record["operator_decision_id"],
        before["latest_decision"]["id"]
    );
    assert_eq!(record["corrected_target_service"], "Управление транспорта");
    assert_eq!(record["actor_user_id"], "routing-feedback-reviewer");
    assert_eq!(record["source_system"], "DEMO_SIMULATION");
    assert_eq!(record["evaluation_status"], "PENDING_OFFLINE_REVIEW");
    assert!(record["created_at"].as_str().is_some());
    assert!(record.get("text").is_none());

    let history = application
        .clone()
        .oneshot(
            Request::get("/api/v1/tickets/ticket-002/routing-feedback")
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(history.status(), 200);
    let history: serde_json::Value =
        serde_json::from_slice(&to_bytes(history.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(history["items"].as_array().unwrap().len(), 2);
    assert!(history["items"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["id"] == record["id"]));
    assert!(history["items"]
        .as_array()
        .unwrap()
        .iter()
        .any(|item| item["id"] == accepted["id"]));

    let after = application
        .oneshot(
            Request::get("/api/v1/tickets/ticket-002")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let after: serde_json::Value =
        serde_json::from_slice(&to_bytes(after.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(
        after["prediction"]["recommended_service"],
        before["prediction"]["recommended_service"]
    );
    assert_eq!(
        after["latest_decision"]["service"],
        before["latest_decision"]["service"]
    );
}

#[tokio::test]
async fn routing_feedback_requires_a_confirmed_route_and_rejects_invalid_targets() {
    let application = app(AppState::demo());
    let missing_decision = application
        .clone()
        .oneshot(
            Request::post("/api/v1/tickets/ticket-001/routing-feedback")
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"service_feedback":"ACCEPTED"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing_decision.status(), 409);

    let missing_target = application
        .clone()
        .oneshot(
            Request::post("/api/v1/tickets/ticket-002/routing-feedback")
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"service_feedback":"CORRECTED"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing_target.status(), 400);

    let unexpected_target = application
        .clone()
        .oneshot(
            Request::post("/api/v1/tickets/ticket-002/routing-feedback")
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"service_feedback":"ACCEPTED","corrected_target_service":"Управление транспорта"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unexpected_target.status(), 400);

    let unchanged_target = application
        .clone()
        .oneshot(
            Request::post("/api/v1/tickets/ticket-002/routing-feedback")
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"service_feedback":"CORRECTED","corrected_target_service":"Городская инфраструктура"}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unchanged_target.status(), 400);

    let denied = application
        .oneshot(
            Request::get("/api/v1/tickets/ticket-002/routing-feedback")
                .header("x-pulse-role", "ML_REVIEWER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied.status(), 403);
}

#[tokio::test]
async fn outcome_verification_is_separate_from_official_ticket_status() {
    let application = app(AppState::demo());
    let initial = application
        .clone()
        .oneshot(
            Request::get("/api/v1/tickets/ticket-002/outcome-verification")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(initial.status(), 200);
    let initial: serde_json::Value =
        serde_json::from_slice(&to_bytes(initial.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(initial["state"], "UNKNOWN");
    assert_eq!(initial["official_ticket_status"], "open");
    assert!(initial["latest"].is_null());
    assert_eq!(initial["history"].as_array().unwrap().len(), 0);

    for (index, state) in ["VERIFIED", "PARTIAL", "DISPUTED"].into_iter().enumerate() {
        let response = application
            .clone()
            .oneshot(
                Request::post("/api/v1/tickets/ticket-002/outcome-verification")
                    .header("x-pulse-role", "OPERATOR")
                    .header("x-user-id", "outcome-reviewer")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({ "state": state }).to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 201);
        let snapshot: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(snapshot["state"], state);
        assert_eq!(snapshot["latest"]["state"], state);
        assert_eq!(snapshot["latest"]["source_system"], "DEMO_SIMULATION");
        assert_eq!(snapshot["latest"]["channel"], "OPERATOR_PANEL_SIMULATION");
        assert_eq!(snapshot["latest"]["actor_user_id"], "outcome-reviewer");
        assert!(snapshot["latest"]["created_at"].as_str().is_some());
        assert_eq!(snapshot["history"].as_array().unwrap().len(), index + 1);
        assert!(snapshot.get("original_text").is_none());
    }

    let unchanged_ticket = application
        .clone()
        .oneshot(
            Request::get("/api/v1/tickets/ticket-002")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let unchanged_ticket: serde_json::Value = serde_json::from_slice(
        &to_bytes(unchanged_ticket.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(unchanged_ticket["ticket"]["status"], "open");

    let unknown = application
        .clone()
        .oneshot(
            Request::post("/api/v1/tickets/ticket-002/outcome-verification")
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"state":"UNKNOWN"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(unknown.status(), 400);

    let denied = application
        .oneshot(
            Request::post("/api/v1/tickets/ticket-002/outcome-verification")
                .header("x-pulse-role", "ML_REVIEWER")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"state":"VERIFIED"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied.status(), 403);
}

#[tokio::test]
async fn context_handoff_package_preserves_sources_and_requires_a_confirmed_route() {
    let application = app(AppState::demo());
    let before = application
        .clone()
        .oneshot(
            Request::get("/api/v1/tickets/ticket-002")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(before.status(), 200);
    let before: serde_json::Value =
        serde_json::from_slice(&to_bytes(before.into_body(), usize::MAX).await.unwrap()).unwrap();

    let response = application
        .clone()
        .oneshot(
            Request::get("/api/v1/tickets/ticket-002/handoff-package")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let package: serde_json::Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(package["package_version"], "context-handoff.v1");
    assert_eq!(package["ticket_id"], before["ticket"]["id"]);
    assert_eq!(package["what_happened"], before["ticket"]["text"]);
    assert_eq!(package["where"]["region_id"], before["ticket"]["region_id"]);
    assert_eq!(
        package["when_or_since"]["received_at"],
        before["ticket"]["created_at"]
    );
    assert!(package["when_or_since"]["reported_since"].is_null());
    assert!(package["scale"].is_null());
    assert!(package["unknown_facts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|fact| fact.as_str().unwrap().contains("Масштаб")));
    assert_eq!(
        package["route"]["recommended_service"],
        before["latest_decision"]["predicted_service"]
    );
    assert_eq!(
        package["route"]["confirmed_service"],
        before["latest_decision"]["service"]
    );
    assert_eq!(
        package["route"]["operator_decision_id"],
        before["latest_decision"]["id"]
    );
    assert!(package["confirmed_facts"]
        .as_array()
        .unwrap()
        .iter()
        .any(|fact| fact["label"] == "Служба, подтверждённая оператором"));
    assert!(package["evidence_references"]
        .as_array()
        .unwrap()
        .iter()
        .any(|reference| reference["field"] == "original_text"));
    assert!(!package["evidence_references"]
        .to_string()
        .contains(before["ticket"]["text"].as_str().unwrap()));

    let after = application
        .clone()
        .oneshot(
            Request::get("/api/v1/tickets/ticket-002")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let after: serde_json::Value =
        serde_json::from_slice(&to_bytes(after.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(after["ticket"]["text"], before["ticket"]["text"]);
    assert_eq!(after["latest_decision"], before["latest_decision"]);

    let missing_decision = application
        .clone()
        .oneshot(
            Request::get("/api/v1/tickets/ticket-001/handoff-package")
                .header("x-pulse-role", "OPERATOR")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(missing_decision.status(), 409);

    let denied = application
        .oneshot(
            Request::get("/api/v1/tickets/ticket-002/handoff-package")
                .header("x-pulse-role", "ML_REVIEWER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(denied.status(), 403);
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
    // Similarity alone is not recurrence without structured object and close-time evidence.
    assert!(preview["repeat_candidates"].as_array().unwrap().is_empty());
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
async fn learning_cycle_counts_operator_decisions_but_not_freeform_feedback() {
    // The in-memory demo starts with an active EVALUATE cycle. Reject that
    // fixture cycle through the reviewer API before opening the test's own
    // COLLECT cycles, so the single-active-cycle guard remains exercised.
    let application = app(AppState::demo());
    let seeded_cycle = application
        .clone()
        .oneshot(
            Request::post("/api/v1/learning/cycle-001/reject")
                .header("x-pulse-role", "ML_REVIEWER")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"note":"test fixture reset"}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(seeded_cycle.status(), 200);
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
async fn manager_cannot_access_operator_ticket_or_model_workflows() {
    let application = app(AppState::demo());
    let requests = [
        Request::get("/api/v1/tickets")
            .header("x-pulse-role", "MANAGER")
            .body(Body::empty())
            .unwrap(),
        Request::post("/api/v1/tickets")
            .header("x-pulse-role", "MANAGER")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"text":"synthetic manager role probe"}"#))
            .unwrap(),
        Request::get("/api/v1/tickets/demo-0001")
            .header("x-pulse-role", "MANAGER")
            .body(Body::empty())
            .unwrap(),
        Request::get("/api/v1/tickets/demo-0001/handoff-package")
            .header("x-pulse-role", "MANAGER")
            .body(Body::empty())
            .unwrap(),
        Request::get("/api/v1/tickets/demo-0001/prediction")
            .header("x-pulse-role", "MANAGER")
            .body(Body::empty())
            .unwrap(),
        Request::post("/api/v1/assist/preview")
            .header("x-pulse-role", "MANAGER")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"text":"synthetic role probe"}"#))
            .unwrap(),
        Request::post("/api/v1/assist/demo-0001/confirm")
            .header("x-pulse-role", "MANAGER")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap(),
        Request::post("/api/v1/assist/demo-0001/correct")
            .header("x-pulse-role", "MANAGER")
            .header("content-type", "application/json")
            .body(Body::from("{}"))
            .unwrap(),
        Request::post("/api/v1/tickets/demo-0001/relation-feedback")
            .header("x-pulse-role", "MANAGER")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"relation":"SIMILAR","decision":"REJECTED"}"#,
            ))
            .unwrap(),
        Request::post("/api/v1/tickets/demo-0001/routing-feedback")
            .header("x-pulse-role", "MANAGER")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"service_feedback":"ACCEPTED"}"#))
            .unwrap(),
        Request::get("/api/v1/models")
            .header("x-pulse-role", "MANAGER")
            .body(Body::empty())
            .unwrap(),
        Request::get("/api/v1/models/classifier-deterministic-baseline-2026-09-21")
            .header("x-pulse-role", "MANAGER")
            .body(Body::empty())
            .unwrap(),
    ];

    for request in requests {
        let response = application.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), 403);
    }
}

#[tokio::test]
async fn admin_retains_full_demo_access_to_operator_and_model_views() {
    let application = app(AppState::demo());
    let requests = [
        Request::get("/api/v1/tickets")
            .header("x-pulse-role", "ADMIN")
            .body(Body::empty())
            .unwrap(),
        Request::get("/api/v1/tickets/ticket-001")
            .header("x-pulse-role", "ADMIN")
            .body(Body::empty())
            .unwrap(),
        Request::post("/api/v1/assist/preview")
            .header("x-pulse-role", "ADMIN")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"text":"synthetic admin role probe"}"#))
            .unwrap(),
        Request::get("/api/v1/models")
            .header("x-pulse-role", "ADMIN")
            .body(Body::empty())
            .unwrap(),
    ];

    for request in requests {
        let response = application.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), 200);
    }
}

#[tokio::test]
async fn audit_log_is_manager_admin_only_and_requires_postgres_storage() {
    let application = app(AppState::demo());

    for role in ["OPERATOR", "ML_REVIEWER"] {
        let response = application
            .clone()
            .oneshot(
                Request::get("/api/v1/audit")
                    .header("x-pulse-role", role)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 403, "{role} should not read audit log");
    }

    for role in ["MANAGER", "ADMIN"] {
        let response = application
            .clone()
            .oneshot(
                Request::get("/api/v1/audit")
                    .header("x-pulse-role", role)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), 409, "{role} should pass RBAC");
    }

    let invalid_limit = application
        .oneshot(
            Request::get("/api/v1/audit?limit=101")
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(invalid_limit.status(), 400);
}

#[tokio::test]
async fn production_mode_ignores_demo_role_headers() {
    let mut state = AppState::demo();
    state.config.dev_auth = false;
    let application = app(state);

    let demo_header = application
        .clone()
        .oneshot(
            Request::get("/api/v1/analytics")
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(demo_header.status(), 401);

    let gateway_headers = application
        .oneshot(
            Request::get("/api/v1/analytics")
                .header("x-authenticated-role", "MANAGER")
                .header("x-authenticated-user", "gateway-user-17")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(gateway_headers.status(), 200);
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

    let monitored = application
        .clone()
        .oneshot(
            Request::post("/api/v1/alerts/alert-001/monitor")
                .header("x-pulse-role", "MANAGER")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"monitoring_period_days":7}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(monitored.status(), 200);
    let monitoring_change = tokio::time::timeout(Duration::from_secs(2), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(String::from_utf8_lossy(&monitoring_change).contains("event: alerts.changed"));

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
async fn demo_alert_detection_reports_insufficient_history_explicitly() {
    let application = app(AppState::demo());
    let list_response = application
        .clone()
        .oneshot(
            Request::get("/api/v1/alerts")
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(list_response.status(), 200);
    let alerts: serde_json::Value = serde_json::from_slice(
        &to_bytes(list_response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(alerts["total"], 2);
    for alert in alerts["items"].as_array().unwrap() {
        assert_eq!(
            alert["current_count"].as_u64(),
            Some(alert["linked_ticket_ids"].as_array().unwrap().len() as u64)
        );
        assert_eq!(
            alert["description"],
            "Зафиксирован необычный рост обращений. Требуется проверка."
        );
    }

    let response = application
        .oneshot(
            Request::post("/api/v1/alerts/detect")
                .header("x-pulse-role", "MANAGER")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let detection: serde_json::Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(detection["status"], "INSUFFICIENT_HISTORY");
    assert_eq!(detection["source"], "memory_demo");
    assert_eq!(detection["evaluated_series"], 0);
    assert_eq!(detection["items"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn manager_can_start_one_bounded_alert_monitoring_period_without_changing_alert_status() {
    let application = app(AppState::demo());
    let forbidden = application
        .clone()
        .oneshot(
            Request::post("/api/v1/alerts/alert-001/monitor")
                .header("x-pulse-role", "OPERATOR")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"monitoring_period_days":14}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(forbidden.status(), 403);

    let invalid_period = application
        .clone()
        .oneshot(
            Request::post("/api/v1/alerts/alert-001/monitor")
                .header("x-pulse-role", "MANAGER")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"monitoring_period_days":8}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(invalid_period.status(), 400);

    let started = application
        .clone()
        .oneshot(
            Request::post("/api/v1/alerts/alert-001/monitor")
                .header("x-pulse-role", "MANAGER")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"monitoring_period_days":14}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(started.status(), 200);
    let started: serde_json::Value =
        serde_json::from_slice(&to_bytes(started.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(started["status"], "open");
    assert_eq!(started["monitoring"]["state"], "MONITORING");
    assert_eq!(started["monitoring"]["monitoring_period_days"], 14);
    assert_eq!(started["monitoring"]["observation_period_days"], 7);
    assert!(started["monitoring"]["evidence"].is_null());
    assert!(started["description"]
        .as_str()
        .unwrap()
        .contains("Требуется проверка"));

    let duplicate = application
        .oneshot(
            Request::post("/api/v1/alerts/alert-001/monitor")
                .header("x-pulse-role", "MANAGER")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"monitoring_period_days":7}"#))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(duplicate.status(), 409);
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
