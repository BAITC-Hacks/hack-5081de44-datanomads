use pulse109_core::{app, AppState};
use tokio::net::TcpListener;
use tracing::info;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let state = AppState::from_env();
    let host = state.config.host.clone();
    let port = state.config.port;
    tracing_subscriber::fmt()
        .json()
        .flatten_event(true)
        .with_target(false)
        .with_current_span(true)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("pulse109_core=info")),
        )
        .init();

    state
        .initialize()
        .await
        .map_err(|error| format!("core initialization failed: {error}"))?;
    let listener = TcpListener::bind((host.as_str(), port)).await?;
    info!(
        service = "pulse109-core",
        address = %listener.local_addr()?,
        storage = %state.config.storage,
        request_id = "startup",
        trace_id = "startup",
        endpoint = "startup",
        latency_ms = 0.0_f64,
        model_version = "n/a",
        status = 200_u16,
        error_code = "none",
        "server_started"
    );
    axum::serve(listener, app(state)).await?;
    Ok(())
}
