FROM rust:1.88-bookworm AS builder
WORKDIR /app
COPY backend/Cargo.toml backend/Cargo.lock* ./
COPY backend/src ./src
# sqlx embeds and applies the repository migrations at startup.
COPY migrations /migrations
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/pulse109-core /usr/local/bin/pulse109-core
EXPOSE 8080
ENV PULSE_HOST=0.0.0.0 \
    PULSE_PORT=8080 \
    PULSE_DEV_AUTH=true \
    RUST_LOG=pulse109_core=info
CMD ["/usr/local/bin/pulse109-core"]
