"""Pulse 109 data foundation.

The package intentionally has no runtime dependencies.  It is used by import
jobs and by the deterministic demo-data generator; the Rust API can consume
the JSONL and SQL contracts without importing this package.
"""

__all__ = ["schemas", "importers", "normalization"]
