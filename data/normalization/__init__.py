"""Normalization and privacy-safe preprocessing for source rows."""

from .pii import PIIReport, minimize_mapping, minimize_text, scan_pii


def __getattr__(name):
    # Keep pii.py importable from quarantine.records without importing the
    # pipeline back through the package during module initialization.
    if name in {"NormalizationResult", "normalize_row"}:
        from .pipeline import NormalizationResult, normalize_row

        return {"NormalizationResult": NormalizationResult, "normalize_row": normalize_row}[name]
    raise AttributeError(name)

__all__ = [
    "PIIReport",
    "minimize_mapping",
    "minimize_text",
    "scan_pii",
    "NormalizationResult",
    "normalize_row",
]
