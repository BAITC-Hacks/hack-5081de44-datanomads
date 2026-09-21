"""Deterministic PII detection and minimization.

The normalized contract never stores raw PII-bearing columns.  Known values in
ticket text are replaced with typed tokens so that operators can still read
the issue while ML and analytics receive a safe representation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple


_PATTERNS = (
    ("EMAIL", re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")),
    (
        "PHONE",
        re.compile(
            r"(?<!\d)(?:\+?7|8)[\s().-]*\d{3}[\s().-]*\d{3}[\s().-]*\d{2}[\s().-]*\d{2}(?!\d)"
        ),
    ),
    ("IIN", re.compile(r"(?<!\d)\d{12}(?!\d)")),
    (
        "CARD",
        re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    ),
)

# A label is deliberately required for names and addresses.  Guessing that a
# pair of ordinary words is a person's name would remove useful ticket text.
_LABELED_VALUE_PATTERNS = (
    (
        "NAME",
        re.compile(
            r"(?is)(?P<label>\b(?:фио|ф\.и\.о\.|имя|name|аты[- ]?жөні)\s*[:№-]?\s*)(?P<value>[^,;\n|]+)"
        ),
    ),
    (
        "ADDRESS",
        re.compile(
            r"(?is)(?P<label>\b(?:адрес|address|мекенжай)\s*[:№-]?\s*)(?P<value>[^,;\n|]+)"
        ),
    ),
)

_SENSITIVE_KEY_PARTS = (
    "iin",
    "phone",
    "телефон",
    "email",
    "e-mail",
    "name",
    "full_name",
    "fio",
    "фио",
    "аты жөні",
    "address",
    "адрес",
    "мекенжай",
)

_REDACTION_TOKEN_RE = re.compile(
    r"\[(?:EMAIL|PHONE|IIN|CARD|NAME|ADDRESS|ATTACHMENT(?:_REMOVED)?|PII_FIELD)\]"
)


@dataclass(frozen=True)
class PIIReport:
    """PII categories observed before masking."""

    categories: Tuple[str, ...] = ()
    redaction_count: int = 0

    @property
    def detected(self) -> bool:
        return bool(self.categories)


def _token(category: str) -> str:
    return f"[{category}]"


def _normalize_text(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def scan_pii(value: object) -> PIIReport:
    """Scan a value without returning matched sensitive text."""

    text = _REDACTION_TOKEN_RE.sub("", _normalize_text(value))
    categories = []
    count = 0
    for category, pattern in _PATTERNS:
        matches = pattern.findall(text)
        if matches:
            categories.append(category)
            count += len(matches)
    for category, pattern in _LABELED_VALUE_PATTERNS:
        match = pattern.search(text)
        if match and match.group("value").strip():
            categories.append(category)
            count += 1
    return PIIReport(tuple(dict.fromkeys(categories)), count)


def minimize_text(value: object) -> Tuple[str, PIIReport]:
    """Mask known PII in text and return the safe text plus an audit report."""

    text = _normalize_text(value)
    categories = []
    count = 0

    for category, pattern in _LABELED_VALUE_PATTERNS:
        def replace_labeled(match: re.Match[str], category: str = category) -> str:
            nonlocal count
            categories.append(category)
            count += 1
            # Keep the label for operator readability while removing the value.
            return f"{match.group('label')}{_token(category)}"

        text = pattern.sub(replace_labeled, text)

    for category, pattern in _PATTERNS:
        def replace_value(match: re.Match[str], category: str = category) -> str:
            nonlocal count
            categories.append(category)
            count += 1
            return _token(category)

        text = pattern.sub(replace_value, text)

    report = PIIReport(tuple(dict.fromkeys(categories)), count)
    return text, report


def _key_is_sensitive(key: object) -> bool:
    folded = unicodedata.normalize("NFKC", str(key or "")).lower().replace("-", "_")
    return any(part in folded for part in _SENSITIVE_KEY_PARTS)


def minimize_mapping(row: Mapping[str, Any]) -> Tuple[Dict[str, Any], PIIReport]:
    """Return a copy safe for quarantine or diagnostics.

    Sensitive scalar columns are replaced with typed tokens; attachment
    contents are removed rather than copied to quarantine.  Other text fields
    receive the same deterministic text masking as ``minimize_text``.
    """

    safe: Dict[str, Any] = {}
    categories = []
    count = 0
    for key, value in row.items():
        key_text = str(key)
        if value is None:
            safe[key_text] = None
            continue
        if "attachment" in key_text.lower() or "влож" in key_text.lower():
            if str(value).strip():
                categories.append("ATTACHMENT")
                count += 1
            safe[key_text] = _token("ATTACHMENT_REMOVED")
            continue
        if _key_is_sensitive(key_text):
            category = "PII_FIELD"
            lowered = key_text.lower()
            if "phone" in lowered or "телефон" in lowered:
                category = "PHONE"
            elif "email" in lowered or "e-mail" in lowered:
                category = "EMAIL"
            elif "iin" in lowered:
                category = "IIN"
            elif "name" in lowered or "fio" in lowered or "фио" in lowered or "аты" in lowered:
                category = "NAME"
            elif "address" in lowered or "адрес" in lowered or "мекенжай" in lowered:
                category = "ADDRESS"
            categories.append(category)
            count += 1
            safe[key_text] = _token(category)
            continue
        if isinstance(value, str):
            masked, report = minimize_text(value)
            safe[key_text] = masked
            categories.extend(report.categories)
            count += report.redaction_count
        else:
            safe[key_text] = value
    return safe, PIIReport(tuple(dict.fromkeys(categories)), count)


__all__ = ["PIIReport", "minimize_mapping", "minimize_text", "scan_pii"]
