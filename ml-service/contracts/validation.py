"""Validate versioned JSON contracts without importing Core or training code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource


CONTRACT_ROOT = Path(__file__).resolve().parent
INDEX_PATH = CONTRACT_ROOT / "index.json"


class ContractValidationError(ValueError):
    """A contract failed without including rejected payload values in the error."""

    def __init__(self, schema_name: str, instance_path: str, keyword: str):
        self.schema_name = schema_name
        self.instance_path = instance_path
        self.keyword = keyword
        location = instance_path or "<root>"
        super().__init__(f"{schema_name} invalid at {location} ({keyword})")


def _schema_index() -> dict[str, str]:
    payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return dict(payload["schemas"])


SCHEMA_NAMES = _schema_index()
_SCHEMA_PAYLOADS: dict[str, Mapping[str, Any]] = {}
_SCHEMA_REGISTRY = Registry()


def _load_schemas() -> tuple[dict[str, Mapping[str, Any]], Registry]:
    schemas: dict[str, Mapping[str, Any]] = {}
    registry = Registry()
    for name, relative_path in SCHEMA_NAMES.items():
        payload = json.loads((CONTRACT_ROOT / relative_path).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(payload)
        schemas[name] = payload
        registry = registry.with_resource(
            payload["$id"], Resource.from_contents(payload)
        )
    return schemas, registry


def validate_document(document: Any, schema_name: str) -> None:
    """Validate one JSON-compatible value against a named versioned schema."""

    if schema_name not in SCHEMA_NAMES:
        raise KeyError(f"unknown contract schema: {schema_name}")
    global _SCHEMA_PAYLOADS, _SCHEMA_REGISTRY
    if not _SCHEMA_PAYLOADS:
        _SCHEMA_PAYLOADS, _SCHEMA_REGISTRY = _load_schemas()
    schema = _SCHEMA_PAYLOADS[schema_name]
    validator = Draft202012Validator(
        schema,
        registry=_SCHEMA_REGISTRY,
        format_checker=FormatChecker(),
    )
    error = next(validator.iter_errors(document), None)
    if error is None:
        return
    instance_path = ".".join(str(part) for part in error.absolute_path)
    raise ContractValidationError(schema_name, instance_path, str(error.validator))
