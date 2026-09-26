"""Versioned, dependency-light contracts shared at the Data/ML boundary."""

from .validation import ContractValidationError, SCHEMA_NAMES, validate_document

__all__ = ["ContractValidationError", "SCHEMA_NAMES", "validate_document"]
