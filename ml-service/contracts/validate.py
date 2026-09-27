"""CLI for offline validation of Pulse Data/ML contract artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from . import SCHEMA_NAMES, ContractValidationError, validate_document


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_file(path: Path, schema_name: str) -> None:
    try:
        validate_document(_read_json(path), schema_name)
    except (OSError, json.JSONDecodeError, ContractValidationError) as error:
        raise ValueError(f"{path}: {error}") from None


def validate_demo_artifacts() -> list[str]:
    """Validate checked-in synthetic fixtures; never treat them as model evidence."""

    checked: list[str] = []
    data_schema = _read_json(REPOSITORY_ROOT / "data/schemas/unified_ticket.schema.json")
    shared_schema = _read_json(
        Path(__file__).resolve().parent / SCHEMA_NAMES["UnifiedTicket"]
    )
    if data_schema != shared_schema:
        raise ValueError("Data UnifiedTicket schema differs from the shared v1 contract")

    dataset_manifest = REPOSITORY_ROOT / "data/manifests/demo-2026-09-21.json"
    _validate_file(dataset_manifest, "DatasetManifest")
    checked.append("DatasetManifest")

    ticket_path = REPOSITORY_ROOT / "data/demo/tickets.jsonl"
    with ticket_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    validate_document(json.loads(line), "UnifiedTicket")
                except (json.JSONDecodeError, ContractValidationError) as error:
                    raise ValueError(f"{ticket_path}:{line_number}: {error}") from None
    checked.append("UnifiedTicket records")

    model_manifest_path = Path(__file__).resolve().parent.parent / "artifacts/manifest.json"
    _validate_file(model_manifest_path, "ModelManifest")
    checked.append("ModelManifest")
    model_manifest = _read_json(model_manifest_path)
    validate_document(model_manifest["models"]["classifier"], "ClassifierManifest")
    validate_document(model_manifest["models"]["embedder"], "EmbedderManifest")
    checked.extend(["ClassifierManifest", "EmbedderManifest"])

    examples = Path(__file__).resolve().parent / "examples"
    for schema_name, file_name in (
        ("ModelEvaluation", "model-evaluation.demo.json"),
        ("LearningFeedbackExport", "learning-feedback-export.demo.json"),
        ("CandidateDatasetBuildRequest", "candidate-dataset-build-request.demo.json"),
        ("CandidateDatasetManifest", "candidate-dataset-manifest.demo.json"),
        ("CandidateEvaluation", "candidate-evaluation.demo.json"),
    ):
        _validate_file(examples / file_name, schema_name)
        checked.append(schema_name)
    return checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", choices=sorted(SCHEMA_NAMES))
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--check-demo",
        action="store_true",
        help="validate checked-in synthetic Data and ML fixtures",
    )
    args = parser.parse_args(argv)
    if args.check_demo:
        try:
            checked = validate_demo_artifacts()
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(f"contract validation failed: {error}", file=sys.stderr)
            return 1
        print("Validated: " + ", ".join(checked))
        return 0
    if not args.schema or not args.input:
        parser.error("--schema and --input are required unless --check-demo is used")
    try:
        _validate_file(args.input, args.schema)
    except ValueError as error:
        print(f"contract validation failed: {error}", file=sys.stderr)
        return 1
    print(f"Valid {args.schema}: {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
