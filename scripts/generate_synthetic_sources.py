#!/usr/bin/env python3
"""Generate synthetic raw exports for exercising all seven source importers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.importers import get_importer


FIXTURE_VERSION = "synthetic-sources.v1"
SPECS = (
    ("iKOMEK109", ("requestNumber", "region", "createdDate", "appealText", "direction", "status"),
     ("ticket_id", "region_id", "dateCreated", "description", "serviceDirection"), ","),
    ("АС Комек 109", ("nomer_obrasheniya", "регион", "data_sozdaniya", "opisanie", "vid_rabot", "статус"),
     ("номер обращения", "область", "дата создания", "текст обращения", "категория"), ";"),
    ("AIKEY", ("appealId", "region", "registeredAt", "messageText", "categoryName", "status"),
     ("id", "oblast", "createdAt", "message", "category"), ","),
    ("Открытый город", ("requestId", "city", "date_created", "request_text", "topic", "status"),
     ("request_id", "region", "created", "description", "category"), "\t"),
    ("E-SEP.SU", ("case_number", "region", "registration_date", "case_text", "topic", "status"),
     ("caseNo", "region_id", "registered", "description", "category"), ","),
    ("ЕКЦ-109", ("card_id", "region", "opened_at", "complaint", "topic", "status"),
     ("cardId", "region_id", "openDate", "complaintText", "category"), "\t"),
    ("RDJardem3.0", ("ticketNo", "region", "created_on", "appeal", "topic", "status"),
     ("ticket_number", "region_id", "createdAt", "appeal_body", "category"), ","),
)


def _write_table(path: Path, headers: tuple[str, ...], rows: list[tuple[str, ...]], delimiter: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(output_dir: Path, *, check: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = []
    for display_name, primary, alternate, delimiter in SPECS:
        importer = get_importer(display_name)
        source_dir = output_dir / importer.source_system
        source_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".tsv" if delimiter == "\t" else ".csv"
        # Keep these invented identifiers and the reserved .invalid domain out of real-source evidence.
        rows = [
            ("synthetic-1", "Астана", "2026-01-02T00:00:00Z", "У дома не горит фонарь", "Наружное освещение", "OPEN"),
            ("synthetic-2", "Астана", "2026-01-03T00:00:00Z", "На остановке темно", "Наружное освещение", ""),
            ("synthetic-3", "Астана", "not-a-date", "Нужен свет", "Наружное освещение", "OPEN"),
            ("synthetic-4", "Астана", "2026-01-04T00:00:00Z", "", "Наружное освещение", "OPEN"),
            ("synthetic-5", "Астана", "2026-01-05T00:00:00Z", "Нет воды, e-mail synthetic@example.invalid", "Водоснабжение", "OPEN"),
            ("synthetic-1", "Астана", "2026-01-06T00:00:00Z", "Повторное сообщение", "Наружное освещение", "OPEN"),
            ("synthetic-7", "Астана", "2026-01-07T00:00:00Z", "Неясная новая проблема", "Несуществующая категория", "OPEN"),
        ]
        main_path = source_dir / ("primary" + suffix)
        _write_table(main_path, primary, rows, delimiter)
        alternate_path = source_dir / "alternate.csv"
        csv_delimiter = importer.csv_delimiter
        _write_table(alternate_path, alternate,
                     [("synthetic-alt", "Астана", "2026-01-08T00:00:00Z", "Не горит свет", "Наружное освещение")], csv_delimiter)
        malformed_path = source_dir / "malformed.csv"
        _write_table(malformed_path, primary, [rows[0]], csv_delimiter)
        with malformed_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(f'synthetic-bad{csv_delimiter}Астана{csv_delimiter}2026-01-09T00:00:00Z{csv_delimiter}"unterminated\n')
        unknown_path = source_dir / "unknown.csv"
        _write_table(unknown_path, ("unrecognized_id", "unrecognized_date", "unrecognized_text"),
                     [("synthetic-unknown", "2026-01-09", "Текст")], ",")
        paths = {"primary": main_path, "alternate": alternate_path, "malformed": malformed_path, "unknown": unknown_path}
        if check:
            primary_result = importer.import_file(main_path)
            alternate_result = importer.import_file(alternate_path)
            malformed_result = importer.import_file(malformed_path)
            unknown_result = importer.import_file(unknown_path)
            assert (primary_result.valid_count, primary_result.quarantine_count) == (4, 3)
            assert {item.reason for item in primary_result.quarantine} == {"INVALID_DATE", "MISSING_REQUIRED_FIELD", "INVALID_VALUE"}
            assert primary_result.tickets[-1].topic_id == "unknown"
            assert "synthetic@example.invalid" not in primary_result.tickets[2].original_text
            assert (alternate_result.valid_count, alternate_result.quarantine_count) == (1, 0)
            assert (malformed_result.valid_count, [item.reason for item in malformed_result.quarantine]) == (1, ["BAD_CSV_STRUCTURE"])
            assert (unknown_result.valid_count, [item.reason for item in unknown_result.quarantine]) == (0, ["UNKNOWN_SCHEMA"])
        sources.append({
            "source_system": importer.source_system,
            "display_name": display_name,
            "profile_version": importer.profile_version,
            "profile_status": importer.profile_status,
            "schema_fingerprints": {
                "primary": importer.schema_fingerprint(primary),
                "alternate": importer.schema_fingerprint(alternate),
            },
            "files": {kind: {"path": str(path.relative_to(output_dir)), "sha256": _checksum(path)} for kind, path in paths.items()},
        })
    manifest = {"fixture_version": FIXTURE_VERSION, "synthetic": True, "seed": 109,
                "purpose": "importer validation only; not real source evidence or training data", "sources": sources}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "synthetic_raw" / "v1")
    parser.add_argument("--check", action="store_true", help="run all fixtures through their source importers")
    args = parser.parse_args()
    manifest = generate(args.output_dir, check=args.check)
    print(json.dumps({"fixture_version": manifest["fixture_version"], "source_count": len(manifest["sources"]),
                      "synthetic": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
