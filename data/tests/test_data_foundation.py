from __future__ import annotations

import csv
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from data.importers import IKOMEK109Importer, get_importer
from data.normalization import minimize_text, scan_pii
from data.normalization.pipeline import normalize_row
from data.schemas.taxonomy import REGION_DEFINITIONS, TOPIC_DEFINITIONS, canonical_topic_id
from data.schemas.unified_ticket import SchemaValidationError, UnifiedTicket
from scripts.data_audit import build_normalized_report, build_report, build_source_report
from scripts.generate_synthetic_sources import SPECS, generate as generate_synthetic_sources
from scripts import import_tickets


ROOT = Path(__file__).resolve().parents[2]


class UnifiedTicketTests(unittest.TestCase):
    def test_minimal_contract_serializes_dates_and_defaults(self) -> None:
        ticket = UnifiedTicket.from_mapping(
            {
                "external_ticket_id": "x-1",
                "source_system": "ikomek109",
                "region_id": "KZ-ABAY",
                "created_at": "2026-01-01T00:00:00Z",
                "original_text": "Проверить освещение",
            }
        )
        payload = ticket.to_dict()
        self.assertEqual(payload["schema_version"], "unified-ticket.v1")
        self.assertEqual(payload["status"], "UNKNOWN")
        self.assertEqual(payload["created_at"], "2026-01-01T00:00:00+00:00")

    def test_unsupported_schema_version_is_rejected(self) -> None:
        with self.assertRaises(SchemaValidationError) as raised:
            UnifiedTicket.from_mapping({
                "external_ticket_id": "x-1", "source_system": "ikomek109",
                "region_id": "KZ-ABAY", "created_at": "2026-01-01T00:00:00Z",
                "original_text": "Проверить освещение", "schema_version": "unified-ticket.v2",
            })
        self.assertEqual(raised.exception.code, "UNKNOWN_SCHEMA")

    def test_pii_is_masked_before_ticket_is_created(self) -> None:
        text, report = minimize_text("ФИО: Иван Иванов, телефон +7 777 123-45-67, ИИН 900101123456")
        self.assertTrue(report.detected)
        self.assertNotIn("Иван Иванов", text)
        self.assertNotIn("+7 777", text)
        self.assertNotIn("900101123456", text)
        self.assertEqual(scan_pii(text).categories, ())
        result = normalize_row(
            {
                "external_ticket_id": "pii-1",
                "region_id": "Акмолинская область",
                "created_at": "2026-01-01",
                "original_text": "ФИО: Иван Иванов, телефон +7 777 123-45-67",
            },
            source_system="ikomek109",
        )
        self.assertTrue(result.valid)
        self.assertNotIn("Иван Иванов", result.ticket.original_text)

    def test_quarantine_omits_sensitive_values_from_detail_and_snapshot(self) -> None:
        result = normalize_row(
            {
                "external_ticket_id": "bad-date-1",
                "region_id": "Акмолинская область",
                "created_at": "+7 777 123-45-67",
                "original_text": "ФИО: Иван Иванов, нет воды",
            },
            source_system="ikomek109",
        )
        self.assertIsNotNone(result.quarantine)
        encoded = json.dumps(result.quarantine.to_dict(), ensure_ascii=False)
        self.assertNotIn("+7 777 123-45-67", encoded)
        self.assertNotIn("Иван Иванов", encoded)
        self.assertEqual(result.quarantine.reason, "INVALID_DATE")

    def test_sensitive_source_identifier_is_quarantined_without_leaking_it(self) -> None:
        result = normalize_row(
            {
                "external_ticket_id": "ИИН 900101123456",
                "region_id": "Астана",
                "created_at": "2026-01-01",
                "original_text": "Не горит фонарь",
            },
            source_system="ikomek109",
        )
        self.assertIsNone(result.ticket)
        self.assertEqual(result.quarantine.reason, "PII_REVIEW")
        self.assertNotIn("900101123456", json.dumps(result.quarantine.to_dict(), ensure_ascii=False))

    def test_quarantine_does_not_serialize_unknown_sensitive_fields(self) -> None:
        result = get_importer("AIKEY").import_rows([{
            "appealId": "synthetic-1", "region": "Астана", "registeredAt": "bad",
            "messageText": "Текст", "secret_contact": "Персональный тестовый секрет",
            "private@example.invalid": "value",
        }])
        encoded = json.dumps(result.quarantine[0].to_dict(), ensure_ascii=False)
        self.assertNotIn("Персональный тестовый секрет", encoded)
        self.assertNotIn("private@example.invalid", encoded)
        self.assertNotIn("secret_contact", encoded)

    def test_optional_text_and_precise_location_are_minimized(self) -> None:
        result = normalize_row({
            "external_ticket_id": "synthetic-safe-1",
            "region_id": "Астана",
            "created_at": "2026-01-01",
            "original_text": "Нет воды",
            "topic_raw": "Водоснабжение synthetic@example.invalid",
            "district": "Район, телефон +7 700 000-00-00",
            "official_response": "Ответ на synthetic@example.invalid",
            "coordinates": {"latitude": 51.1, "longitude": 71.4},
        }, source_system="AIKEY")
        self.assertTrue(result.valid)
        encoded = json.dumps(result.ticket.to_dict(), ensure_ascii=False)
        self.assertNotIn("synthetic@example.invalid", encoded)
        self.assertNotIn("+7 700 000-00-00", encoded)
        self.assertNotIn("51.1", encoded)
        self.assertEqual(result.ticket.topic_id, "unknown")
        self.assertTrue(result.ticket.needs_review)
        self.assertGreaterEqual(result.ticket.text_redaction_count, 3)


class ImporterTests(unittest.TestCase):
    def test_every_declared_header_alias_is_exactly_mapped(self) -> None:
        for source_name, *_ in SPECS:
            importer = get_importer(source_name)
            required = {field: aliases[0] for field, aliases in importer.field_aliases.items()
                        if field in importer.required_headers}
            for field, aliases in importer.field_aliases.items():
                for alias in aliases:
                    with self.subTest(source=source_name, field=field, alias=alias):
                        headers = {**required, field: alias}
                        self.assertEqual(importer._header_map(list(headers.values()))[field], alias)

    def test_synthetic_exports_cover_each_source_and_bad_row_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            manifest = generate_synthetic_sources(output_dir, check=True)
            first_manifest = (output_dir / "manifest.json").read_bytes()
            self.assertEqual(len(manifest["sources"]), 7)
            self.assertTrue(manifest["synthetic"])
            for source in manifest["sources"]:
                self.assertEqual(source["profile_status"], "SYNTHETIC_TEST_ONLY")
                for entry in source["files"].values():
                    self.assertEqual(hashlib.sha256((output_dir / entry["path"]).read_bytes()).hexdigest(), entry["sha256"])
            generate_synthetic_sources(output_dir, check=True)
            self.assertEqual((output_dir / "manifest.json").read_bytes(), first_manifest)

    def test_ambiguous_and_unsupported_headers_are_quarantined(self) -> None:
        importer = get_importer("AIKEY")
        base = {"appealId": "synthetic-1", "region": "Астана", "registeredAt": "2026-01-01", "messageText": "Текст"}
        for extra in ({"id": "synthetic-2"}, {"message": "Текст"}, {"other": "x"}):
            with self.subTest(extra=extra):
                result = importer.import_rows([{**base, **extra}])
                expected = "UNKNOWN_SCHEMA" if "id" in extra or "message" in extra else None
                self.assertEqual(result.quarantine[0].reason if result.quarantine else None, expected)
        self.assertEqual(importer.import_rows([{"appeal-id": "synthetic-1", "region": "Астана", "registeredAt": "2026-01-01", "messageText": "Текст"}]).quarantine[0].reason, "UNKNOWN_SCHEMA")

    def test_invalid_csv_encoding_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_bytes(b"appealId,region,registeredAt,messageText\nsynthetic-1,Astana,2026-01-01,ok\n\xff")
            result = get_importer("AIKEY").import_file(path)
        self.assertEqual([row.reason for row in result.quarantine], ["BAD_CSV_STRUCTURE"])
        self.assertEqual(result.valid_count, 0)

    def test_csv_quarantine_keeps_physical_lines_after_bad_and_multiline_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_text(
                'appealId,region,registeredAt,messageText\n'
                'one,Астана,2026-01-01,"Первая\nстрока"\n'
                'two,Астана,2026-01-02\n'
                'three,Астана,bad,Текст\n', encoding="utf-8",
            )
            result = get_importer("AIKEY").import_file(path)
        self.assertEqual(result.valid_count, 1)
        self.assertEqual([(row.reason, row.row_number) for row in result.quarantine],
                         [("BAD_CSV_STRUCTURE", 4), ("INVALID_DATE", 5)])

    def test_unrecoverable_csv_parse_rejects_entire_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            path.write_text(
                'appealId,region,registeredAt,messageText\n'
                'one,Астана,2026-01-01,Текст\n'
                'two,Астана,2026-01-02,"unterminated\n', encoding="utf-8",
            )
            result = get_importer("AIKEY").import_file(path)
        self.assertEqual(result.valid_count, 0)
        self.assertEqual([(row.reason, row.row_number) for row in result.quarantine],
                         [("BAD_CSV_STRUCTURE", 3)])
        self.assertIn("entire file rejected", result.quarantine[0].detail)

    def test_jsonl_quarantine_keeps_physical_lines_after_bad_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.jsonl"
            path.write_text(
                '{"appealId":"one","region":"Астана","registeredAt":"2026-01-01","messageText":"Текст"}\n'
                '{broken json}\n'
                '{"appealId":"three","region":"Астана","registeredAt":"bad","messageText":"Текст"}\n',
                encoding="utf-8",
            )
            result = get_importer("AIKEY").import_file(path)
        self.assertEqual(result.valid_count, 1)
        self.assertEqual([(row.reason, row.row_number) for row in result.quarantine],
                         [("UNKNOWN_SCHEMA", 2), ("INVALID_DATE", 3)])

    def test_duplicate_json_fields_are_quarantined_before_mapping(self) -> None:
        valid = '{"appealId":"one","region":"Астана","registeredAt":"2026-01-01","messageText":"Текст"}'
        duplicate = '{"appealId":"spoof","appealId":"two","region":"Астана","registeredAt":"2026-01-02","messageText":"Текст"}'
        nonstandard = '{"appealId":NaN,"region":"Астана","registeredAt":"2026-01-03","messageText":"Текст"}'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "source.jsonl"
            jsonl.write_text(valid + "\n" + duplicate + "\n" + nonstandard + "\n", encoding="utf-8")
            result = get_importer("AIKEY").import_file(jsonl)
            self.assertEqual(result.valid_count, 1)
            self.assertEqual([(row.reason, row.row_number) for row in result.quarantine],
                             [("UNKNOWN_SCHEMA", 2), ("UNKNOWN_SCHEMA", 3)])

            array = root / "source.json"
            nested = valid[:-1] + ',"metadata":{"key":"one","key":"two"}}'
            array.write_text("[" + valid + "," + nested + "]", encoding="utf-8")
            result = get_importer("AIKEY").import_file(array)
            self.assertEqual(result.valid_count, 0)
            self.assertEqual([(row.reason, row.row_number) for row in result.quarantine], [("UNKNOWN_SCHEMA", 1)])

    def test_xlsx_quarantine_uses_sheet_row_number(self) -> None:
        def cells(values: tuple[str, ...]) -> str:
            return "".join(f'<c t="inlineStr"><is><t>{value}</t></is></c>' for value in values)

        sheet = (
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData>'
            f'<row r="1">{cells(("appealId", "region", "registeredAt", "messageText"))}</row>'
            f'<row r="3">{cells(("one", "Астана", "2026-01-01", "Текст"))}</row>'
            f'<row r="5">{cells(("two", "Астана", "bad", "Текст"))}</row>'
            '</sheetData></worksheet>'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.xlsx"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("xl/worksheets/sheet1.xml", sheet)
            result = get_importer("AIKEY").import_file(path)
        self.assertEqual(result.valid_count, 1)
        self.assertEqual([(row.reason, row.row_number) for row in result.quarantine],
                         [("INVALID_DATE", 5)])

    def test_source_specific_aliases_and_bad_rows(self) -> None:
        importer = IKOMEK109Importer()
        result = importer.import_rows(
            [
                {
                    "requestNumber": "a-1",
                    "createdDate": "2026-01-02T00:00:00Z",
                    "appealText": "Не горит фонарь",
                    "region": "Ақмола облысы",
                    "direction": "Наружное освещение",
                },
                {
                    "requestNumber": "a-2",
                    "createdDate": "not-a-date",
                    "appealText": "Текст",
                    "region": "Ақмола облысы",
                },
                {
                    "requestNumber": "a-3",
                    "createdDate": "2026-01-02",
                    "appealText": "",
                    "region": "Ақмола облысы",
                },
            ]
        )
        self.assertEqual(result.valid_count, 1)
        self.assertEqual(result.quarantine_count, 2)
        self.assertEqual(
            {row.reason for row in result.quarantine},
            {"INVALID_DATE", "MISSING_REQUIRED_FIELD"},
        )
        self.assertEqual(result.tickets[0].topic_id, "street_lighting")

    def test_csv_structure_is_quarantined(self) -> None:
        importer = get_importer("AIKEY")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text(
                "id,region,created_at,text\n1,Астана,2026-01-01,ok\n2,Астана,2026-01-02\n",
                encoding="utf-8",
            )
            result = importer.import_file(path)
        self.assertEqual(result.valid_count, 1)
        self.assertEqual(result.quarantine_count, 1)
        self.assertEqual(result.quarantine[0].reason, "BAD_CSV_STRUCTURE")

    def test_duplicate_ids_are_not_loaded_twice(self) -> None:
        result = get_importer("open_city").import_rows(
            [
                {"id": "same", "region": "Астана", "created": "2026-01-01", "description": "A"},
                {"id": "same", "region": "Астана", "created": "2026-01-02", "description": "B"},
            ]
        )
        self.assertEqual(result.valid_count, 1)
        self.assertEqual(result.duplicate_external_ids, ["same"])


class ImportCliTests(unittest.TestCase):
    def test_core_failure_does_not_print_exception_details(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text(
                "appealId,region,registeredAt,messageText\n"
                "one,Астана,2026-01-01,Текст\n", encoding="utf-8",
            )
            output = root / "normalized.jsonl"
            quarantine = root / "quarantine.jsonl"
            stdout = StringIO()
            argv = ["import_tickets.py", "--source", "AIKEY", str(source),
                    "--output", str(output), "--quarantine", str(quarantine),
                    "--api-url", "http://core.invalid"]
            with (patch.object(sys, "argv", argv),
                  patch("scripts.import_tickets.urlopen", side_effect=OSError("PRIVATE_SENTINEL")),
                  redirect_stdout(stdout)):
                self.assertEqual(import_tickets.main(), 1)
            self.assertEqual(json.loads(stdout.getvalue())["error"], "CORE_IMPORT_FAILED")
            self.assertNotIn("PRIVATE_SENTINEL", stdout.getvalue())
            self.assertTrue(output.is_file())
            self.assertTrue(quarantine.is_file())

    def test_import_creates_outputs_once_and_preserves_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text(
                "appealId,region,registeredAt,messageText\n"
                "one,Астана,2026-01-01,Текст\n", encoding="utf-8",
            )
            output = root / "normalized.jsonl"
            quarantine = root / "quarantine.jsonl"
            command = [sys.executable, str(ROOT / "scripts/import_tickets.py"),
                       "--source", "AIKEY", str(source), "--output", str(output),
                       "--quarantine", str(quarantine)]
            environment = {key: value for key, value in os.environ.items() if key != "PULSE_CORE_URL"}
            first = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            saved_output = output.read_bytes()
            saved_quarantine = quarantine.read_bytes()

            repeated = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
            self.assertEqual(repeated.returncode, 2)
            self.assertEqual(output.read_bytes(), saved_output)
            self.assertEqual(quarantine.read_bytes(), saved_quarantine)

            next_output = root / "next.jsonl"
            blocked = subprocess.run([*command[:5], "--output", str(next_output),
                                      "--quarantine", str(quarantine)],
                                     cwd=ROOT, env=environment, capture_output=True, text=True)
            self.assertEqual(blocked.returncode, 2)
            self.assertFalse(next_output.exists())
            self.assertEqual(quarantine.read_bytes(), saved_quarantine)


class DemoFixtureTests(unittest.TestCase):
    def test_manifest_and_fixture_coverage(self) -> None:
        manifest_path = ROOT / "data" / "manifests" / "demo-2026-09-29.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["synthetic"])
        self.assertGreaterEqual(manifest["coverage"]["region_count"], 20)
        self.assertGreaterEqual(manifest["coverage"]["topic_count"], 10)
        self.assertEqual(set(manifest["languages"]), {"RU", "KZ"})
        self.assertEqual(
            set(manifest["pii_policy"]["vector_payload_fields"]),
            {"ticket_id", "region_id", "topic_id", "created_at"},
        )
        tickets_path = ROOT / "data" / "demo" / "tickets.jsonl"
        rows = [json.loads(line) for line in tickets_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(rows), manifest["record_count"])
        self.assertEqual(len({row["region_id"] for row in rows}), 20)
        self.assertGreaterEqual(len({row["topic_id"] for row in rows}), 10)
        self.assertFalse(any(scan_pii(row["original_text"]).detected for row in rows))
        self.assertTrue(all(row["address"] is None for row in rows))
        self.assertTrue(all(row["object"] is None for row in rows))
        self.assertTrue(all(row["coordinates"] is None for row in rows))
        self.assertTrue(all(not row["attachments"] for row in rows))
        for entry in manifest["files"]:
            path = ROOT / entry["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, entry["sha256"])

    def test_taxonomy_matches_fixture_contract(self) -> None:
        self.assertEqual(len(REGION_DEFINITIONS), 20)
        self.assertGreaterEqual(len(TOPIC_DEFINITIONS), 10)

    def test_topic_aliases_are_exact_and_ambiguous_labels_need_review(self) -> None:
        self.assertEqual(canonical_topic_id("waste"), "waste_management")
        self.assertEqual(canonical_topic_id("ecology"), "environment")
        self.assertEqual(canonical_topic_id("Наружное освещение"), "street_lighting")
        self.assertEqual(canonical_topic_id("свет"), "unknown")
        self.assertEqual(canonical_topic_id("вода"), "unknown")
        self.assertEqual(canonical_topic_id("Наружное освещение или электроснабжение"), "unknown")


class DataAuditTests(unittest.TestCase):
    def test_normalized_duplicate_ids_are_scoped_to_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tickets.jsonl"
            rows = [
                {"source_system": "aikey", "external_ticket_id": "shared", "created_at": "2026-01-01",
                 "region_id": "KZ-ABAY", "original_text": "Не горит фонарь"},
                {"source_system": "ikomek109", "external_ticket_id": "shared", "created_at": "2026-01-01",
                 "region_id": "KZ-ABAY", "original_text": "Не горит фонарь"},
                {"source_system": "aikey", "external_ticket_id": "shared", "created_at": "2026-01-02",
                 "region_id": "KZ-ABAY", "original_text": "Не горит фонарь"},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            report = build_normalized_report(path)
        self.assertEqual(report["duplicate_external_id_count"], 1)

    def test_normalized_audit_rejects_invalid_schema_ambiguous_json_and_pii(self) -> None:
        base = {"source_system": "aikey", "external_ticket_id": "safe-1",
                "region_id": "KZ-ABAY", "created_at": "2026-01-01",
                "original_text": "Не горит фонарь"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tickets.jsonl"
            rows = [
                json.dumps(base),
                json.dumps({key: value for key, value in base.items() if key != "original_text"}),
                json.dumps({**base, "schema_version": "unified-ticket.v2"}),
                json.dumps({**base, "created_at": "bad-date"}),
                json.dumps({**base, "original_text": "Телефон +7 777 123-45-67"}),
                json.dumps({**base, "resolution_text": "Ответ: synthetic@example.invalid"}),
                json.dumps({**base, "address": "ул. Тестовая, дом 1"}),
                '{"external_ticket_id":"safe-1","external_ticket_id":"spoof"}',
                '{"external_ticket_id":NaN}',
            ]
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            report = build_normalized_report(path)
        self.assertEqual(report["record_count"], 9)
        self.assertEqual(report["valid_record_count"], 1)
        self.assertEqual(report["parse_or_schema_error_count"], 5)
        self.assertEqual(report["pii_rejected_record_count"], 3)
        self.assertEqual(report["invalid_date_count"], 1)
        self.assertNotIn("+7 777", json.dumps(report, ensure_ascii=False))

    def test_source_report_is_aggregate_and_marks_unverified_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = generate_synthetic_sources(root, check=True)
            source = manifest["sources"][0]
            path = root / source["files"]["primary"]["path"]
            report = build_source_report(path, source["source_system"], synthetic=True)
        self.assertEqual((report["valid_record_count"], report["quarantine_count"]), (4, 3))
        self.assertEqual(report["quarantine_reasons"], {"INVALID_DATE": 1, "INVALID_VALUE": 1, "MISSING_REQUIRED_FIELD": 1})
        self.assertEqual(report["pii_redacted_ticket_count"], 1)
        self.assertEqual(report["label_semantics"], "UNVERIFIED_SEMANTICS")
        self.assertEqual(report["label_ground_truth"]["routing"]["status"], "UNSUITABLE")
        self.assertEqual(report["label_ground_truth"]["priority"]["status"], "UNSUITABLE")
        self.assertFalse(report["label_ground_truth"]["routing"]["approved_for_training"])
        self.assertTrue(report["synthetic"])
        encoded = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("synthetic@example.invalid", encoded)
        self.assertNotIn("У дома не горит фонарь", encoded)

    def test_raw_csv_report_contains_counts_without_source_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "application_number", "creation_date", "category", "service",
                    "com_exp", "full_name", "region", "private@example.com",
                ])
                writer.writerow([
                    "ticket-1", "01.02.2025 12:30:00", "Дороги", "Ремонт",
                    "ФИО: Иван Иванов, нет освещения", "Иван Иванов", "Усть-Каменогорск", "secret",
                ])
            report = build_report(path)

        self.assertEqual(report["record_count"], 1)
        self.assertEqual(report["nonempty_by_column"]["com_exp"], 1)
        self.assertEqual(report["by_year"], {"2025": 1})
        self.assertEqual(report["unknown_column_count"], 1)
        self.assertEqual(report["possible_shifted_column_row_count"], 0)
        encoded = json.dumps(report, ensure_ascii=False)
        for sensitive in ("Иван Иванов", "ФИО:", "ticket-1", "private@example.com", "secret"):
            self.assertNotIn(sensitive, encoded)

    def test_raw_csv_report_does_not_require_com_exp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["application_number", "creation_date", "category", "service"])
                writer.writerow(["ticket-1", "01.02.2025 12:30:00", "Дороги", "Ремонт"])
            report = build_report(path)

        self.assertEqual(report["record_count"], 1)
        self.assertNotIn("com_exp", report["nonempty_by_column"])
        self.assertFalse(report["has_original_text_column"])

    def test_raw_csv_report_flags_date_in_non_date_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["application_number", "creation_date", "category", "service"])
                writer.writerow(["ticket-1", "Дороги", "01.02.2025 12:30:00", "Ремонт"])
                writer.writerow(["ticket-2", "02.02.2025 12:30:00", "Дороги", "Ремонт"])
            report = build_report(path)

        self.assertEqual(report["malformed_row_count"], 0)
        self.assertEqual(report["invalid_date_count"], 1)
        self.assertEqual(report["possible_shifted_column_row_count"], 1)
        self.assertNotIn("ticket-1", json.dumps(report, ensure_ascii=False))

    def test_raw_csv_report_rejects_unterminated_quoted_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.csv"
            output = Path(directory) / "report.json"
            path.write_text(
                "application_number,creation_date,category,service\n"
                "ticket-1,01.02.2025 12:30:00,Дороги,Ремонт\n"
                'ticket-2,02.02.2025 12:30:00,Дороги,"PII-SENTINEL\n',
                encoding="utf-8",
            )
            with self.assertRaises(csv.Error):
                build_report(path)
            command = [sys.executable, str(ROOT / "scripts/data_audit.py"), str(path), "--output", str(output)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())
            self.assertIn("CSV_PARSE_FAILED", result.stderr)
            self.assertNotIn("PII-SENTINEL", result.stdout + result.stderr)


class MigrationTests(unittest.TestCase):
    def test_postgres_migrations_include_source_of_truth_tables(self) -> None:
        schema = (ROOT / "migrations" / "001_data_foundation.sql").read_text(encoding="utf-8").lower()
        seed = (ROOT / "migrations" / "002_seed_data_taxonomy.sql").read_text(encoding="utf-8").lower()
        for table in ("regions", "topics", "topic_source_mappings", "tickets", "ticket_predictions", "operator_decisions", "quarantine_rows", "dataset_versions", "model_versions"):
            self.assertIn(f"create table if not exists {table}", schema)
        self.assertIn("insert into regions", seed)
        self.assertIn("insert into topics", seed)
        self.assertIn("service_other", seed)

    def test_pulse_state_migration_covers_learning_and_alerts(self) -> None:
        state = (ROOT / "migrations" / "003_pulse_state.sql").read_text(encoding="utf-8").lower()
        for table in (
            "response_templates",
            "alerts",
            "alert_ticket_links",
            "relation_feedback",
            "learning_cycles",
            "learning_feedback",
            "model_evaluations",
            "background_jobs",
            "audit_log",
        ):
            self.assertIn(f"create table if not exists {table}", state)
        self.assertIn("operator_confirmed_decision", state)
        self.assertIn("production_model_version", state)


if __name__ == "__main__":
    unittest.main()
