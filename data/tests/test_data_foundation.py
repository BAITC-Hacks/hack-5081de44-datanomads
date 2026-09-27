from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from data.importers import IKOMEK109Importer, get_importer
from data.normalization import minimize_text, scan_pii
from data.normalization.pipeline import normalize_row
from data.schemas.taxonomy import REGION_DEFINITIONS, TOPIC_DEFINITIONS
from data.schemas.unified_ticket import UnifiedTicket
from scripts.data_audit import build_report, build_source_report
from scripts.generate_synthetic_sources import SPECS, generate as generate_synthetic_sources


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
        self.assertEqual(result.ticket.topic_id, "water_supply")
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


class DemoFixtureTests(unittest.TestCase):
    def test_manifest_and_fixture_coverage(self) -> None:
        manifest_path = ROOT / "data" / "manifests" / "demo-2026-09-21.json"
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
        for entry in manifest["files"]:
            path = ROOT / entry["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, entry["sha256"])

    def test_taxonomy_matches_fixture_contract(self) -> None:
        self.assertEqual(len(REGION_DEFINITIONS), 20)
        self.assertGreaterEqual(len(TOPIC_DEFINITIONS), 10)


class DataAuditTests(unittest.TestCase):
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
        encoded = json.dumps(report, ensure_ascii=False)
        for sensitive in ("Иван Иванов", "ФИО:", "ticket-1", "private@example.com", "secret"):
            self.assertNotIn(sensitive, encoded)


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
