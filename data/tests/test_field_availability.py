from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_field_availability import HEADERS, build_report, scan_profile


def write_csv(path: Path, profile: str, rows: list[dict]) -> None:
    headers = sorted(HEADERS[profile])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


class FieldAvailabilityTests(unittest.TestCase):
    def test_customer_facts_are_counts_without_values_or_fabricated_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vko = root / "vko.csv"
            almaty = root / "almaty.csv"
            write_csv(vko, "vko_109", [
                {"application_number": "1", "creation_date": "01.01.2025 10:00:00",
                 "category": "roads", "service": "unverified", "region": "city",
                 "district": "district", "street": "PII-STREET-SENTINEL",
                 "full_name": "PII-NAME-SENTINEL", "com_exp": "PII-TEXT-SENTINEL"},
                {"application_number": "1", "creation_date": "bad-date", "category": "roads"},
            ])
            write_csv(almaty, "almaty_109", [
                {"application_number": "2", "creation_date": "02.01.2025 10:00:00",
                 "category": "water", "service": "issue-type", "status_1": "raw-status"},
            ])
            report = build_report(vko, almaty)
            self.assertEqual(build_report(vko, almaty), report)
            vko_report, almaty_report = report["profiles"]
            self.assertEqual(vko_report["record_count"], 2)
            self.assertEqual(vko_report["duplicate_source_id_count"], 1)
            self.assertEqual(vko_report["invalid_creation_date_count"], 1)
            self.assertEqual(vko_report["joint_presence"]["created_and_category"], 1)
            self.assertEqual(vko_report["fields"]["original_appeal_text"]["present_count"], 0)
            self.assertEqual(almaty_report["fields"]["raw_service"]["present_count"], 1)
            self.assertIsNone(almaty_report["fields"]["region_hint"]["source_column"])
            self.assertEqual(almaty_report["fields"]["region_hint"]["semantic_status"], "ABSENT")
            self.assertEqual(almaty_report["fields"]["raw_service"]["semantic_status"],
                             "OBSERVED_ISSUE_TYPE_NOT_VERIFIED_EXECUTOR")
            self.assertEqual(report["required_follow_up_questions"], [])
            self.assertEqual(set(report["cross_source_value_comparison"]), set(vko_report["fields"]))
            encoded = json.dumps(report, ensure_ascii=False)
            for sentinel in ("PII-STREET-SENTINEL", "PII-NAME-SENTINEL", "PII-TEXT-SENTINEL"):
                self.assertNotIn(sentinel, encoded)

    def test_changed_header_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.csv"
            write_csv(path, "vko_109", [{"application_number": "1"}])
            content = path.read_text(encoding="utf-8")
            path.write_text(content.replace("creation_date", "created_at", 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema differs"):
                scan_profile(path, "vko_109")


if __name__ == "__main__":
    unittest.main()
