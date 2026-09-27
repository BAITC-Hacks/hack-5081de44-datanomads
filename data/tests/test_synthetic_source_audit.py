from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_synthetic_sources import build_report
from scripts.generate_synthetic_sources import generate


class SyntheticSourceAuditTests(unittest.TestCase):
    def test_all_sources_audited_and_checksum_tampering_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = generate(root, check=True)
            report = build_report(root / "manifest.json")
            committed = json.loads(
                (Path(__file__).resolve().parents[1] / "reports/synthetic_source_quality_v1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report, committed)
            self.assertTrue(report["synthetic"])
            self.assertEqual(report["source_count"], 7)
            self.assertEqual(report["totals"]["file_count"], 28)
            self.assertEqual(report["totals"]["valid_ticket_count"], 35)
            self.assertEqual(report["totals"]["quarantine_count"], 35)
            self.assertEqual(report["totals"]["pii_redacted_ticket_count"], 7)
            self.assertEqual(set(report["totals"]["quarantine_reasons"].values()), {7})
            serialized = json.dumps(report, ensure_ascii=False)
            self.assertNotIn("synthetic@example.invalid", serialized)
            self.assertNotIn("У дома не горит фонарь", serialized)

            primary = root / manifest["sources"][0]["files"]["primary"]["path"]
            with primary.open("a", encoding="utf-8") as stream:
                stream.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                build_report(root / "manifest.json")


if __name__ == "__main__":
    unittest.main()
