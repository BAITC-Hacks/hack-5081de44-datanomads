from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from data.normalization.pii import scan_pii
from data.schemas.unified_ticket import UnifiedTicket
from scripts.data_audit import build_normalized_report
from scripts.build_synthetic_source_corpus import DEFAULT_QUALITY_REPORT, build_corpus
from scripts.generate_synthetic_sources import generate


REFERENCE_MANIFEST = Path(__file__).resolve().parents[1] / "manifests/synthetic-source-normalized-v1.json"


class SyntheticSourceCorpusTests(unittest.TestCase):
    def test_corpus_is_reproducible_private_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            generate(raw, check=True)
            package = root / "package"
            manifest = build_corpus(raw / "manifest.json", DEFAULT_QUALITY_REPORT, package)
            self.assertEqual(manifest, json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8")))
            self.assertFalse(manifest["approved_for_training"])
            self.assertEqual((manifest["record_count"], manifest["quarantine_count"]), (35, 35))
            for name, info in manifest["files"].items():
                content = (package / name).read_bytes()
                self.assertEqual(len(content), info["bytes"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), info["sha256"])
            tickets = [json.loads(line) for line in (package / "tickets.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(tickets), 35)
            self.assertTrue(all(UnifiedTicket.from_mapping(row).to_dict() == row for row in tickets))
            self.assertTrue(all(not scan_pii(row["original_text"]).detected for row in tickets))
            normalized_audit = build_normalized_report(package / "tickets.jsonl")
            self.assertEqual(normalized_audit["valid_record_count"], 35)
            self.assertEqual(normalized_audit["duplicate_external_id_count"], 0)
            quarantine = (package / "quarantine.jsonl").read_text(encoding="utf-8")
            self.assertEqual(len(quarantine.splitlines()), 35)
            self.assertNotIn("synthetic@example.invalid", quarantine)
            self.assertNotIn("synthetic@example.invalid", (package / "tickets.jsonl").read_text(encoding="utf-8"))

            saved_manifest = (package / "manifest.json").read_bytes()
            with self.assertRaises(FileExistsError):
                build_corpus(raw / "manifest.json", DEFAULT_QUALITY_REPORT, package)
            self.assertEqual((package / "manifest.json").read_bytes(), saved_manifest)

            reformatted = root / "reformatted-quality.json"
            reformatted.write_bytes(DEFAULT_QUALITY_REPORT.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "quality report differs"):
                build_corpus(raw / "manifest.json", reformatted, root / "unverified")
            self.assertFalse((root / "unverified").exists())

            primary = raw / "ikomek109/primary.csv"
            with primary.open("a", encoding="utf-8") as stream:
                stream.write("tampered\n")
            blocked = root / "blocked"
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                build_corpus(raw / "manifest.json", DEFAULT_QUALITY_REPORT, blocked)
            self.assertFalse(blocked.exists())


if __name__ == "__main__":
    unittest.main()
