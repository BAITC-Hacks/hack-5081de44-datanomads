"""Verify the deterministic demo seed and guarded local reset command."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SEED_SCRIPT = ROOT / "scripts" / "seed_demo.py"
RESET_SCRIPT = ROOT / "scripts" / "demo-reset"


def load_seed_module():
    spec = importlib.util.spec_from_file_location("pulse109_seed_demo", SEED_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SEED_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DemoBootstrapTests(unittest.TestCase):
    def test_seed_payload_is_deterministic_and_repeat_import_is_accepted(self) -> None:
        seed_demo = load_seed_module()
        manifest_path = ROOT / "data" / "manifests" / "demo-2026-09-21.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record_count = manifest["record_count"]
        responses = iter(
            (
                {"imported_rows": record_count, "duplicate_rows": 0, "indexed_rows": record_count},
                {"imported_rows": 0, "duplicate_rows": record_count, "indexed_rows": record_count},
            )
        )
        requests = []
        events = []
        output = io.StringIO()

        class JsonResponse(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_):
                self.close()

        def urlopen(request, timeout):
            self.assertEqual(timeout, 180)
            self.assertEqual(request.get_method(), "POST")
            requests.append(request)
            events.append("import")
            return JsonResponse(json.dumps(next(responses)).encode("utf-8"))

        with patch.dict(os.environ, {"PULSE_ENV": "demo"}):
            with patch.object(seed_demo, "DATA_ROOT", ROOT / "data"):
                with patch.object(seed_demo, "MANIFEST_PATH", manifest_path):
                    with patch.object(seed_demo, "CORE_URL", "http://core-api:8080"):
                        with patch.object(
                            seed_demo,
                            "wait_for_core",
                            side_effect=lambda: events.append("ready"),
                        ):
                            with patch.object(seed_demo, "urlopen", side_effect=urlopen):
                                with redirect_stdout(output):
                                    seed_demo.main()
                                    seed_demo.main()

        self.assertEqual(events, ["ready", "import", "ready", "import"])
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0].data, requests[1].data)
        payload = json.loads(requests[0].data)
        self.assertTrue(payload["is_synthetic"])
        self.assertEqual(payload["dataset_version"], manifest["dataset_version"])
        self.assertEqual(len(payload["tickets"]), record_count)
        results = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(results[0]["imported_rows"], record_count)
        self.assertEqual(results[1]["duplicate_rows"], record_count)

    def test_reset_requires_explicit_confirmation_before_docker(self) -> None:
        environment = os.environ.copy()
        environment.pop("PULSE_CONFIRM_RESET", None)
        result = subprocess.run(
            [str(RESET_SCRIPT)],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("PULSE_CONFIRM_RESET=1", result.stderr)
        self.assertIn("ML artifact volumes", result.stderr)


if __name__ == "__main__":
    unittest.main()
