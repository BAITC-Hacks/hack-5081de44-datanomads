"""Small offline worker entrypoint for ML jobs.

The production orchestration persists job state in PostgreSQL and may invoke
this module from the same image.  This local worker keeps the boundary useful
without introducing Celery/Redis: it accepts one JSON job from stdin and
prints one JSON result to stdout.

Examples::

    echo '{"kind":"training","model_type":"classifier","samples":[]}' | \
      python worker.py
    echo '{"kind":"evaluation","model_type":"forecast","values":[1,2,3,4,5,6,7,8]}' | \
      python worker.py
"""

from __future__ import annotations

import argparse
import json
import sys

from app.schemas import EvaluationRequest, TrainingRequest
from app.services import make_services


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Pulse 109 offline ML job")
    parser.add_argument("--kind", choices=("training", "evaluation"), help="override job kind in input")
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        kind = args.kind or payload.pop("kind", "evaluation")
        _, _, _, _, _, trainer, evaluator = make_services()
        if kind == "training":
            result = trainer.train(TrainingRequest.model_validate(payload))
        elif kind == "evaluation":
            result = evaluator.evaluate(EvaluationRequest.model_validate(payload))
        else:
            raise ValueError(f"unsupported job kind: {kind}")
        json.dump(result.model_dump(mode="json"), sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:  # pragma: no cover - CLI failure path
        json.dump({"error": str(exc)}, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

