from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import TrainingRequest
from app.services import test_fake_trainer_enabled as _fake_trainer_enabled
from worker import safe_job_error


def test_worker_error_code_does_not_include_job_payload() -> None:
    marker = "SENSITIVE_TICKET_TEXT_123"
    with pytest.raises(ValidationError) as caught:
        TrainingRequest.model_validate({"min_samples": marker})

    assert marker in str(caught.value)
    assert safe_job_error(caught.value) == "INVALID_JOB_PAYLOAD"
    assert safe_job_error(RuntimeError(marker)) == "JOB_FAILED"
    assert safe_job_error(RuntimeError("TRAINER_NOT_CONFIGURED")) == "TRAINER_NOT_CONFIGURED"


def test_fake_trainer_is_limited_to_demo_and_test_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PULSE_TEST_FAKE_TRAINER", "true")
    monkeypatch.setenv("PULSE_ENV", "production")
    assert not _fake_trainer_enabled()
    assert safe_job_error(RuntimeError("TEST_FAKE_TRAINER_NOT_ALLOWED")) == "TEST_FAKE_TRAINER_NOT_ALLOWED"

    monkeypatch.setenv("PULSE_ENV", "demo")
    assert _fake_trainer_enabled()
