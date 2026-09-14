from __future__ import annotations

import json

import pytest
import requests

from trialmatchai.matching.eligibility_reasoning_deepseek import (
    BatchTrialProcessorDeepSeek,
)


VALID_ASSESSMENT = {
    "Inclusion_Criteria_Evaluation": [
        {"Criterion": "Age >= 18", "Classification": "Met", "Justification": "Age 42."}
    ],
    "Exclusion_Criteria_Evaluation": [],
    "Final Decision": "Eligible",
}


class FakeResponse:
    def __init__(self, body=None, status=200):
        self.body = (
            body
            if body is not None
            else {"choices": [{"message": {"content": json.dumps(VALID_ASSESSMENT)}}]}
        )
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self.body


def _processor(**overrides):
    settings = {
        "base_url": "https://api.deepseek.com/",
        "model": "deepseek-v4-pro",
        "timeout_seconds": 12,
        "max_tokens": 5000,
        "temperature": 0.0,
        "max_retries": 3,
    }
    settings.update(overrides)
    return BatchTrialProcessorDeepSeek(
        api_key="secret-test-key",
        settings=settings,
        batch_size=1,
        use_cot=False,
    )


def test_request_and_persistence(tmp_path, monkeypatch):
    proc = _processor()
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return FakeResponse()

    monkeypatch.setattr(proc.session, "post", post)
    proc._process_batch(
        [{"nct_id": "NCT1", "prompt": "synthetic prompt"}], str(tmp_path)
    )

    assert calls == [
        (
            "https://api.deepseek.com/chat/completions",
            {
                "headers": {
                    "Authorization": "Bearer secret-test-key",
                    "Content-Type": "application/json",
                },
                "json": {
                    "model": "deepseek-v4-pro",
                    "messages": [{"role": "user", "content": "synthetic prompt"}],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 5000,
                    "temperature": 0.0,
                    "stream": False,
                },
                "timeout": 12.0,
            },
        )
    ]
    assert json.loads((tmp_path / "NCT1.json").read_text()) == VALID_ASSESSMENT
    assert "Final Decision" in (tmp_path / "NCT1.txt").read_text()


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse({"choices": [{"message": {"content": ""}}]}),
        FakeResponse({"choices": [{"message": {"content": "not json"}}]}),
        FakeResponse(
            {"choices": [{"message": {"content": '{"Final Decision":"Eligible"}'}}]}
        ),
    ],
)
def test_invalid_responses_are_safe_failures(tmp_path, monkeypatch, caplog, response):
    proc = _processor(max_retries=1)
    monkeypatch.setattr(proc.session, "post", lambda *a, **kw: response)
    proc._process_batch(
        [{"nct_id": "NCT2", "prompt": "PRIVATE SYNTHETIC PATIENT"}], str(tmp_path)
    )

    saved = (tmp_path / "NCT2.json").read_text()
    assert json.loads(saved) == {"error": "deepseek_api_error"}
    assert "secret-test-key" not in saved + caplog.text
    assert "PRIVATE SYNTHETIC PATIENT" not in saved + caplog.text


@pytest.mark.parametrize(
    "failure", [requests.Timeout(), FakeResponse(status=429), FakeResponse(status=500)]
)
def test_transient_failures_retry_and_recover(tmp_path, monkeypatch, failure):
    proc = _processor()
    outcomes = [failure, FakeResponse()]

    def post(*args, **kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(proc.session, "post", post)
    proc._process_batch([{"nct_id": "NCT3", "prompt": "patient"}], str(tmp_path))

    assert json.loads((tmp_path / "NCT3.json").read_text()) == VALID_ASSESSMENT
    assert outcomes == []
