from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from trialmatchai.matching.eligibility_reasoning_deepseek import (
    BatchTrialProcessorDeepSeek,
)
from trialmatchai.utils.json_utils import extract_json_object

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")


@pytest.mark.skipif(
    os.getenv("TRIALMATCHAI_RUN_DEEPSEEK_LIVE") != "1"
    or not os.getenv("DEEPSEEK_API_KEY", "").strip(),
    reason="live DeepSeek smoke test is opt-in",
)
def test_live_deepseek_returns_schema_valid_assessment():
    processor = BatchTrialProcessorDeepSeek(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        settings={
            "base_url": os.getenv("TRIALMATCHAI_DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "model": os.getenv("TRIALMATCHAI_DEEPSEEK_MODEL", "deepseek-v4-pro"),
            "timeout_seconds": 60,
            "max_tokens": 1200,
            "temperature": 0,
            "max_retries": 2,
        },
        batch_size=1,
        use_cot=False,
    )
    prompt = processor._format_prompt(
        "Inclusion Criteria:\n- Age 18 years or older\nExclusion Criteria:\n- Age below 18 years",
        "Synthetic patient. Age: 42 years.",
    )
    raw = processor._request_with_retry(prompt)
    parsed = extract_json_object(processor._strip_thinking_tags(raw))

    assert isinstance(parsed["Inclusion_Criteria_Evaluation"], list)
    assert isinstance(parsed["Exclusion_Criteria_Evaluation"], list)
    assert isinstance(parsed["Final Decision"], str)
    assert json.dumps(parsed)
