"""DeepSeek online backend for criterion-level eligibility assessment."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping

import requests
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from trialmatchai.matching.eligibility_base import BaseTrialProcessor
from trialmatchai.utils.file_utils import write_json_file
from trialmatchai.utils.json_utils import extract_json_object
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)


class DeepSeekResponseError(ValueError):
    """The provider returned a response that cannot be used as an assessment."""


class BatchTrialProcessorDeepSeek(BaseTrialProcessor):
    def __init__(
        self,
        *,
        api_key: str,
        settings: Mapping[str, Any],
        batch_size: int = 1,
        use_cot: bool = True,
        no_think: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("DEEPSEEK_API_KEY is empty")
        self.api_key = api_key
        self.base_url = str(
            settings.get("base_url", "https://api.deepseek.com")
        ).rstrip("/")
        self.model = str(settings.get("model", "deepseek-v4-pro"))
        self.timeout_seconds = float(settings.get("timeout_seconds", 60.0))
        self.max_tokens = int(settings.get("max_tokens", 5000))
        self.temperature = float(settings.get("temperature", 0.0))
        self.max_retries = int(settings.get("max_retries", 3))
        self.batch_size = int(batch_size)
        self.use_cot = bool(use_cot)
        self.no_think = bool(no_think)
        self.length_bucket = False
        self.session = session or requests.Session()

    def _progress_desc(self) -> str:
        return "Assessing Trials with DeepSeek"

    def _request(self, prompt: str) -> str:
        response = self.session.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False,
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise DeepSeekResponseError("missing response content") from exc
        if not isinstance(content, str) or not content.strip():
            raise DeepSeekResponseError("empty response content")
        return content

    def _validated_response(self, prompt: str) -> str:
        response = self._request(prompt)
        try:
            parsed = extract_json_object(self._strip_thinking_tags(response))
        except (json.JSONDecodeError, ValueError) as exc:
            raise DeepSeekResponseError("invalid JSON assessment") from exc
        if not isinstance(parsed.get("Inclusion_Criteria_Evaluation"), list):
            raise DeepSeekResponseError("missing inclusion evaluation")
        if not isinstance(parsed.get("Exclusion_Criteria_Evaluation"), list):
            raise DeepSeekResponseError("missing exclusion evaluation")
        if (
            not isinstance(parsed.get("Final Decision"), str)
            or not parsed["Final Decision"].strip()
        ):
            raise DeepSeekResponseError("missing final decision")
        return response

    def _request_with_retry(self, prompt: str) -> str:
        retrying = Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=0.1, min=0.1, max=1.0),
            retry=retry_if_exception_type(
                (requests.RequestException, DeepSeekResponseError)
            ),
            reraise=True,
        )
        return retrying(self._validated_response, prompt)

    def _process_batch(self, batch: List[Dict], output_folder: str) -> None:
        for item in batch:
            nct_id = str(item["nct_id"])
            try:
                response = self._request_with_retry(str(item["prompt"]))
                self._save_outputs(nct_id, response, output_folder)
            except (requests.RequestException, DeepSeekResponseError):
                logger.error("DeepSeek assessment failed for %s after retries", nct_id)
                write_json_file(
                    {"error": "deepseek_api_error"},
                    f"{output_folder}/{nct_id}.json",
                )
