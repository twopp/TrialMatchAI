# DeepSeek Online Eligibility Assessment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a DeepSeek online backend to TrialMatchAI's criterion-level eligibility assessment while preserving existing output, resume, and ranking behavior.

**Architecture:** A focused `BatchTrialProcessorDeepSeek` subclasses `BaseTrialProcessor`, sends existing prompts to DeepSeek's OpenAI-compatible chat-completions endpoint, and persists responses through the existing result path. Pydantic configuration and environment overrides select the backend and hold non-secret settings; `DEEPSEEK_API_KEY` remains environment-only.

**Tech Stack:** Python 3.11, requests, tenacity, Pydantic 2, pytest

**Spec:** `docs/superpowers/specs/2026-09-14-deepseek-api-design.md`

## Global Constraints

- Only the eligibility assessment stage gains the online backend.
- Retrieval, reranking, scoring, reporting, and local model behavior remain unchanged.
- Read the secret only from `DEEPSEEK_API_KEY`; never serialize or log it.
- Use the existing direct JSON or reasoning prompt and existing per-trial output shape.
- Default endpoint is `https://api.deepseek.com` and default model is `deepseek-v4-pro`.

---

### Task 1: Configuration and backend selection

**Files:**
- Modify: `src/trialmatchai/config/settings.py`
- Modify: `src/trialmatchai/config/config.json`
- Modify: `src/trialmatchai/matching/assessment.py`
- Modify: `src/trialmatchai/services/preflight.py`
- Modify: `src/trialmatchai/main.py`
- Test: `tests/test_config_settings.py`
- Test: `tests/test_preflight.py`
- Test: `tests/test_assessment_controls.py`

**Interfaces:**
- Produces: `DeepSeekAPISettings` with `base_url`, `model`, `timeout_seconds`, `max_tokens`, `temperature`, and `max_retries`.
- Produces: `rag.backend` accepting `"deepseek_api"`.
- Consumes later: `config["deepseek_api"]` and `os.environ["DEEPSEEK_API_KEY"]`.

- [ ] **Step 1: Add failing settings tests**

Add tests that load a config with `rag.backend = "deepseek_api"`, assert the six DeepSeek settings and defaults survive validation, and assert environment overrides update base URL and model without accepting an API key in the config object.

- [ ] **Step 2: Run the focused tests and verify the backend is rejected**

Run: `pytest -q tests/test_config_settings.py tests/test_assessment_controls.py tests/test_preflight.py`

Expected: failure because `RagSettings.backend` rejects `deepseek_api` or `deepseek_api` settings are absent.

- [ ] **Step 3: Implement settings, environment overrides, provenance, and preflight**

Add:

```python
class DeepSeekAPISettings(BaseModel):
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-pro"
    timeout_seconds: float = Field(60.0, gt=0)
    max_tokens: int = Field(5000, ge=1)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_retries: int = Field(3, ge=1)
```

Add `deepseek_api` to the RAG backend literal, attach the settings to `TrialMatchSettings`, add non-secret environment overrides, include model/endpoint identity in assessment settings, and require a non-empty `DEEPSEEK_API_KEY` during online-backend preflight without logging its value.

- [ ] **Step 4: Wire backend construction in `run_rag_processing`**

Instantiate:

```python
BatchTrialProcessorDeepSeek(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    settings=config["deepseek_api"],
    batch_size=config["rag"].get("batch_size", 1),
    use_cot=config.get("use_cot_reasoning", True),
    no_think=config["rag"].get("no_think", False),
)
```

Keep the current `transformers` and `vllm` branches unchanged.

- [ ] **Step 5: Run focused tests and commit**

Run: `pytest -q tests/test_config_settings.py tests/test_assessment_controls.py tests/test_preflight.py`

Expected: all focused tests pass.

Commit: `feat: configure DeepSeek eligibility backend`

---

### Task 2: DeepSeek processor and response validation

**Files:**
- Create: `src/trialmatchai/matching/eligibility_reasoning_deepseek.py`
- Test: `tests/test_eligibility_deepseek.py`

**Interfaces:**
- Consumes: `BaseTrialProcessor._format_prompt`, `_strip_thinking_tags`, and its work-list batching contract.
- Produces: `BatchTrialProcessorDeepSeek._process_batch(batch: list[dict], output_folder: str) -> None`.
- Produces: per-trial `.txt` raw output and `.json` parsed assessment files.

- [ ] **Step 1: Write failing request and persistence tests**

Mock `requests.Session.post` and assert the request uses:

```python
{
    "model": "deepseek-v4-pro",
    "messages": [{"role": "user", "content": prompt}],
    "response_format": {"type": "json_object"},
    "max_tokens": 5000,
    "temperature": 0.0,
    "stream": False,
}
```

Assert the bearer header is present, the raw response is written, and the parsed JSON contains inclusion/exclusion evaluations and `Final Decision`.

- [ ] **Step 2: Run the processor test and verify import failure**

Run: `pytest -q tests/test_eligibility_deepseek.py`

Expected: failure because `eligibility_reasoning_deepseek` does not exist.

- [ ] **Step 3: Implement request transport**

Create a processor that normalizes `base_url`, posts to `<base_url>/chat/completions`, sets JSON content and bearer authorization headers, uses the configured timeout, and calls `raise_for_status()`.

- [ ] **Step 4: Implement strict response validation and safe persistence**

Validate all of the following before a response counts as successful:

```python
content = body["choices"][0]["message"]["content"]
parsed = extract_json_object(self._strip_thinking_tags(content))
assert isinstance(parsed, dict)
assert isinstance(parsed.get("Inclusion_Criteria_Evaluation"), list)
assert isinstance(parsed.get("Exclusion_Criteria_Evaluation"), list)
assert isinstance(parsed.get("Final Decision"), str)
```

Write failures with the existing retryable error-output convention. Do not include patient prompts, authorization headers, or response bodies in log messages.

- [ ] **Step 5: Add retry and secrecy tests**

Test timeout, HTTP 429, HTTP 500, empty content, malformed JSON, and recovery on a later attempt. Capture logs and serialized outputs and assert neither the API key nor the synthetic patient narrative appears.

- [ ] **Step 6: Run tests and commit**

Run: `pytest -q tests/test_eligibility_deepseek.py tests/test_eligibility_base.py`

Expected: all tests pass.

Commit: `feat: assess trial eligibility with DeepSeek API`

---

### Task 3: Documentation and end-to-end verification

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Create: `tests/test_deepseek_live.py`

**Interfaces:**
- Produces: documented `.env` and `config.json` setup.
- Produces: optional live test selected with `TRIALMATCHAI_RUN_DEEPSEEK_LIVE=1`.

- [ ] **Step 1: Add environment and README instructions**

Document:

```dotenv
DEEPSEEK_API_KEY=
TRIALMATCHAI_DEEPSEEK_BASE_URL=https://api.deepseek.com
TRIALMATCHAI_DEEPSEEK_MODEL=deepseek-v4-pro
```

Show `rag.backend: "deepseek_api"`, explain that `.env` is ignored, and provide a synthetic-data invocation that reuses the normal TrialMatchAI pipeline.

- [ ] **Step 2: Add an opt-in live smoke test**

The test skips unless both `DEEPSEEK_API_KEY` and `TRIALMATCHAI_RUN_DEEPSEEK_LIVE=1` are present. It sends a synthetic adult-patient prompt with one age inclusion criterion and asserts a non-empty, schema-valid assessment without printing the response or secret.

- [ ] **Step 3: Run formatting and relevant test suites**

Run:

```bash
ruff check src/trialmatchai tests
pytest -q tests/test_eligibility_deepseek.py tests/test_config_settings.py tests/test_assessment_controls.py tests/test_preflight.py
```

Expected: lint and focused tests pass.

- [ ] **Step 4: Run the live smoke test**

Run: `TRIALMATCHAI_RUN_DEEPSEEK_LIVE=1 pytest -q tests/test_deepseek_live.py`

Expected: one live test passes and produces no key or patient narrative in output.

- [ ] **Step 5: Run the full suite and inspect the diff**

Run:

```bash
pytest -q
git diff --check
git status --short
```

Expected: full suite passes; only intended implementation, tests, and documentation are changed.

- [ ] **Step 6: Commit**

Commit: `docs: explain DeepSeek online assessment`
