# DeepSeek Online Eligibility Assessment Design

## Goal

Add a DeepSeek online backend to TrialMatchAI's existing criterion-level eligibility assessment stage. The first version is intended to test whether TrialMatchAI produces useful structured results for supplied patient and trial data. Retrieval, scoring, reporting, and other business rules remain unchanged.

## Configuration

The backend is selected with `rag.backend: "deepseek_api"`. Non-secret settings live under a new `deepseek_api` configuration section:

- `base_url`, defaulting to `https://api.deepseek.com`
- `model`, defaulting to `deepseek-v4-pro`
- `timeout_seconds`
- `max_tokens`
- `temperature`
- `max_retries`

The API key is read only from `DEEPSEEK_API_KEY`. It is never accepted in `config.json`, logged, persisted in results, or committed. `.env` remains ignored by Git.

## Architecture

A new `BatchTrialProcessorDeepSeek` implements the same `BaseTrialProcessor` contract as the existing Transformers and vLLM processors. This preserves prompt construction, trial work lists, output files, resume behavior, and downstream ranking.

`run_rag_processing` selects the processor when the configured backend is `deepseek_api`. Local model backends continue to behave as before. The DeepSeek processor calls the OpenAI-compatible chat-completions endpoint with JSON output enabled, parses the response through the existing JSON extraction path, and writes the same per-trial JSON shape used by the rest of TrialMatchAI.

The implementation uses the project's existing `requests` and `tenacity` dependencies. No additional runtime package is required.

## Data Flow

For every shortlisted trial:

1. Load the trial's eligibility criteria through the existing loader.
2. Build the existing direct-JSON or reasoning prompt with the patient narrative.
3. Send the prompt to DeepSeek with JSON output requested.
4. Reject missing, empty, truncated, malformed, or off-shape responses.
5. Retry transient failures with bounded exponential backoff.
6. Save successful raw text and parsed JSON through the existing result flow.
7. Leave failed assessments retryable on resume.

## Error Handling

Configuration validation fails before inference when `DEEPSEEK_API_KEY` is absent. Authentication and non-retryable request errors include a concise diagnostic without response secrets. Timeouts, rate limits, server errors, and empty responses are retried up to the configured limit. Logs identify the trial and failure category without including patient text, the API key, or full provider responses.

## Testing

Unit tests mock all network calls and verify:

- request URL, authorization header presence, model, and JSON mode
- successful criterion-level JSON persistence
- retry behavior for timeouts, rate limits, server failures, and empty content
- permanent failure for missing keys and invalid responses
- unchanged selection of Transformers and vLLM backends
- API keys and patient text are absent from logs and result metadata

An optional live smoke test runs only when `DEEPSEEK_API_KEY` is present. It uses synthetic patient and trial data and is excluded from the default test suite.

## Documentation

The environment example and README document how to select `deepseek_api`, store the key in `.env`, run the test flow, and recognize generated result files. The documentation warns users not to commit keys or send identifiable patient data during early testing.

## Out of Scope

This change does not add an HTTP product service, redesign input schemas, change scoring, add manual-review rules, replace retrieval or reranking models, or validate clinical accuracy.
