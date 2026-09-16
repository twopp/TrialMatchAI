# Chinese Reranking and DeepSeek Eligibility Implementation Plan

**Goal:** Run local Qwen3 criterion reranking on Apple Silicon and DeepSeek eligibility assessment on synthetic data, without loading the legacy Gemma, Phi-4, or GLiNER2 models.

**Design:** `docs/superpowers/specs/2026-09-16-chinese-reranking-deepseek-design.md`

## Task 1: Specify Qwen3 reranker behavior

- Add focused tests for the official Qwen3 instruction/prefix/suffix format.
- Test lowercase `yes`/`no` token selection and normalized probability scoring.
- Test configured reranker device precedence and CPU fallback selection.
- Run the focused tests and confirm they fail for the missing behavior.

## Task 2: Implement the local Qwen3 reranker path

- Extend `TransformersReranker` with a Qwen3 mode while preserving the generic Gemma-compatible path.
- Implement official Qwen3 token construction, truncation, padding, and final-token scoring.
- Resolve `auto`, `mps`, and unavailable-device fallback safely.
- Allow the reranker device, model family, and maximum length to be configured under `LLM_reranker`.
- Run focused and existing retrieval/reranker tests.

## Task 3: Configure the Mac pipeline

- Point the Mac configuration at `Qwen/Qwen3-Reranker-0.6B`.
- Enable the Transformers reranker with a conservative batch size.
- Enable DeepSeek eligibility assessment for at most six trials.
- Retain regex entities, BGE-M3, and disabled query expansion.
- Add or update configuration tests where behavior is encoded in source.

## Task 4: Download and smoke-test Qwen3

- Download the model into the Hugging Face cache once.
- Load it locally with network access disabled.
- Score a relevant and an unrelated synthetic Chinese criterion.
- Verify finite scores and correct relative ordering; fall back to CPU if MPS is unsupported.

## Task 5: Validate DeepSeek with synthetic data

- Run preflight without exposing the API key.
- Send one synthetic patient/trial assessment through the existing backend.
- If the configured model alias is rejected, use a provider-supported model name and record the non-secret setting.
- Verify schema-valid assessment output.

## Task 6: Run and inspect the complete pipeline

- Re-run the synthetic Chinese patient against the current 20-trial registry.
- Confirm Qwen reranking and DeepSeek assessment both execute.
- Verify final metadata, assessed IDs, ranked JSON, and HTML report.
- Run the relevant automated test suite and healthcheck.
- Document model cache size, runtime mode, outputs, and any remaining limitations.
