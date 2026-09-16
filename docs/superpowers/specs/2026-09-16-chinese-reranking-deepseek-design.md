# Chinese Reranking and DeepSeek Eligibility Design

## Goal

Upgrade the current Chinese patient-to-trial pipeline from retrieval-only ranking to a two-stage decision flow:

1. locally rerank retrieved eligibility criteria with `Qwen/Qwen3-Reranker-0.6B`;
2. assess the shortlisted trials with the existing DeepSeek API backend.

Medical entity extraction remains on the deterministic regex backend for this phase. GLiNER2 and UIE are explicitly out of scope.

## Current State

- `BAAI/bge-m3` is downloaded and working locally on CPU.
- Twenty ClinicalTrials.gov lung-cancer trials and 670 criteria are indexed in LanceDB.
- Chinese patient retrieval and HTML reporting work end to end.
- Entity extraction uses the regex backend.
- LLM reranking and eligibility assessment are disabled in `config.mac.json`.
- The DeepSeek backend and API credential are already configured through environment settings.
- The machine is an Apple M4 MacBook Air with 32 GB unified memory and PyTorch MPS support.

## Selected Architecture

```text
Chinese patient narrative
        |
        v
Regex entity and constraint extraction
        |
        v
BGE-M3 + BM25 hybrid trial/criterion retrieval
        |
        v
Qwen3-Reranker-0.6B local relevance scoring
        |
        v
DeepSeek criterion-level eligibility assessment
        |
        v
Eligibility-first final ranking and HTML/JSON report
```

The reranker only decides relevance between patient text and retrieved criteria. It does not decide eligibility. DeepSeek remains responsible for `Met`, `Not Met`, `Unclear`, and `Irrelevant` criterion classifications.

## Components

### Qwen3 reranker

Add Qwen3-specific prompt construction and Yes/No token scoring to the existing Transformers reranker path. Keep the generic reranker interface (`rank_pairs`) unchanged so retrieval code does not become model-specific.

Runtime requirements:

- model: `Qwen/Qwen3-Reranker-0.6B`;
- backend: `transformers`;
- preferred device: Apple MPS;
- safe fallback: CPU;
- initial batch size: 2, increased only after a successful smoke test;
- local cached inference after the initial model download.

The implementation must use the model's documented reranking template and token scoring rather than assuming the existing Gemma prompt is equivalent.

### DeepSeek eligibility assessment

Use the existing `deepseek_api` backend without adding a second local eligibility model. Before an end-to-end run, perform a synthetic-data live check to verify that the configured endpoint and model name are accepted. Never log or persist the API key.

Initial controls:

- assess at most six shortlisted trials;
- batch size 1 or the backend's existing safe default;
- deterministic temperature;
- direct structured JSON output, with failures retained as retryable rather than treated as ineligible.

### Entity extraction

Keep `entity_extraction.backend=regex`. No GLiNER2, UIE, or DeepSeek extraction changes are included in this phase. The full patient narrative still reaches retrieval and eligibility assessment, so the regex output is supplementary rather than the sole source of evidence.

## Configuration

Create Mac-specific settings in `config.mac.json`:

- enable `LLM_reranker`;
- set its backend to `transformers` and batch size to 2;
- set `model.reranker_model_path` to `Qwen/Qwen3-Reranker-0.6B`;
- set the reranker runtime device to `mps` through a dedicated reranker setting rather than reusing the CUDA-oriented numeric global device;
- enable `rag` with backend `deepseek_api` and cap assessment at six trials;
- retain regex entity extraction, disabled query expansion, and the existing BGE-M3 index.

No Phi-4, Gemma, GLiNER2, or UIE weights will be downloaded.

## Data Flow and Privacy

BGE-M3 retrieval and Qwen reranking run locally. For shortlisted trials only, the patient narrative and trial eligibility criteria are sent to the configured DeepSeek service. Local outputs record model/endpoint identity and assessment results but never the API key.

The first live validation uses only the existing synthetic patient. Real patient data is not required for acceptance testing.

## Error Handling

- If MPS cannot execute a Qwen operation, retry the reranker on CPU and record the selected device.
- If the Qwen download is interrupted, preserve the Hugging Face cache and resume rather than starting a duplicate download.
- If the DeepSeek endpoint rejects the configured model name, stop after the synthetic check and report the exact non-secret API error.
- If an individual DeepSeek assessment fails, retain it as pending/failed; do not convert it into a negative eligibility verdict.
- Retrieval-only results remain available if either optional stage fails.

## Verification

1. Unit tests for Qwen prompt formatting, Yes/No token IDs, score ordering, device selection, and CPU fallback.
2. Existing reranker, preflight, configuration, assessment, and report tests remain green.
3. Offline model-load smoke test after the Qwen download.
4. Synthetic pair test where a clearly relevant lung-cancer criterion outranks an unrelated criterion.
5. Synthetic DeepSeek live test with no real patient information.
6. Full synthetic end-to-end run against the current 20-trial local registry.
7. Verify that the final run metadata says `eligibility_assessment`, contains assessed trial IDs, and produces JSON and HTML reports.

## Acceptance Criteria

- The Qwen reranker loads on this Mac and assigns finite relevance scores.
- The relevant synthetic criterion ranks above the unrelated criterion.
- DeepSeek returns at least one schema-valid criterion-level assessment.
- End-to-end matching completes without loading Phi-4, Gemma, GLiNER2, or UIE.
- Results clearly distinguish retrieval relevance from eligibility verdicts.
- No real patient data is used during implementation or validation.

## Out of Scope

- Chinese medical NER replacement or fine-tuning.
- Re-embedding the registry with another embedding model.
- Downloading the full ClinicalTrials.gov corpus.
- Clinical validation, diagnostic claims, or autonomous enrollment decisions.
