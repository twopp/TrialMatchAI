# UIE Medical Subprocess Integration Design

## Goal

Add Chinese medical entity extraction to the trial-registry preparation stage without installing PaddlePaddle or PaddleNLP into TrialMatchAI's existing PyTorch environment.

The selected model is PaddleNLP `uie-medical-base`, already downloaded and smoke-tested in `.venv-uie` on the target Apple Silicon Mac. The integration improves structured annotations for Chinese trial eligibility criteria; it does not replace BGE-M3 retrieval, Qwen reranking, or DeepSeek eligibility assessment.

## Current State

- The end-to-end patient-to-trial pipeline is working.
- `BAAI/bge-m3` performs local retrieval.
- `Qwen/Qwen3-Reranker-0.6B` performs local reranking.
- DeepSeek performs criterion-level eligibility assessment.
- Entity extraction currently uses deterministic regex rules.
- `.venv-uie` contains PaddlePaddle 3.3.0, PaddleNLP 3.0.0b4, and `uie-medical-base`.
- A local smoke test correctly extracted `非小细胞肺癌`, `奥希替尼`, and `EGFR L861Q`.
- Measured local performance was approximately 1.0 second to load the cached model and 2.7 seconds to extract six entity categories from one short sentence.

## Selected Architecture

Keep PaddleNLP isolated from the main `.venv`. TrialMatchAI starts one persistent worker process with `.venv-uie/bin/python` during registry preparation and communicates with it through newline-delimited JSON over standard input and output.

```text
Trial criteria preparation (main .venv)
        |
        | JSON batches
        v
Persistent UIE worker (.venv-uie, CPU)
        |
        | extracted spans + probabilities
        v
TrialMatchAI EntityAnnotation objects
        |
        +-- deterministic variant augmentation
        +-- optional concept linking
        v
Prepared criteria and LanceDB index
```

The worker loads `uie-medical-base` once and serves every trial processed by the build. It must not launch and reload the model once per criterion or once per trial.

## Components

### UIE subprocess recognizer

Add a `uie` entity backend implementing the existing `EntityRecognizer` protocol. It will:

- resolve the configured Python executable relative to the project directory;
- start the standalone worker lazily on the first extraction request;
- wait for a structured ready message;
- send batches of texts and extraction prompts as JSON;
- validate worker responses before converting them to `EntityAnnotation` values;
- keep the process alive across repeated `recognize` calls;
- terminate and reap the process when the recognizer is closed or the parent exits.

The main process must not import PaddlePaddle or PaddleNLP.

### Standalone worker

Add a worker script that imports only the Python standard library, PaddlePaddle, and PaddleNLP. It will:

- force CPU inference with FP32;
- load `uie-medical-base` from the existing local cache;
- accept JSON-lines requests;
- run batched information extraction;
- return original text spans, offsets, labels, and probabilities;
- return structured error messages rather than Python tracebacks on stdout.

Warnings and framework diagnostics go to stderr so they cannot corrupt the JSON protocol.

### Chinese prompt mapping

Extend the entity schema with an optional UIE prompt for every supported type. Initial prompts are:

| Schema ID | UIE prompt |
|---|---|
| `disease` | 疾病 |
| `gene` | 基因 |
| `medication` | 药物 |
| `procedure` | 治疗或医疗操作 |
| `diagnostic_test` | 诊断检查 |
| `laboratory_test` | 实验室检查或检查指标 |
| `radiology` | 影像检查或影像学发现 |
| `sign_symptom` | 症状或体征 |
| `cell_type` | 细胞类型 |
| `species` | 物种 |
| `variant` | 基因突变 |

Worker responses map back to the existing schema ID and entity group. UIE-specific labels must not leak into downstream interfaces.

### Confidence and overlaps

Use a UIE confidence threshold of 0.5 initially, matching PaddleNLP's documented/default span threshold and retaining the successful smoke-test drug extraction at 0.739. The value remains configurable.

Preserve overlapping spans when they represent different schema types, such as `EGFR L861Q` classified as both gene and genetic variant. Resolve duplicate or overlapping spans only within the same schema type. The deterministic genetic-variant recognizer remains enabled and exact duplicate annotations are deduplicated.

## Configuration

Extend `entity_extraction` with settings equivalent to:

```json
{
  "backend": "uie",
  "model_name": "uie-medical-base",
  "python_path": ".venv-uie/bin/python",
  "threshold": 0.5,
  "batch_size": 8,
  "startup_timeout_seconds": 30,
  "request_timeout_seconds": 120,
  "fallback_backend": "regex",
  "variant_regex": true
}
```

`config.mac.json` will opt into this backend. The repository default configuration remains unchanged so other installations do not unexpectedly require `.venv-uie`.

The configured Python path and worker script must be included in the prepare-stage build signature. Changing either invalidates prepared criteria so stale regex-only annotations are not reused.

## Error Handling

A UIE technical failure means the worker cannot produce a valid extraction response. Examples include a missing executable, missing/corrupt model files, startup failure, timeout, premature process exit, malformed JSON, or an out-of-memory error. An empty but valid extraction result is not a technical failure.

On the first technical failure:

1. log one clear warning containing the non-sensitive failure reason;
2. stop and reap the UIE worker;
3. switch the recognizer to regex for the failed batch and all remaining batches in that build;
4. continue registry preparation rather than aborting the build;
5. expose the fallback in logs and build metadata so it is never silent.

The worker must not automatically download a missing model during normal offline builds. A missing cache is an actionable startup failure and triggers the documented fallback.

## Performance

The persistent-worker design avoids repeated model startup. Batching is used within the worker, starting at eight criteria per batch and reducible through configuration if memory pressure appears.

For the current 29-criterion test trial, the acceptance target is completion within five minutes on the target Mac CPU. This is a generous functional ceiling, not a throughput claim. Large registry imports remain offline build work and may take hours; normal patient matching uses the stored annotations and does not rerun UIE.

## Testing

1. Unit-test schema prompt parsing and validation.
2. Unit-test UIE response conversion, offsets, thresholds, cross-type overlaps, and deduplication without loading Paddle.
3. Unit-test subprocess startup, request/response handling, timeout, malformed response, process exit, and cleanup with a fake worker.
4. Unit-test regex fallback and verify it is logged and recorded.
5. Keep all existing GLiNER2, regex, disabled-backend, preparation, configuration, and preflight tests green.
6. Run a local live smoke test against cached `uie-medical-base` with synthetic Chinese medical text.
7. Rebuild the supplied `ALSC013AST2818` trial and verify all 29 criteria complete.
8. Inspect representative Chinese criteria for disease, drug, gene, variant, test, and procedure entities.
9. Rerun the three synthetic patient matches and verify report generation remains successful.

No real patient data is required for implementation testing.

## Acceptance Criteria

- The main TrialMatchAI process never imports PaddlePaddle or PaddleNLP.
- One UIE worker is reused throughout a registry build.
- Chinese medical entities are stored in prepared criterion documents using the existing annotation shape.
- The current 29-criterion test corpus completes within five minutes on the target Mac.
- A UIE technical failure visibly falls back to regex without aborting the build.
- Existing retrieval, Qwen reranking, DeepSeek assessment, and reports remain operational.
- Rebuilding is triggered when the entity backend, model, worker executable, worker implementation, or confidence threshold changes.

## Out of Scope

- Patient-summary generation or translation.
- Replacing BGE-M3, Qwen, or DeepSeek.
- UIE fine-tuning or clinical accuracy claims.
- GPU/MPS acceleration for PaddlePaddle on macOS.
- Remote UIE services.
- Automatic installation or repair of `.venv-uie` during a normal build.
