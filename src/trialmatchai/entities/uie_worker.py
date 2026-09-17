"""Dependency-isolated PaddleNLP UIE JSON-lines worker.

This file is executed directly by the Python interpreter in ``.venv-uie``. Keep
it free of imports from the rest of TrialMatchAI so the worker environment only
needs PaddlePaddle and PaddleNLP.
"""

from __future__ import annotations

import sys

# Executing this file directly puts its ``entities`` directory first on sys.path,
# where ``types.py`` would shadow Python's standard-library ``types`` module.
if sys.path:
    sys.path.pop(0)

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any


def _write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _cached_model_path(model_name: str) -> Path:
    direct = Path(model_name).expanduser()
    if direct.exists():
        return direct
    home = Path(os.environ.get("PPNLP_HOME", Path.home() / ".paddlenlp"))
    return home / "taskflow" / "information_extraction" / model_name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--schema-json", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cache-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        prompts = json.loads(args.schema_json)
        if not isinstance(prompts, list) or not all(
            isinstance(prompt, str) and prompt.strip() for prompt in prompts
        ):
            raise ValueError("schema-json must be a non-empty JSON list of strings")
        if args.cache_only:
            cache = _cached_model_path(args.model)
            required = cache / "model_state.pdparams"
            if not required.exists():
                raise FileNotFoundError(
                    f"cached UIE model is missing: {required}; download it before building"
                )

        import paddle

        paddle.set_device("cpu")
        from paddlenlp import Taskflow

        extractor = Taskflow(
            "information_extraction",
            schema=prompts,
            model=args.model,
            device_id=-1,
            batch_size=args.batch_size,
            precision="fp32",
            position_prob=args.threshold,
        )
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        _write({"type": "startup_error", "error": f"{type(exc).__name__}: {exc}"})
        return 2

    _write({"type": "ready", "model": args.model})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("type") == "shutdown":
                return 0
            request_id = str(request.get("id") or "")
            texts = request.get("texts")
            if not request_id or not isinstance(texts, list) or not all(
                isinstance(text, str) for text in texts
            ):
                raise ValueError("request requires a non-empty id and a string-list texts")
            results = extractor(texts)
            _write({"id": request_id, "results": results})
        except Exception as exc:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            request_id = ""
            try:
                request_id = str(request.get("id") or "")
            except Exception:
                pass
            _write({"id": request_id, "error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
