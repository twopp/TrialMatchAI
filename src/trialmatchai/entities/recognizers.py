from __future__ import annotations

import atexit
import json
import os
import re
import selectors
import subprocess
import tempfile
import threading
from importlib import resources
from pathlib import Path
from typing import Any, Protocol, Sequence

from trialmatchai.entities.schemas import schema_by_label
from trialmatchai.entities.types import EntityAnnotation, EntitySchema, NO_ENTITY_ID
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)

VARIANT_PATTERNS_RESOURCE = ("trialmatchai.entities", "resources/variant_patterns.tsv")

# Variant cues that are also ordinary English words ("loss", "increased"): keep a match
# only when a gene/biomarker-like token sits nearby ("PTEN loss"), else they fire on prose.
_AMBIGUOUS_VARIANT_WORDS = frozenset(
    {
        "increase",
        "increased",
        "increases",
        "loss",
        "lost",
        "insertion",
        "insertions",
        "inhibitor",
        "inhibitors",
        "inhibition",
        "inhibited",
        "inhibits",
    }
)
# Gene/biomarker shapes: all-caps (EGFR), title-case+digit (Brca1), p53-style.
# Conservative on purpose so plain lowercase prose isn't rescued.
_GENE_CONTEXT = re.compile(r"\b(?:[A-Z]{2,}[0-9]*|[A-Z][a-z]+[0-9]+|p[0-9]{2})\b")
_GENE_CONTEXT_WINDOW = 30


class EntityRecognizer(Protocol):
    def recognize(
        self, texts: Sequence[str], schemas: Sequence[EntitySchema]
    ) -> list[list[EntityAnnotation]]:
        ...


class DisabledRecognizer:
    def recognize(
        self, texts: Sequence[str], schemas: Sequence[EntitySchema]
    ) -> list[list[EntityAnnotation]]:
        return [[] for _ in texts]


class RegexSchemaRecognizer:
    """Small deterministic recognizer used for smoke tests and fixture runs."""

    def recognize(
        self, texts: Sequence[str], schemas: Sequence[EntitySchema]
    ) -> list[list[EntityAnnotation]]:
        compiled = [
            (schema, re.compile(pattern, re.IGNORECASE))
            for schema in schemas
            for pattern in schema.patterns
        ]
        results: list[list[EntityAnnotation]] = []
        for text in texts:
            annotations: list[EntityAnnotation] = []
            for schema, pattern in compiled:
                for match in pattern.finditer(text):
                    mention = match.group(0)
                    annotations.append(
                        EntityAnnotation(
                            entity_group=schema.entity_group,
                            text=mention,
                            start=match.start(),
                            end=match.end(),
                            score=max(schema.threshold, 0.95),
                            normalized_id=(NO_ENTITY_ID,),
                            schema_id=schema.id,
                        )
                    )
            results.append(resolve_overlaps(annotations))
        return results


def _load_variant_patterns() -> list[tuple[str, "re.Pattern[str]"]]:
    """Load curated genetic-variant regexes (label, compiled pattern)."""
    package, name = VARIANT_PATTERNS_RESOURCE
    try:
        raw = resources.files(package).joinpath(name).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        logger.warning("Variant pattern resource not found; variant detection off.")
        return []
    patterns: list[tuple[str, "re.Pattern[str]"]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        label = parts[0].strip() or "variant"
        # The table stores patterns with doubled backslashes; collapse to one.
        pattern_text = parts[-1].replace("\\\\", "\\")
        try:
            patterns.append((label, re.compile(pattern_text)))
        except re.error as exc:
            logger.warning("Skipping invalid variant pattern %r: %s", label, exc)
    return patterns


class RegexVariantRecognizer:
    """Deterministic recognizer for genetic variants (HGVS, fusions, chromosome arms)
    from a curated pattern table, run alongside the model to catch precise strings
    (``p.V600E``, ``EGFR fusion``) a generalist NER model misses.
    """

    def __init__(self, patterns: list[tuple[str, "re.Pattern[str]"]] | None = None):
        self._patterns = patterns if patterns is not None else _load_variant_patterns()

    def recognize(
        self, texts: Sequence[str], schemas: Sequence[EntitySchema]
    ) -> list[list[EntityAnnotation]]:
        results: list[list[EntityAnnotation]] = []
        for text in texts:
            annotations: list[EntityAnnotation] = []
            for label, pattern in self._patterns:
                for match in pattern.finditer(text):
                    start, end = match.start(), match.end()
                    matched = match.group(0).strip()
                    if end <= start or not matched:
                        continue  # skip zero-width / whitespace-only matches
                    if matched.lower() in _AMBIGUOUS_VARIANT_WORDS:
                        window = text[max(0, start - _GENE_CONTEXT_WINDOW) : end + _GENE_CONTEXT_WINDOW]
                        if not _GENE_CONTEXT.search(window):
                            continue  # bare ambiguous word, no nearby gene -> drop
                    annotations.append(
                        EntityAnnotation(
                            entity_group=label,
                            text=match.group(0).strip(),
                            start=start,
                            end=end,
                            score=0.97,
                            normalized_id=(NO_ENTITY_ID,),
                            schema_id=None,
                        )
                    )
            results.append(resolve_overlaps(annotations))
        return results


class CompositeRecognizer:
    """Runs a primary recognizer plus augmenters and merges their annotations.

    Overlaps are resolved by confidence then span length, so a high-precision
    variant match wins over a lower-confidence model span covering the same text.
    """

    def __init__(self, primary: EntityRecognizer, *augmenters: EntityRecognizer):
        self._recognizers = [primary, *augmenters]

    def recognize(
        self, texts: Sequence[str], schemas: Sequence[EntitySchema]
    ) -> list[list[EntityAnnotation]]:
        per_recognizer = [r.recognize(texts, schemas) for r in self._recognizers]
        merged: list[list[EntityAnnotation]] = []
        for i in range(len(texts)):
            combined: list[EntityAnnotation] = []
            for recognizer_results in per_recognizer:
                combined.extend(recognizer_results[i])
            merged.append(resolve_overlaps_by_group(combined))
        return merged

    def close(self) -> None:
        for recognizer in self._recognizers:
            close = getattr(recognizer, "close", None)
            if callable(close):
                close()

    def runtime_status(self) -> dict[str, Any]:
        status = getattr(self._recognizers[0], "runtime_status", None)
        return status() if callable(status) else {"active_backend": "composite"}


class FallbackRecognizer:
    """Permanently switch to a safe recognizer after the primary first fails."""

    def __init__(
        self,
        primary: EntityRecognizer,
        fallback: EntityRecognizer,
        *,
        primary_name: str,
        fallback_name: str,
    ):
        self.primary = primary
        self.fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name
        self._failed = False
        self._failure_reason: str | None = None

    def recognize(
        self, texts: Sequence[str], schemas: Sequence[EntitySchema]
    ) -> list[list[EntityAnnotation]]:
        if self._failed:
            return self.fallback.recognize(texts, schemas)
        try:
            return self.primary.recognize(texts, schemas)
        except Exception as exc:
            self._failed = True
            self._failure_reason = str(exc)
            close = getattr(self.primary, "close", None)
            if callable(close):
                close()
            logger.warning(
                "Entity extraction backend %s failed; using %s for this and "
                "all remaining requests: %s",
                self.primary_name,
                self.fallback_name,
                exc,
            )
            return self.fallback.recognize(texts, schemas)

    def close(self) -> None:
        for recognizer in (self.primary, self.fallback):
            close = getattr(recognizer, "close", None)
            if callable(close):
                close()

    def runtime_status(self) -> dict[str, Any]:
        return {
            "configured_backend": self.primary_name,
            "active_backend": self.fallback_name if self._failed else self.primary_name,
            "fallback_used": self._failed,
            "fallback_reason": self._failure_reason,
        }


class UIESubprocessRecognizer:
    """Run PaddleNLP UIE in a persistent, dependency-isolated subprocess."""

    def __init__(
        self,
        model_name: str,
        *,
        python_path: str,
        threshold: float = 0.5,
        batch_size: int = 8,
        startup_timeout_seconds: float = 30.0,
        request_timeout_seconds: float = 120.0,
        worker_path: str | Path | None = None,
    ):
        self.model_name = model_name
        self.python_path = _resolve_executable(python_path)
        self.threshold = float(threshold)
        self.batch_size = int(batch_size)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.worker_path = Path(worker_path or Path(__file__).with_name("uie_worker.py"))
        self._process: subprocess.Popen[str] | None = None
        self._stderr = None
        self._prompt_signature: tuple[tuple[str, str], ...] | None = None
        self._request_id = 0
        self._lock = threading.Lock()
        atexit.register(self.close)

    def recognize(
        self, texts: Sequence[str], schemas: Sequence[EntitySchema]
    ) -> list[list[EntityAnnotation]]:
        if not texts:
            return []
        prompt_schemas = [
            (schema.uie_prompt or schema.label, schema)
            for schema in schemas
            if (schema.uie_prompt or schema.label).strip()
        ]
        signature = tuple((prompt, schema.id) for prompt, schema in prompt_schemas)
        with self._lock:
            self._ensure_started(signature)
            all_results: list[list[EntityAnnotation]] = []
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                response = self._request(batch)
                raw_results = response.get("results")
                if not isinstance(raw_results, list) or len(raw_results) != len(batch):
                    raise RuntimeError("UIE worker returned an invalid result count.")
                all_results.extend(
                    _parse_uie_results(
                        raw_results,
                        batch,
                        prompt_schemas,
                        threshold=self.threshold,
                    )
                )
            return all_results

    def _ensure_started(self, signature: tuple[tuple[str, str], ...]) -> None:
        if self._process is not None and self._process.poll() is None:
            if signature != self._prompt_signature:
                raise RuntimeError("UIE schema changed after the worker was started.")
            return
        if not self.python_path.exists():
            raise RuntimeError(f"UIE Python executable does not exist: {self.python_path}")
        if not self.worker_path.exists():
            raise RuntimeError(f"UIE worker script does not exist: {self.worker_path}")

        prompts = [prompt for prompt, _schema_id in signature]
        self._stderr = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self._process = subprocess.Popen(
            [
                str(self.python_path),
                str(self.worker_path),
                "--model",
                self.model_name,
                "--schema-json",
                json.dumps(prompts, ensure_ascii=False),
                "--batch-size",
                str(self.batch_size),
                "--threshold",
                str(self.threshold),
                "--cache-only",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        self._prompt_signature = signature
        ready = self._read_response(self.startup_timeout_seconds)
        if ready.get("type") != "ready":
            raise RuntimeError(f"UIE worker did not become ready: {ready}")

    def _request(self, texts: list[str]) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("UIE worker is not running.")
        self._request_id += 1
        request_id = str(self._request_id)
        payload = {"id": request_id, "texts": texts}
        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"UIE worker input pipe failed: {exc}") from exc
        response = self._read_response(self.request_timeout_seconds)
        if response.get("id") != request_id:
            raise RuntimeError("UIE worker returned a mismatched response id.")
        if response.get("error"):
            raise RuntimeError(f"UIE worker error: {response['error']}")
        return response

    def _read_response(self, timeout: float) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("UIE worker output pipe is unavailable.")
        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError(f"UIE worker timed out after {timeout:g} seconds.")
            line = process.stdout.readline()
        finally:
            selector.close()
        if not line:
            code = process.poll()
            detail = self._stderr_tail()
            raise RuntimeError(
                f"UIE worker exited unexpectedly (code={code})."
                + (f" Diagnostics: {detail}" if detail else "")
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"UIE worker returned invalid JSON: {line[:200]!r}") from exc
        if not isinstance(response, dict):
            raise RuntimeError("UIE worker response must be a JSON object.")
        return response

    def _stderr_tail(self) -> str:
        if self._stderr is None:
            return ""
        try:
            self._stderr.flush()
            self._stderr.seek(0)
            return self._stderr.read()[-1000:].strip()
        except Exception:
            return ""

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.write('{"type":"shutdown"}\n')
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def runtime_status(self) -> dict[str, Any]:
        return {
            "configured_backend": "uie",
            "active_backend": "uie",
            "fallback_used": False,
            "model_name": self.model_name,
        }


class GLiNER2Recognizer:
    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str | None = None,
        trust_remote_code: bool = False,
        batch_size: int = 8,
    ):
        try:
            from gliner2 import GLiNER2  # type: ignore
        except Exception as exc:  # pragma: no cover - exercised without optional dep
            raise RuntimeError(
                "entity_extraction.backend=gliner2 requires the entity extra "
                "(`uv sync --extra entity`) and a GLiNER2-compatible model."
            ) from exc

        kwargs: dict[str, Any] = {}
        if revision:
            kwargs["revision"] = revision
        if trust_remote_code:
            kwargs["trust_remote_code"] = trust_remote_code
        self.model = GLiNER2.from_pretrained(model_name, **kwargs)
        if device and device != "auto" and hasattr(self.model, "to"):
            self.model.to(device)
        self.batch_size = batch_size

    def recognize(
        self, texts: Sequence[str], schemas: Sequence[EntitySchema]
    ) -> list[list[EntityAnnotation]]:
        labels = [schema.label for schema in schemas]
        label_map = schema_by_label(list(schemas))
        return [
            resolve_overlaps(
                _parse_model_entities(
                    _call_extractor(self.model, text, labels, schemas),
                    text,
                    label_map,
                )
            )
            for text in texts
        ]


def build_recognizer(config: dict[str, Any]) -> EntityRecognizer:
    backend = str(config.get("backend", "gliner2")).lower()
    if backend == "disabled":
        return DisabledRecognizer()
    if backend == "regex":
        return RegexSchemaRecognizer()
    if backend == "uie":
        primary = UIESubprocessRecognizer(
            model_name=config.get("model_name", "uie-medical-base"),
            python_path=config.get("python_path", ".venv-uie/bin/python"),
            threshold=float(config.get("threshold", 0.5)),
            batch_size=int(config.get("batch_size", 8)),
            startup_timeout_seconds=float(config.get("startup_timeout_seconds", 30)),
            request_timeout_seconds=float(config.get("request_timeout_seconds", 120)),
            worker_path=config.get("worker_path"),
        )
        fallback_name = str(config.get("fallback_backend", "regex")).lower()
        fallback: EntityRecognizer
        if fallback_name == "disabled":
            fallback = DisabledRecognizer()
        else:
            fallback_name = "regex"
            fallback = RegexSchemaRecognizer()
        recognizer: EntityRecognizer = FallbackRecognizer(
            primary,
            fallback,
            primary_name="uie",
            fallback_name=fallback_name,
        )
        if bool(config.get("variant_regex", True)):
            variants = _load_variant_patterns()
            if variants:
                return CompositeRecognizer(recognizer, RegexVariantRecognizer(variants))
        return recognizer
    if backend == "gliner2":
        recognizer = GLiNER2Recognizer(
            model_name=config.get("model_name", "fastino/gliner2-base-v1"),
            revision=config.get("model_revision"),
            device=config.get("device", "auto"),
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            batch_size=int(config.get("batch_size", 8)),
        )
    else:
        raise ValueError(
            "entity_extraction.backend must be one of: gliner2, uie, regex, disabled."
        )

    # Augment model NER with the deterministic variant recognizer (on by default).
    if bool(config.get("variant_regex", True)):
        variants = _load_variant_patterns()
        if variants:
            return CompositeRecognizer(recognizer, RegexVariantRecognizer(variants))
    return recognizer


def resolve_overlaps(
    annotations: Sequence[EntityAnnotation],
) -> list[EntityAnnotation]:
    ranked = sorted(
        annotations,
        key=lambda ann: (ann.score, ann.end - ann.start),
        reverse=True,
    )
    accepted: list[EntityAnnotation] = []
    for candidate in ranked:
        if candidate.start < 0 or candidate.end <= candidate.start:
            continue
        if any(_overlaps(candidate, current) for current in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda ann: (ann.start, ann.end))


def resolve_overlaps_by_group(
    annotations: Sequence[EntityAnnotation],
) -> list[EntityAnnotation]:
    """Resolve competing spans within a semantic group, preserving cross-type spans."""
    grouped: dict[str, list[EntityAnnotation]] = {}
    for annotation in annotations:
        key = annotation.schema_id or annotation.entity_group
        grouped.setdefault(key, []).append(annotation)
    resolved = [item for group in grouped.values() for item in resolve_overlaps(group)]
    unique: dict[tuple[str, int, int, str], EntityAnnotation] = {}
    for annotation in resolved:
        key = (
            annotation.schema_id or annotation.entity_group,
            annotation.start,
            annotation.end,
            annotation.text,
        )
        current = unique.get(key)
        if current is None or annotation.score > current.score:
            unique[key] = annotation
    return sorted(unique.values(), key=lambda ann: (ann.start, ann.end, ann.entity_group))


def _resolve_executable(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    # Do not resolve the final symlink: a venv's ``bin/python`` points at the base
    # interpreter, and invoking that resolved target would lose the venv site-packages.
    cwd_candidate = (Path.cwd() / path).absolute()
    if cwd_candidate.exists():
        return cwd_candidate
    project_candidate = (Path(__file__).resolve().parents[3] / path).absolute()
    return project_candidate


def _parse_uie_results(
    raw_results: Sequence[Any],
    texts: Sequence[str],
    prompt_schemas: Sequence[tuple[str, EntitySchema]],
    *,
    threshold: float,
) -> list[list[EntityAnnotation]]:
    prompt_map = {prompt: schema for prompt, schema in prompt_schemas}
    parsed_results: list[list[EntityAnnotation]] = []
    for raw_result, text in zip(raw_results, texts):
        if not isinstance(raw_result, dict):
            raise RuntimeError("UIE result item must be an object.")
        annotations: list[EntityAnnotation] = []
        for prompt, values in raw_result.items():
            schema = prompt_map.get(str(prompt))
            if schema is None or not isinstance(values, list):
                continue
            for raw in values:
                if not isinstance(raw, dict):
                    continue
                start = _as_int(raw.get("start"))
                end = _as_int(raw.get("end"))
                score = float(raw.get("probability") or raw.get("score") or 0.0)
                if (
                    start is None
                    or end is None
                    or start < 0
                    or end <= start
                    or end > len(text)
                    or score < threshold
                ):
                    continue
                mention = text[start:end]
                reported = str(raw.get("text") or "")
                if reported and reported != mention:
                    continue
                annotations.append(
                    EntityAnnotation(
                        entity_group=schema.entity_group,
                        text=mention,
                        start=start,
                        end=end,
                        score=score,
                        normalized_id=(NO_ENTITY_ID,),
                        schema_id=schema.id,
                    )
                )
        parsed_results.append(resolve_overlaps_by_group(annotations))
    return parsed_results


def _overlaps(left: EntityAnnotation, right: EntityAnnotation) -> bool:
    return left.start < right.end and right.start < left.end


def _call_extractor(
    model: Any,
    text: str,
    labels: list[str],
    schemas: Sequence[EntitySchema],
) -> list[dict[str, Any]]:
    if hasattr(model, "extract_entities"):
        schema_payload = {
            schema.label: schema.description
            for schema in schemas
        }
        try:
            return _flatten_gliner2_entities(
                model.extract_entities(
                    text,
                    schema_payload,
                    include_confidence=True,
                    include_spans=True,
                )
            )
        except TypeError:
            pass
        for kwargs in (
            {"schema": schema_payload},
            {"labels": labels},
            {},
        ):
            try:
                return model.extract_entities(text, **kwargs)
            except TypeError:
                continue
    if hasattr(model, "predict_entities"):
        return model.predict_entities(text, labels)
    raise RuntimeError("Selected entity model does not expose an entity extraction API.")


def _parse_model_entities(
    raw_entities: Sequence[dict[str, Any]] | dict[str, Any],
    text: str,
    label_map: dict[str, EntitySchema],
) -> list[EntityAnnotation]:
    parsed: list[EntityAnnotation] = []
    for raw in _flatten_gliner2_entities(raw_entities):
        label = str(raw.get("label") or raw.get("entity_group") or raw.get("type") or "")
        schema = label_map.get(label.casefold())
        if schema is None:
            continue
        mention = str(raw.get("text") or raw.get("span") or raw.get("mention") or "")
        start = _as_int(raw.get("start"))
        end = _as_int(raw.get("end"))
        if start is None or end is None:
            start, end = _find_span(text, mention)
        if start is None or end is None:
            continue
        mention = text[start:end]
        score = float(raw.get("score") or raw.get("confidence") or 0.0)
        if score < schema.threshold:
            continue
        parsed.append(
            EntityAnnotation(
                entity_group=schema.entity_group,
                text=mention,
                start=start,
                end=end,
                score=score,
                normalized_id=(NO_ENTITY_ID,),
                schema_id=schema.id,
            )
        )
    return parsed


def _flatten_gliner2_entities(raw_entities: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_entities, dict):
        return list(raw_entities or [])
    entities = raw_entities.get("entities")
    if not isinstance(entities, dict):
        return []
    flattened: list[dict[str, Any]] = []
    for label, values in entities.items():
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict):
                item = dict(value)
            else:
                item = {"text": str(value)}
            item.setdefault("label", label)
            flattened.append(item)
    return flattened


def _find_span(text: str, mention: str) -> tuple[int | None, int | None]:
    if not mention:
        return None, None
    # Match on the ORIGINAL text: casefold() can change length (ß->ss) and drift offsets.
    match = re.search(re.escape(mention), text, re.IGNORECASE)
    if not match:
        return None, None
    return match.start(), match.end()


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
