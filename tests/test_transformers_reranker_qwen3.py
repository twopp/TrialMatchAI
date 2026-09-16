from __future__ import annotations

from types import SimpleNamespace

from trialmatchai.models.llm.transformers_reranker import (
    QWEN3_CLINICAL_INSTRUCTION,
    _is_qwen3_reranker,
    _qwen3_pair_text,
    _resolve_device_name,
)


def test_detects_qwen3_reranker_model_case_insensitively():
    assert _is_qwen3_reranker("Qwen/Qwen3-Reranker-0.6B") is True
    assert _is_qwen3_reranker("/models/qwen3-reranker-4b") is True
    assert _is_qwen3_reranker("google/gemma-2-2b-it") is False


def test_qwen3_pair_uses_clinical_instruction_and_official_fields():
    text = _qwen3_pair_text("患者患有肺癌", "必须存在 EGFR 突变")

    assert text.startswith(f"<Instruct>: {QWEN3_CLINICAL_INSTRUCTION}\n")
    assert "<Query>: 患者患有肺癌" in text
    assert "<Document>: 必须存在 EGFR 突变" in text


def test_auto_device_prefers_mps_then_cpu():
    torch_with_mps = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    torch_without_accelerator = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )

    assert _resolve_device_name("auto", torch_with_mps) == "mps"
    assert _resolve_device_name("auto", torch_without_accelerator) == "cpu"


def test_requested_unavailable_mps_falls_back_to_cpu():
    torch_without_mps = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )

    assert _resolve_device_name("mps", torch_without_mps) == "cpu"
