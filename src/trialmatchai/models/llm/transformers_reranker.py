from __future__ import annotations

import math
from typing import Any, List, Optional

from tqdm import tqdm

from trialmatchai.models.llm.llm_reranker import LLMReranker
from trialmatchai.utils.logging_config import setup_logging

logger = setup_logging(__name__)

QWEN3_CLINICAL_INSTRUCTION = (
    "Given a patient profile, retrieve clinical trial eligibility criteria that "
    "contain information needed to determine whether the patient can participate."
)
_QWEN3_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on '
    'the Query and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
_QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _is_qwen3_reranker(model_path: str) -> bool:
    return "qwen3-reranker" in str(model_path).lower()


def _qwen3_pair_text(patient_text: str, criterion_text: str) -> str:
    return (
        f"<Instruct>: {QWEN3_CLINICAL_INSTRUCTION}\n"
        f"<Query>: {patient_text}\n"
        f"<Document>: {criterion_text}"
    )


def _resolve_device_name(requested: str, torch_module: Any) -> str:
    value = str(requested or "auto").lower()
    cuda_available = bool(torch_module.cuda.is_available())
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if value == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if value == "mps" and not mps_available:
        logger.warning("MPS was requested but is unavailable; using CPU for reranking.")
        return "cpu"
    if value.startswith("cuda") and not cuda_available:
        logger.warning("CUDA was requested but is unavailable; using CPU for reranking.")
        return "cpu"
    if value.isdigit():
        if cuda_available:
            return f"cuda:{value}"
        logger.warning("CUDA device %s was requested but is unavailable; using CPU.", value)
        return "cpu"
    return value


class TransformersReranker:
    """CPU-capable next-token Yes/No reranker for smoke tests and small runs."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        batch_size: int = 8,
        model_family: str = "auto",
        max_length: int = 4096,
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "Transformers reranker requires the llm extra "
                "(`uv sync --extra llm`)."
            ) from exc

        self.torch = torch
        self.device = torch.device(_resolve_device_name(device, torch))
        self.batch_size = batch_size
        self.max_length = max(256, int(max_length))
        self.is_qwen3 = (
            _is_qwen3_reranker(model_path)
            if model_family == "auto"
            else str(model_family).lower() == "qwen3"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        # Left-pad so logits[:, -1, :] reads each prompt's real final token, not a pad token.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }
        if self.device.type == "mps":
            model_kwargs["dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        ).to(self.device)
        self.model.eval()
        if self.is_qwen3:
            self.yes_token_id = self._first_token_id("yes", "yes")
            self.no_token_id = self._first_token_id("no", "no")
            self.qwen_prefix_tokens = self.tokenizer.encode(
                _QWEN3_PREFIX, add_special_tokens=False
            )
            self.qwen_suffix_tokens = self.tokenizer.encode(
                _QWEN3_SUFFIX, add_special_tokens=False
            )
        else:
            self.yes_token_id = self._first_token_id(" Yes", "Yes")
            self.no_token_id = self._first_token_id(" No", "No")
        logger.info("Loaded Transformers reranker %s on %s.", model_path, self.device)

    def _first_token_id(self, preferred: str, fallback: str) -> int:
        token_ids = self.tokenizer(preferred, add_special_tokens=False)["input_ids"]
        if not token_ids:
            token_ids = self.tokenizer(fallback, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError(f"Tokenizer could not encode {fallback!r}.")
        return int(token_ids[0])

    def rank_pairs(self, patient_trial_pairs: List[tuple]) -> List[dict[str, Any]]:
        results: List[dict[str, Any]] = []
        for start in tqdm(
            range(0, len(patient_trial_pairs), self.batch_size),
            desc="Transformers reranking batches",
        ):
            batch = patient_trial_pairs[start : start + self.batch_size]
            encoded = self._encode_batch(batch)
            next_logits = self._last_token_logits(encoded)
            for row in next_logits:
                yes_logit = float(row[self.yes_token_id])
                no_logit = float(row[self.no_token_id])
                highest = max(yes_logit, no_logit)
                yes = math.exp(yes_logit - highest)
                no = math.exp(no_logit - highest)
                prob = yes / (yes + no)
                results.append(
                    {"llm_score": prob, "answer": "Yes" if prob > 0.5 else "No"}
                )
        return results

    def _encode_batch(self, batch: List[tuple]):
        if not self.is_qwen3:
            prompts = [self._build_prompt(patient, trial) for patient, trial in batch]
            return self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

        pair_texts = [_qwen3_pair_text(patient, trial) for patient, trial in batch]
        content_budget = max(
            1,
            self.max_length
            - len(self.qwen_prefix_tokens)
            - len(self.qwen_suffix_tokens),
        )
        encoded = self.tokenizer(
            pair_texts,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=content_budget,
        )
        encoded["input_ids"] = [
            self.qwen_prefix_tokens + ids + self.qwen_suffix_tokens
            for ids in encoded["input_ids"]
        ]
        return self.tokenizer.pad(
            encoded,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

    def _last_token_logits(self, encoded):
        try:
            with self.torch.inference_mode():
                return self.model(**encoded).logits[:, -1, :]
        except RuntimeError:
            if self.device.type != "mps":
                raise
            logger.warning(
                "Qwen reranking failed on MPS; moving the model and batch to CPU."
            )
            self.device = self.torch.device("cpu")
            self.model = self.model.to(device=self.device, dtype=self.torch.float32)
            encoded = encoded.to(self.device)
            with self.torch.inference_mode():
                return self.model(**encoded).logits[:, -1, :]

    def _build_prompt(self, patient_text: str, trial_text: str) -> str:
        messages = LLMReranker.create_messages(patient_text, trial_text)
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return (
            f"{messages[0]['content']}\n\n"
            f"{messages[2]['content']}\nAnswer Yes or No:"
        )
