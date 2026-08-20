from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

MODEL_ID = "intfloat/multilingual-e5-small"
MAX_SEQ_LEN = 512  # verified: config.json max_position_embeddings

# Kept outside the repo tree by default (large, churny model files). D: has the room;
# C: was down to ~15GB free as of the project's move off OneDrive. Override with
# RAINGOA_MODEL_CACHE for a different location.
_DEFAULT_CACHE = Path(os.environ.get("RAINGOA_MODEL_CACHE", r"D:\dev-cache\raingoa\models"))


class E5Encoder:
    def __init__(self, cache_dir: Path = _DEFAULT_CACHE):
        self._quantized_dir = cache_dir / "multilingual-e5-small-int8"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        quantized_dir = self._load_or_quantize(cache_dir)
        # quantizer.quantize() writes "model_quantized.onnx", not the "model.onnx"
        # from_pretrained looks for by default — must be named explicitly or it
        # silently mismatches and falls back to picking whatever .onnx file it finds.
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            quantized_dir, file_name="model_quantized.onnx"
        )

    def _load_or_quantize(self, cache_dir: Path) -> Path:
        if (self._quantized_dir / "model_quantized.onnx").exists():
            return self._quantized_dir

        onnx_model = ORTModelForFeatureExtraction.from_pretrained(MODEL_ID, export=True)
        quantizer = ORTQuantizer.from_pretrained(onnx_model)
        # avx2, not avx512_vnni — the target host's instruction-set support isn't known,
        # avx2 is the broadly-compatible choice for a generic cloud x86_64 CPU.
        qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=self._quantized_dir, quantization_config=qconfig)
        self.tokenizer.save_pretrained(self._quantized_dir)
        return self._quantized_dir

    def _encode(self, prefixed_texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(
            prefixed_texts,
            max_length=MAX_SEQ_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self.model(**inputs)
        pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        return torch.nn.functional.normalize(pooled, p=2, dim=1).numpy()

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"query: {t}" for t in texts])

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([f"passage: {t}" for t in texts])

    def encode_passages_with_context(self, chunks: list[str], document: str) -> np.ndarray:
        """Late chunking: one forward pass over the whole document, then mean-pool each
        chunk's own token range, so every chunk vector carries surrounding context.

        Chunks beyond the model's 512-token window fall outside the encoded span and are
        encoded standalone instead — reported rather than silently mislabeled.
        """
        encoded = self.tokenizer(
            f"passage: {document}",
            max_length=MAX_SEQ_LEN,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        hidden = self.model(**encoded).last_hidden_state[0]

        prefix_len = len("passage: ")
        vectors: list[np.ndarray] = []
        uncovered: list[int] = []
        cursor = 0
        for i, chunk_text in enumerate(chunks):
            start = document.find(chunk_text, cursor)
            if start == -1:
                start = cursor
            end = start + len(chunk_text)
            cursor = end

            token_ids = [
                idx
                for idx, (tok_start, tok_end) in enumerate(offsets)
                if tok_end > tok_start  # skip special tokens, which map to (0, 0)
                and tok_start - prefix_len < end
                and tok_end - prefix_len > start
            ]
            if token_ids:
                pooled = hidden[token_ids].mean(dim=0)
                vectors.append(
                    torch.nn.functional.normalize(pooled, p=2, dim=0).detach().numpy()
                )
            else:
                uncovered.append(i)
                vectors.append(np.zeros(hidden.shape[-1], dtype=np.float32))

        if uncovered:
            fallback = self.encode_passages([chunks[i] for i in uncovered])
            for slot, vector in zip(uncovered, fallback):
                vectors[slot] = vector

        return np.vstack(vectors)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts
