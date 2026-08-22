from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "intfloat/multilingual-e5-small"
MAX_SEQ_LEN = 512  # verified: config.json max_position_embeddings

# Kept outside the repo tree by default (large, churny model files). D: has the room;
# C: was down to ~15GB free as of the project's move off OneDrive. Override with
# RAINGOA_MODEL_CACHE for a different location.
_DEFAULT_CACHE = Path(os.environ.get("RAINGOA_MODEL_CACHE", r"D:\dev-cache\raingoa\models"))


class E5Encoder:
    """Two backends, chosen automatically by what's available:

    - GPU present (local dev machine building the corpus/index): plain PyTorch, fp16,
      no ONNX/quantization. int8 CPU quantization exists specifically to make ONNX
      Runtime's CPU execution provider fast — it doesn't help a GPU and the model
      wouldn't even load that way here.
    - No GPU (the deployed backend — Railway has none): the original ONNX int8-CPU
      path, byte-for-byte unchanged from before this class supported CUDA at all. The
      one-time index build is what benefits from a GPU; the deployed serving path
      (query encoding) never sees this branch and its behavior is untouched.
    """

    def __init__(self, cache_dir: Path = _DEFAULT_CACHE):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cuda":
            self.model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
            self.model.eval()
        else:
            self._quantized_dir = cache_dir / "multilingual-e5-small-int8"
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
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)
        pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        # .float() before .numpy(): fp16 output on the CUDA path would otherwise hand
        # back a float16 array — every downstream consumer (FAISS, cosine-similarity
        # scoring) is typed for float32, matching the CPU/ONNX path's output dtype.
        return normalized.float().cpu().numpy()

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
        if self.device == "cuda":
            encoded = {k: v.to("cuda") for k, v in encoded.items()}
            with torch.no_grad():
                hidden = self.model(**encoded).last_hidden_state[0]
        else:
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
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=0)
                vectors.append(normalized.float().detach().cpu().numpy())
            else:
                uncovered.append(i)
                vectors.append(np.zeros(hidden.shape[-1], dtype=np.float32))

        if uncovered:
            fallback = self.encode_passages([chunks[i] for i in uncovered])
            for slot, vector in zip(uncovered, fallback):
                vectors[slot] = vector

        return np.vstack(vectors)

    def encode_documents_with_context_batch(
        self, doc_chunks: list[tuple[str, list[str]]]
    ) -> list[np.ndarray]:
        """Batched version of encode_passages_with_context: one padded forward pass over
        several documents at once instead of one model call per document.

        Late chunking's real cost isn't compute (a single ~512-token forward pass is
        cheap) — it's that build_index.py calls encode_passages_with_context once per
        document, and with ~1 chunk/doc that's ~840k separate tiny GPU round trips
        (tokenize + kernel launch + sync), each paying fixed overhead for almost no
        actual work. Batching amortizes that fixed cost across many documents in one
        call. The per-chunk mean-pooling math below is identical to the single-document
        method — only the tokenizer/model call is batched, so results for a given
        document are the same (up to ordinary GPU float non-determinism) regardless of
        which other documents share its batch, since attention_mask already isolates
        each sequence from the padding around it.
        """
        documents = [d for d, _ in doc_chunks]
        encoded = self.tokenizer(
            [f"passage: {d}" for d in documents],
            max_length=MAX_SEQ_LEN,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets_batch = encoded.pop("offset_mapping").tolist()
        if self.device == "cuda":
            encoded = {k: v.to("cuda") for k, v in encoded.items()}
            with torch.no_grad():
                hidden_batch = self.model(**encoded).last_hidden_state
        else:
            hidden_batch = self.model(**encoded).last_hidden_state

        prefix_len = len("passage: ")
        per_doc_vectors: list[list[np.ndarray | None]] = []
        uncovered: list[tuple[int, int]] = []  # (doc_idx, chunk_idx)

        for doc_idx, (document, chunks) in enumerate(doc_chunks):
            offsets = offsets_batch[doc_idx]
            hidden = hidden_batch[doc_idx]
            vectors: list[np.ndarray | None] = []
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
                    if tok_end > tok_start
                    and tok_start - prefix_len < end
                    and tok_end - prefix_len > start
                ]
                if token_ids:
                    pooled = hidden[token_ids].mean(dim=0)
                    normalized = torch.nn.functional.normalize(pooled, p=2, dim=0)
                    vectors.append(normalized.float().detach().cpu().numpy())
                else:
                    uncovered.append((doc_idx, i))
                    vectors.append(None)
            per_doc_vectors.append(vectors)

        if uncovered:
            fallback_texts = [doc_chunks[d][1][c] for d, c in uncovered]
            fallback = self.encode_passages(fallback_texts)
            for (d, c), vector in zip(uncovered, fallback):
                per_doc_vectors[d][c] = vector

        return [np.vstack(vectors) for vectors in per_doc_vectors]


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts
