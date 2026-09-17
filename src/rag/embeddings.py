"""向量层。

默认中文句向量（bge-small-zh）。哈希向量是教学开关：不下载模型、秒级看懂
「文本 → 向量 → 相似度」，但不是生产 Embedding。
"""

from __future__ import annotations

import hashlib
from typing import List

from langchain_core.embeddings import Embeddings

from rag import config

LAST_ERROR = ""


class HashedNgramEmbeddings(Embeddings):
    """教学用稀疏哈希向量：把字符 n-gram 映射到固定维度。

    它不是生产级 Embedding，但能让你看见向量检索的输入输出，
    并且对专有名词（产品名、电话、利率数字）检索足够用。
    """

    def __init__(self, dim: int = 256, ngram_sizes: tuple[int, ...] = (2, 3, 4)):
        self.dim = dim
        self.ngram_sizes = ngram_sizes

    def _vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        normalized = "".join(text.lower().split())
        if not normalized:
            return vec
        for n in self.ngram_sizes:
            if len(normalized) < n:
                continue
            for i in range(len(normalized) - n + 1):
                gram = normalized[i : i + n]
                digest = hashlib.md5(gram.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


def build_embeddings() -> Embeddings:
    global LAST_ERROR
    LAST_ERROR = ""
    if config.EMBEDDING_BACKEND in {"hashed", "hash", "ngram"}:
        return HashedNgramEmbeddings()
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        LAST_ERROR = (
            "句向量依赖未安装。请确认 requirements.txt 含 sentence-transformers "
            "与 langchain-huggingface，Cloud 需 Reboot。"
        )
        raise ImportError(LAST_ERROR) from exc
    try:
        return HuggingFaceEmbeddings(
            model_name=config.HF_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        LAST_ERROR = f"句向量模型加载失败（{exc}）。已回退教学哈希，便于继续走完教室。"
        config.EMBEDDING_BACKEND = "hashed"
        config.COLLECTION_NAME = f"bank_kb_{config.EMBEDDING_BACKEND}"
        return HashedNgramEmbeddings()
