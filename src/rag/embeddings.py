"""向量层。

默认 hashed：不下载模型，立刻能跑通「文本 → 向量 → 相似度」这一课。
切换 huggingface 后，中文语义检索会明显更好（需安装 sentence-transformers）。
"""

from __future__ import annotations

import hashlib
from typing import List

from langchain_core.embeddings import Embeddings

from rag import config


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
    if config.EMBEDDING_BACKEND in {"huggingface", "sentence-transformers", "st"}:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "句向量依赖未安装。Streamlit Cloud 只安装仓库里 requirements.txt 列出的包；"
                "请确认该文件含 sentence-transformers 与 langchain-huggingface，并在 Cloud 里 Reboot 重新安装。"
            ) from exc
        return HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL)
    return HashedNgramEmbeddings()
