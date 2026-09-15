"""检索阶段：BM25 主召回，向量检索作对照。

中文专有名词（随心贷、活期）在小语料上，词袋 BM25 通常比哈希向量更稳。
换 HuggingFace embedding 后，可把 RETRIEVER=vector 打开对比差异。
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag import config
from rag.ingest import get_vectorstore
from rag.textutil import BM25Index, lexical_overlap


def retrieve(question: str, k: int | None = None) -> list[Document]:
    top_k = k or config.TOP_K
    store = get_vectorstore(reset=False)
    raw = store.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw.get("documents") or [], raw.get("metadatas") or [])
    ]
    if not docs:
        return []

    if config.RETRIEVER == "vector":
        fetch_k = min(max(top_k * 4, top_k), len(docs))
        candidates = store.similarity_search(question, k=fetch_k)
        ranked = sorted(candidates, key=lambda doc: lexical_overlap(question, doc.page_content), reverse=True)
        return ranked[:top_k]

    return BM25Index(docs).search(question, top_k)


def format_context(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[资料{i} | {source}]\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)
