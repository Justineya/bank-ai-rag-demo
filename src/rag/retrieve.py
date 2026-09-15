"""检索阶段：BM25 主召回，向量检索作对照。

中文专有名词（随心贷、活期）在小语料上，词袋 BM25 通常比哈希向量更稳。
换 HuggingFace embedding 后，可把 RETRIEVER=vector 打开对比差异。
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag import config
from rag.ingest import get_vectorstore
from rag.textutil import BM25Index, tokenize


def _all_docs() -> list[Document]:
    store = get_vectorstore(reset=False)
    raw = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw.get("documents") or [], raw.get("metadatas") or [])
    ]


def retrieve_ranked(
    question: str,
    k: int | None = None,
    retriever: str | None = None,
    fetch_k: int | None = None,
) -> list[dict]:
    top_k = k or config.TOP_K
    recall_k = fetch_k or max(top_k, config.FETCH_K)
    docs = _all_docs()
    if not docs:
        return []
    recall_k = min(recall_k, len(docs))
    mode = (retriever or config.RETRIEVER).lower()
    if mode == "vector":
        store = get_vectorstore(reset=False)
        try:
            pairs = store.similarity_search_with_score(question, k=recall_k)
        except Exception:
            pairs = [(doc, 0.0) for doc in store.similarity_search(question, k=recall_k)]
        ranked = []
        for doc, dist in pairs:
            terms = set(tokenize(question)) & set(tokenize(doc.page_content))
            ranked.append(
                {
                    "score": round(float(dist), 4),
                    "doc": doc,
                    "matched": sorted(terms, key=len, reverse=True),
                    "query_terms": tokenize(question),
                    "score_kind": "vector_distance",
                }
            )
        return ranked
    return BM25Index(docs).ranked_search(question, recall_k)


def retrieve(question: str, k: int | None = None) -> list[Document]:
    top_k = k or config.TOP_K
    return [item["doc"] for item in retrieve_ranked(question, k=top_k, fetch_k=top_k)]


def format_context(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[资料{i} | {source}]\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)
