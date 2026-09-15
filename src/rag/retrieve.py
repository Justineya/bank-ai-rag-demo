"""检索阶段：问题 → 向量 → top-k chunk。"""

from __future__ import annotations

from langchain_core.documents import Document

from rag import config
from rag.ingest import get_vectorstore


def retrieve(question: str, k: int | None = None) -> list[Document]:
    store = get_vectorstore(reset=False)
    retriever = store.as_retriever(search_kwargs={"k": k or config.TOP_K})
    return retriever.invoke(question)


def format_context(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[资料{i} | {source}]\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)
