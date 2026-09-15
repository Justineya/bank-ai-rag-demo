"""把检索和生成串成一次 ask()。这是你对外应该调用的唯一入口。"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from rag import config
from rag.generate import Generator, build_generator, postprocess_answer
from rag.rerank import rerank_hits
from rag.retrieve import retrieve_ranked


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[Document]
    generator: str
    top_k: int


def ask(question: str, k: int | None = None, generator: Generator | None = None) -> RAGAnswer:
    top_k = k or config.TOP_K
    fetch_k = max(top_k * 4, config.FETCH_K, top_k)
    recalled = retrieve_ranked(question, k=top_k, fetch_k=fetch_k)
    kept = rerank_hits(question, recalled, keep=top_k)
    docs = [item["doc"] for item in kept]
    gen, name = (generator, "custom") if generator else build_generator()
    raw = gen.generate(question, docs)
    answer, _notes = postprocess_answer(raw, question, docs)
    return RAGAnswer(
        question=question,
        answer=answer,
        sources=docs,
        generator=name,
        top_k=top_k,
    )
