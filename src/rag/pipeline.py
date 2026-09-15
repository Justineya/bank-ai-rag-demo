"""把检索和生成串成一次 ask()。这是你对外应该调用的唯一入口。"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from rag import config
from rag.generate import Generator, build_generator
from rag.retrieve import retrieve


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[Document]
    generator: str
    top_k: int


def ask(question: str, k: int | None = None, generator: Generator | None = None) -> RAGAnswer:
    top_k = k or config.TOP_K
    docs = retrieve(question, k=top_k)
    gen, name = (generator, "custom") if generator else build_generator()
    answer = gen.generate(question, docs)
    return RAGAnswer(
        question=question,
        answer=answer,
        sources=docs,
        generator=name,
        top_k=top_k,
    )
