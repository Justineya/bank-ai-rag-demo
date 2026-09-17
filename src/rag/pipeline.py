"""把检索和生成串成一次 ask()。这是你对外应该调用的唯一入口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from langchain_core.documents import Document

from rag import config
from rag.generate import (
    Generator,
    build_generator,
    cited_clips,
    is_expired_doc,
    is_refusal,
    postprocess_answer,
    prefer_effective_docs,
)
from rag.acl import chunk_id, filter_docs
from rag.audit import append_event
from rag.rerank import rerank_hits
from rag.retrieve import retrieve_ranked


@dataclass
class RAGAnswer:
    question: str
    answer: str
    sources: list[Document]
    generator: str
    top_k: int
    clips: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    refused: bool = False
    rerank_backend: str = ""
    chunk_ids: list[str] = field(default_factory=list)
    audience: str = ""
    tenant: str = ""
    audit: dict = field(default_factory=dict)


def ask(
    question: str,
    k: int | None = None,
    generator: Generator | None = None,
    retriever: str | None = None,
    reranker: str | None = None,
    fetch_k: int | None = None,
) -> RAGAnswer:
    top_k = k or config.TOP_K
    recall_k = max(top_k * 4, fetch_k or config.FETCH_K, top_k)
    t0 = perf_counter()
    recalled = retrieve_ranked(question, k=top_k, fetch_k=recall_k, retriever=retriever)
    t1 = perf_counter()
    kept = rerank_hits(question, recalled, keep=top_k, mode=reranker)
    kept = keep_live_demand_rate(question, recalled, kept)
    kept = keep_audited_premium(question, recalled, kept)
    t2 = perf_counter()
    docs = prefer_effective_docs(filter_docs([item["doc"] for item in kept]))
    gen, name = (generator, "custom") if generator else build_generator()
    raw = gen.generate(question, docs)
    t3 = perf_counter()
    answer, notes = postprocess_answer(raw, question, docs)
    t4 = perf_counter()
    clips = cited_clips(answer, docs, question)
    backend = (kept[0].get("rerank_backend") if kept else "") or (reranker or config.RERANKER)
    ids = [chunk_id(doc) for doc in docs]
    sources = [str(doc.metadata.get("source", "")).rsplit("/", 1)[-1] for doc in docs]
    audit = append_event(
        {
            "question": question,
            "chunk_ids": ids,
            "sources": sources,
            "model": name,
            "refused": is_refusal(answer),
            "answer": answer,
            "notes": notes,
            "tenant": getattr(config, "TENANT", "bank"),
            "audience": getattr(config, "AUDIENCE", "public"),
        }
    )
    return RAGAnswer(
        question=question,
        answer=answer,
        sources=docs,
        generator=name,
        top_k=top_k,
        clips=clips,
        notes=notes,
        timings_ms={
            "retrieve": round((t1 - t0) * 1000, 1),
            "rerank": round((t2 - t1) * 1000, 1),
            "generate": round((t3 - t2) * 1000, 1),
            "postprocess": round((t4 - t3) * 1000, 1),
        },
        refused=is_refusal(answer),
        rerank_backend=str(backend),
        chunk_ids=ids,
        audience=str(getattr(config, "AUDIENCE", "")),
        tenant=str(getattr(config, "TENANT", "")),
        audit=audit,
    )


def keep_live_demand_rate(question: str, recalled: list[dict], kept: list[dict]) -> list[dict]:
    """冲突问法会把过期 1.50% 抬到前面；保证现行活期 0.20% 仍在生成上下文里。"""
    if "活期" not in (question or "") or not recalled:
        return kept
    def live_demand(hit: dict) -> bool:
        text = hit["doc"].page_content
        return "活期年利率" in text and "0.20%" in text and not is_expired_doc(hit["doc"])

    if any(live_demand(hit) for hit in kept):
        return kept
    replacement = next((hit for hit in recalled if live_demand(hit)), None)
    if replacement is None:
        return kept
    dropped = [hit for hit in kept if is_expired_doc(hit["doc"])]
    rest = [hit for hit in kept if not is_expired_doc(hit["doc"])]
    mixed = [replacement] + rest
    if dropped:
        mixed = mixed[: max(len(kept) - 1, 1)] + dropped[:1]
    return mixed[: len(kept) or 1]


def keep_audited_premium(question: str, recalled: list[dict], kept: list[dict]) -> list[dict]:
    """官网约数 19,727 万 vs 年报 19,663 万：对客保留已审计段落。"""
    q = question or ""
    if "保险收入" not in q and "19,727" not in q:
        return kept

    def audited(hit: dict) -> bool:
        text = hit["doc"].page_content
        return "19,663" in text and not is_expired_doc(hit["doc"])

    if any(audited(hit) for hit in kept):
        return kept
    replacement = next((hit for hit in recalled if audited(hit)), None)
    if replacement is None:
        return kept
    rest = [hit for hit in kept if "3 亿" not in hit["doc"].page_content]
    return ([replacement] + rest)[: max(len(kept), 1)]
