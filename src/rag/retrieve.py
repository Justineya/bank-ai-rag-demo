"""检索阶段：BM25 主召回，向量检索作对照，Hybrid 用 RRF 融合。

中文专有名词（随心贷、活期）在小语料上，词袋 BM25 通常比哈希向量更稳。
换 HuggingFace embedding 后，把 RETRIEVER=hybrid 打开可看见两路名次如何合成。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from rag import config
from rag.acl import filter_docs, is_allowed
from rag.ingest import get_vectorstore
from rag.textutil import BM25Index, lexical_overlap, tokenize


def _all_docs() -> list[Document]:
    store = get_vectorstore(reset=False)
    raw = store.get(include=["documents", "metadatas"])
    return filter_docs(
        [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(raw.get("documents") or [], raw.get("metadatas") or [])
        ]
    )


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
    if mode == "hybrid":
        table = hybrid_table(question, fetch_k=recall_k, docs=docs)
        return table["fused"]
    if mode == "vector":
        return _vector_ranked(question, recall_k)
    return BM25Index(docs).ranked_search(question, recall_k)


def _vector_ranked(question: str, recall_k: int) -> list[dict]:
    store = get_vectorstore(reset=False)
    try:
        pairs = store.similarity_search_with_score(question, k=recall_k)
    except Exception:
        pairs = [(doc, 0.0) for doc in store.similarity_search(question, k=recall_k)]
    ranked = []
    for doc, dist in pairs:
        if not is_allowed(doc):
            continue
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


def doc_key(doc: Document) -> tuple:
    return (
        Path(str(doc.metadata.get("source", ""))).name,
        str(doc.metadata.get("page") or ""),
        (doc.page_content or "")[:120],
    )


def hybrid_table(question: str, fetch_k: int | None = None, docs: list[Document] | None = None) -> dict:
    """BM25 + 向量两路，再用 RRF 融合。返回对照表所需的三列名次。"""
    pool = docs if docs is not None else _all_docs()
    if not pool:
        return {"bm25": [], "vector": [], "fused": [], "rows": []}
    recall_k = min(fetch_k or config.FETCH_K, len(pool))
    bm25 = BM25Index(pool).ranked_search(question, recall_k)
    vector = _vector_ranked(question, recall_k)
    fused = rrf_fuse(bm25, vector, keep=recall_k)
    by_key = {}
    for rank, hit in enumerate(bm25, 1):
        by_key.setdefault(doc_key(hit["doc"]), {"doc": hit["doc"]})["bm25_rank"] = rank
    for rank, hit in enumerate(vector, 1):
        by_key.setdefault(doc_key(hit["doc"]), {"doc": hit["doc"]})["vector_rank"] = rank
    for rank, hit in enumerate(fused, 1):
        by_key.setdefault(doc_key(hit["doc"]), {"doc": hit["doc"]})["fused_rank"] = rank
        by_key[doc_key(hit["doc"])]["rrf"] = hit["score"]
    rows = []
    for key, row in by_key.items():
        src = key[0]
        text = row["doc"].page_content.strip().replace("\n", " ")
        rows.append(
            {
                "来源": src,
                "BM25名次": row.get("bm25_rank") or "—",
                "向量名次": row.get("vector_rank") or "—",
                "融合名次": row.get("fused_rank") or "—",
                "RRF": row.get("rrf"),
                "开头": text[:32],
            }
        )
    rows.sort(key=lambda r: (999 if r["融合名次"] == "—" else r["融合名次"]))
    return {"bm25": bm25, "vector": vector, "fused": fused, "rows": rows}


def rrf_fuse(bm25_hits: list[dict], vector_hits: list[dict], keep: int, k: int = 60) -> list[dict]:
    scores: dict[tuple, float] = {}
    store: dict[tuple, dict] = {}
    for rank, hit in enumerate(bm25_hits, 1):
        key = doc_key(hit["doc"])
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        item = dict(hit)
        item["bm25_rank"] = rank
        store[key] = item
    for rank, hit in enumerate(vector_hits, 1):
        key = doc_key(hit["doc"])
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key in store:
            store[key]["vector_rank"] = rank
            matched = set(store[key].get("matched") or []) | set(hit.get("matched") or [])
            store[key]["matched"] = sorted(matched, key=len, reverse=True)
        else:
            item = dict(hit)
            item["vector_rank"] = rank
            store[key] = item
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:keep]
    fused = []
    for i, (key, score) in enumerate(ordered, 1):
        item = store[key]
        item["score"] = round(score, 6)
        item["score_kind"] = "rrf"
        item["fused_rank"] = i
        fused.append(item)
    return fused


def explain_hits(question: str, chunks: list[Document], k: int = 5) -> list[dict]:
    """不写向量库：用当前切块当场 BM25，解释为什么命中、overlap 是多少。"""
    if not chunks:
        return []
    ranked = BM25Index(chunks).ranked_search(question, min(k, len(chunks)))
    rows = []
    for item in ranked:
        doc = item["doc"]
        text = doc.page_content
        overlap = str(doc.metadata.get("overlap_preview") or "")
        rows.append(
            {
                "chunk": doc.metadata.get("chunk_index"),
                "source": Path(str(doc.metadata.get("source", ""))).name,
                "section": doc.metadata.get("section") or (text.strip().splitlines() or [""])[0][:40],
                "score": item["score"],
                "matched": item.get("matched") or [],
                "overlap_chars": doc.metadata.get("overlap_chars") or 0,
                "overlap_preview": overlap,
                "why": _why_hit(question, text, item.get("matched") or []),
                "preview": text.strip().replace("\n", " ")[:80],
                "chars": len(text),
            }
        )
    return rows


def _why_hit(question: str, text: str, matched: list[str]) -> str:
    if matched:
        return "命中词：" + "、".join(matched[:8])
    ov = lexical_overlap(question, text)
    if ov > 0:
        return f"分词重叠 {ov:.0%}"
    return "词重叠为 0，多半是向量近邻或标题沾边"


def retrieve(question: str, k: int | None = None) -> list[Document]:
    top_k = k or config.TOP_K
    return [item["doc"] for item in retrieve_ranked(question, k=top_k, fetch_k=top_k)]


def format_context(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[资料{i} | {source}]\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)
