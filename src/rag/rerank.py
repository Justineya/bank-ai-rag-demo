"""召回之后的精排：用「问题和这段话是否对得上」再打一遍分。

生产里常用 Cross-Encoder / bge-reranker。本教室把「召回分」和词重叠混在一起，
避免精排把 BM25 已经排对的结果整盘打乱（两套尺子都只看词时尤其容易）。
"""

from __future__ import annotations

from rag import config
from rag.textutil import lexical_overlap, tokenize


def rerank_hits(question: str, hits: list[dict], keep: int | None = None) -> list[dict]:
    keep_n = keep or config.TOP_K
    if not hits:
        return []
    norms = _recall_norms(hits)
    scored: list[dict] = []
    q_terms = set(tokenize(question))
    for recall_rank, (hit, recall_n) in enumerate(zip(hits, norms), start=1):
        doc = hit["doc"]
        text = doc.page_content
        header = text.split("\n", 1)[0]
        lex = lexical_overlap(question, text)
        title = lexical_overlap(question, header)
        extra = 0.5 if q_terms and q_terms <= set(tokenize(text)) else 0.0
        missing = sum(0.4 for t in q_terms if len(t) >= 3 and t not in set(tokenize(text)))
        item = dict(hit)
        item["recall_rank"] = recall_rank
        item["recall_score"] = hit.get("score")
        item["rerank_score"] = round(recall_n * 2.5 + lex * 2.0 + title * 0.4 + extra - missing, 4)
        matched = set(hit.get("matched") or []) | (q_terms & set(tokenize(text)))
        item["matched"] = sorted(matched, key=len, reverse=True)
        scored.append(item)
    scored.sort(key=lambda row: (-row["rerank_score"], row["recall_rank"]))
    kept = scored[:keep_n]
    for i, row in enumerate(kept, start=1):
        row["rerank_rank"] = i
    return kept


def _recall_norms(hits: list[dict]) -> list[float]:
    raw = [float(hit.get("score") or 0.0) for hit in hits]
    if hits[0].get("score_kind") == "vector_distance":
        lo, hi = min(raw), max(raw)
        span = (hi - lo) or 1.0
        return [(hi - x) / span for x in raw]
    hi = max(raw) or 1.0
    return [x / hi for x in raw]
