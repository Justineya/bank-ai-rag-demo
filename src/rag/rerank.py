"""召回之后的精排：用「问题和这段话是否对得上」再打一遍分。

生产里常用 Cross-Encoder / bge-reranker。本教室用词重叠 + 标题加权，
让你看到「先多捞再精排」的位置，而不必再下一个重模型。
"""

from __future__ import annotations

from rag import config
from rag.textutil import lexical_overlap, tokenize


def rerank_hits(question: str, hits: list[dict], keep: int | None = None) -> list[dict]:
    keep_n = keep or config.TOP_K
    if not hits:
        return []
    scored: list[dict] = []
    q_terms = set(tokenize(question))
    for recall_rank, hit in enumerate(hits, start=1):
        doc = hit["doc"]
        text = doc.page_content
        header = text.split("\n", 1)[0]
        lex = lexical_overlap(question, text)
        title = lexical_overlap(question, header)
        extra = 0.5 if q_terms and q_terms <= set(tokenize(text)) else 0.0
        item = dict(hit)
        item["recall_rank"] = recall_rank
        item["recall_score"] = hit.get("score")
        item["rerank_score"] = round(lex * 3.0 + title * 2.0 + extra, 4)
        matched = set(hit.get("matched") or []) | (q_terms & set(tokenize(text)))
        item["matched"] = sorted(matched, key=len, reverse=True)
        scored.append(item)
    scored.sort(key=lambda row: (-row["rerank_score"], row["recall_rank"]))
    kept = scored[:keep_n]
    for i, row in enumerate(kept, start=1):
        row["rerank_rank"] = i
    return kept
