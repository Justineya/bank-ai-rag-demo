"""召回之后的精排。

默认 Cross-Encoder（bge-reranker）：对 (问题, 段落) 打分，再截 top-k。
词重叠精排只作教学对照——和 BM25 几乎看同一类信号，经常名次不变。
"""

from __future__ import annotations

from rag import config
from rag.textutil import lexical_overlap, tokenize

LAST_ERROR = ""
_CROSS_ENCODER = None


def rerank_hits(
    question: str,
    hits: list[dict],
    keep: int | None = None,
    mode: str | None = None,
) -> list[dict]:
    keep_n = keep or config.TOP_K
    backend = (mode or config.RERANKER or "bge").lower()
    if not hits:
        return []
    if backend in {"none", "off", "passthrough"}:
        return _passthrough(hits, keep_n)
    if backend in {"lexical", "overlap", "bm25"}:
        return _lexical_rerank(question, hits, keep_n)
    return _cross_encoder_rerank(question, hits, keep_n)


def _passthrough(hits: list[dict], keep_n: int) -> list[dict]:
    kept = []
    for i, hit in enumerate(hits[:keep_n], start=1):
        item = dict(hit)
        item["recall_rank"] = item.get("recall_rank") or i
        item["rerank_rank"] = i
        item["rerank_score"] = float(hit.get("score") or 0.0)
        item["rerank_backend"] = "none"
        kept.append(item)
    return kept


def _lexical_rerank(question: str, hits: list[dict], keep_n: int) -> list[dict]:
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
        expired_penalty = 2.0 if any(m in text for m in ("已废止", "过期数字", "禁止对客")) else 0.0
        item = dict(hit)
        item["recall_rank"] = recall_rank
        item["recall_score"] = hit.get("score")
        item["rerank_score"] = round(
            recall_n * 2.5 + lex * 2.0 + title * 0.4 + extra - missing - expired_penalty, 4
        )
        item["rerank_backend"] = "lexical"
        matched = set(hit.get("matched") or []) | (q_terms & set(tokenize(text)))
        item["matched"] = sorted(matched, key=len, reverse=True)
        scored.append(item)
    scored.sort(key=lambda row: (-row["rerank_score"], row["recall_rank"]))
    return _mark_ranks(scored, keep_n)


def _cross_encoder_rerank(question: str, hits: list[dict], keep_n: int) -> list[dict]:
    global LAST_ERROR, _CROSS_ENCODER
    LAST_ERROR = ""
    try:
        if _CROSS_ENCODER is None:
            from sentence_transformers import CrossEncoder

            _CROSS_ENCODER = CrossEncoder(config.RERANKER_MODEL)
        pairs = [(question, hit["doc"].page_content) for hit in hits]
        scores = _CROSS_ENCODER.predict(pairs)
    except Exception as exc:
        LAST_ERROR = f"bge-reranker 不可用（{exc}），本轮回退词重叠精排。"
        return _lexical_rerank(question, hits, keep_n)

    scored: list[dict] = []
    q_terms = set(tokenize(question))
    for recall_rank, (hit, score) in enumerate(zip(hits, scores), start=1):
        text = hit["doc"].page_content
        expired_penalty = 2.0 if any(m in text for m in ("已废止", "过期数字", "禁止对客")) else 0.0
        item = dict(hit)
        item["recall_rank"] = recall_rank
        item["recall_score"] = hit.get("score")
        item["rerank_score"] = round(float(score) - expired_penalty, 4)
        item["rerank_backend"] = "bge"
        matched = set(hit.get("matched") or []) | (q_terms & set(tokenize(hit["doc"].page_content)))
        item["matched"] = sorted(matched, key=len, reverse=True)
        scored.append(item)
    scored.sort(key=lambda row: (-row["rerank_score"], row["recall_rank"]))
    return _mark_ranks(scored, keep_n)


def _mark_ranks(scored: list[dict], keep_n: int) -> list[dict]:
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
