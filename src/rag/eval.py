"""固定评测集：命中率、拒答正确率、引用是否点名。"""

from __future__ import annotations

import json
from pathlib import Path

from rag import config
from rag.generate import is_refusal
from rag.pipeline import RAGAnswer, ask


def load_eval_items(path: Path | None = None) -> list[dict]:
    data = json.loads(Path(path or config.EVAL_PATH).read_text(encoding="utf-8"))
    items = []
    for row in data.get("questions") or []:
        items.append(row)
        for para in row.get("paraphrases") or []:
            clone = dict(row)
            clone["question"] = para
            clone["id"] = f"{row['id']}::para"
            clone["parent_id"] = row["id"]
            items.append(clone)
    return items


def source_names(result: RAGAnswer) -> list[str]:
    names = []
    for doc in result.sources:
        names.append(Path(str(doc.metadata.get("source", ""))).name)
    for clip in result.clips:
        names.append(str(clip.get("name") or ""))
    return names


def score_result(item: dict, result: RAGAnswer) -> dict:
    names = source_names(result)
    expect = list(item.get("expect_sources") or [])
    allow = bool(item.get("allow_answer"))
    must_refuse = bool(item.get("must_refuse") or not allow)
    refused = bool(result.refused or is_refusal(result.answer))
    hit = True
    if expect:
        hit = any(src in names for src in expect)
    contains = list(item.get("expect_contains") or [])
    blob = (
        result.answer
        + " "
        + " ".join(c.get("quote") or "" for c in result.clips)
        + " "
        + " ".join(d.page_content for d in result.sources)
    )
    contains_ok = all(token in blob for token in contains) if contains else True
    if must_refuse:
        hit = True
        contains_ok = True
    cited = bool(result.clips) or ("资料" in (result.answer or "") and allow)
    refuse_ok = refused if must_refuse else (not refused if allow else True)
    answer_ok = refuse_ok and (contains_ok if allow else True)
    if allow:
        answer_ok = answer_ok and hit
    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "allow_answer": allow,
        "must_refuse": must_refuse,
        "hit": hit,
        "refuse_ok": refuse_ok,
        "cited": cited if allow else (not cited),
        "contains_ok": contains_ok,
        "answer_ok": answer_ok,
        "refused": refused,
        "answer": result.answer,
        "sources": names,
        "generator": result.generator,
        "rerank": result.rerank_backend,
        "timings_ms": result.timings_ms,
        "tags": item.get("tags") or [],
    }


def summarize(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["allow_answer"]]
    refusals = [r for r in rows if r["must_refuse"]]
    hit_rate = _avg(answerable, "hit")
    refuse_rate = _avg(refusals, "refuse_ok")
    cite_rate = _avg(answerable, "cited")
    pass_rate = _avg(rows, "answer_ok")
    return {
        "n": len(rows),
        "n_answerable": len(answerable),
        "n_refuse": len(refusals),
        "hit_rate": hit_rate,
        "refuse_accuracy": refuse_rate,
        "cite_named_rate": cite_rate,
        "pass_rate": pass_rate,
        "rows": rows,
    }


def run_eval(
    items: list[dict] | None = None,
    k: int | None = None,
    retriever: str | None = None,
    reranker: str | None = None,
) -> dict:
    pack = items or load_eval_items()
    rows = []
    for item in pack:
        result = ask(
            item["question"],
            k=k,
            retriever=retriever,
            reranker=reranker,
        )
        rows.append(score_result(item, result))
    return summarize(rows)


def _avg(rows: list[dict], key: str) -> float:
    if not rows:
        return 1.0
    return round(sum(1.0 if r.get(key) else 0.0 for r in rows) / len(rows), 4)
