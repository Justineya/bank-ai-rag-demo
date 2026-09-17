"""演示级权限与版本：对客 / 对内、生效日、允许集过滤。"""

from __future__ import annotations

import re
from datetime import date

from langchain_core.documents import Document

from rag import config

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

INTERNAL_MARKS = ("对内", "已废止", "内部邮件", "禁止对客", "偿付准备", "再保对手")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    match = FRONTMATTER_RE.match(text or "")
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    return meta, text[match.end() :]


def infer_audience(text: str, meta: dict | None = None) -> str:
    if meta and meta.get("audience") in {"public", "internal"}:
        return meta["audience"]
    blob = text or ""
    if any(mark in blob for mark in INTERNAL_MARKS):
        return "internal"
    return "public"


def annotate_document(doc: Document) -> Document:
    extra, body = parse_frontmatter(doc.page_content)
    if extra:
        doc.page_content = body.strip() + "\n"
        doc.metadata = {**doc.metadata, **extra}
    audience = infer_audience(doc.page_content, doc.metadata)
    doc.metadata["audience"] = audience
    doc.metadata["audience_label"] = doc.metadata.get("audience_label") or (
        "对内" if audience == "internal" else "对客"
    )
    status = str(doc.metadata.get("status") or ("superseded" if "已废止" in doc.page_content else "active"))
    doc.metadata["status"] = status
    doc.metadata.setdefault("effective_date", extra.get("effective_date", ""))
    return doc


def is_allowed(doc: Document, audience: str | None = None) -> bool:
    role = (audience or getattr(config, "AUDIENCE", "public") or "public").lower()
    if role in {"all", "any", "*"}:
        return True
    doc_role = str(doc.metadata.get("audience") or infer_audience(doc.page_content, doc.metadata))
    if role == "internal":
        return True
    return doc_role == "public"


def filter_docs(docs: list[Document], audience: str | None = None) -> list[Document]:
    return [doc for doc in docs if is_allowed(doc, audience)]


def chunk_id(doc: Document) -> str:
    if doc.metadata.get("chunk_id"):
        return str(doc.metadata["chunk_id"])
    from pathlib import Path

    stem = Path(str(doc.metadata.get("source", "doc"))).stem
    page = doc.metadata.get("page") or 1
    idx = doc.metadata.get("chunk_index") or 0
    return f"{stem}-p{page}-c{idx}"
