"""冷启动：首页显示唤醒中，并预热索引，避免只剩转圈。"""

from __future__ import annotations

from rag.ingest import ensure_index
from rag.retrieve import retrieve_ranked
from rag.tenants import current


def warmup_index(chunk_size: int | None = None, chunk_overlap: int | None = None) -> dict:
    from rag.ingest import reset_persist_dir

    try:
        stats = ensure_index(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception:
        reset_persist_dir()
        stats = ensure_index(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    probe = (current().get("samples") or ["唤醒"])[0]
    try:
        retrieve_ranked(probe, k=2, fetch_k=4)
        stats["warmed"] = True
    except Exception:
        stats["warmed"] = False
    return stats
