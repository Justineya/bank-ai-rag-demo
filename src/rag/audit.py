"""会话审计：问题、命中 chunk、模型、是否拒答，可导出一条 JSON。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from rag import config

_LOG: list[dict] = []


def append_event(event: dict) -> dict:
    row = {
        "id": event.get("id") or str(uuid4()),
        "ts": event.get("ts") or datetime.now(timezone.utc).isoformat(),
        "tenant": event.get("tenant") or getattr(config, "TENANT", "bank"),
        "audience": event.get("audience") or getattr(config, "AUDIENCE", "public"),
        "question": event.get("question") or "",
        "chunk_ids": list(event.get("chunk_ids") or []),
        "sources": list(event.get("sources") or []),
        "model": event.get("model") or "",
        "refused": bool(event.get("refused")),
        "answer_preview": (event.get("answer") or "")[:180],
        "notes": list(event.get("notes") or []),
    }
    _LOG.append(row)
    return row


def last_event() -> dict | None:
    return _LOG[-1] if _LOG else None


def all_events() -> list[dict]:
    return list(_LOG)


def export_json(path: Path | None = None, event: dict | None = None) -> str:
    payload = event or last_event() or {}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if path:
        Path(path).write_text(text, encoding="utf-8")
    return text


def clear() -> None:
    _LOG.clear()
