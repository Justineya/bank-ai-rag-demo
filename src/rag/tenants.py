"""教室叙事：星河银行 vs 闽信保险。切换知识库、标题与示例问题。"""

from __future__ import annotations

from pathlib import Path

from rag import config
from rag.textutil import load_user_dict

ROOT = Path(__file__).resolve().parents[2]

TENANTS = {
    "bank": {
        "id": "bank",
        "title": "星河银行 RAG 教室",
        "icon": "🏦",
        "caption": "虚构银行制度稿。对照教学流水线，不是生产系统。",
        "data_dir": ROOT / "data" / "kb",
        "eval_path": ROOT / "data" / "eval" / "questions.json",
        "terms": ROOT / "data" / "terms.txt",
        "samples": [
            "活期利率是多少？",
            "信用卡还款日是哪天？",
            "随心贷能用来炒股吗？",
            "提前还房贷要不要违约金？",
            "明天股价会涨吗？",
        ],
        "assistant": "星河银行的知识库助手",
        "domain_hint": "不要编造利率、额度或电话。",
    },
    "minxin": {
        "id": "minxin",
        "title": "闽信保险 RAG 教室",
        "icon": "☂️",
        "caption": "材料摘自闽信集团/闽信保险官网与 2025 年报。演示对客允许集，不是生产核保系统。",
        "data_dir": ROOT / "data" / "minxin" / "kb",
        "eval_path": ROOT / "data" / "minxin" / "eval" / "questions.json",
        "terms": ROOT / "data" / "minxin" / "terms.txt",
        "samples": [
            "闽信保险是哪一年成立的？",
            "AM Best 评级是什么？",
            "港车北上能不能在闽信买车险？",
            "2025 年保险收入是多少？",
            "闽信保险净息差是多少？",
        ],
        "assistant": "闽信保险的知识库助手",
        "domain_hint": "不要编造保费、赔付限额或评级。官网与年报数字冲突时，对客以已审计年报为准。",
    },
}

PRESETS = {
    "nim": ("bank", "星河银行净息差是多少？"),
    "demand": ("bank", "活期利率是多少？"),
    "prepay": ("bank", "提前还房贷要不要违约金？"),
    "northbound": ("minxin", "港车北上能不能在闽信买车险？"),
    "rating": ("minxin", "闽信保险的 AM Best 评级是什么？"),
    "premium": ("minxin", "2025 年保险收入是多少？"),
}


def current() -> dict:
    tid = (getattr(config, "TENANT", None) or "bank").lower()
    return TENANTS.get(tid, TENANTS["bank"])


def apply_tenant(tenant_id: str) -> dict:
    profile = TENANTS.get((tenant_id or "bank").lower(), TENANTS["bank"])
    config.TENANT = profile["id"]
    config.DATA_DIR = Path(profile["data_dir"])
    config.UPLOAD_DIR = config.DATA_DIR / "uploads"
    config.EVAL_PATH = Path(profile["eval_path"])
    config.COLLECTION_NAME = f"{profile['id']}_kb_{config.EMBEDDING_BACKEND}"
    load_user_dict(Path(profile["terms"]))
    return profile


def resolve_preset(name: str | None) -> tuple[str, str] | None:
    if not name:
        return None
    return PRESETS.get(name.strip().lower())


def parse_launch_params(qp: dict | None) -> dict:
    """途港课件 / 直链：tenant、preset、q、from=tugang、lesson=banking-ai。"""
    raw = qp or {}
    get = lambda k: str(raw.get(k) or "").strip()
    tenant = get("tenant") or get("brand")
    preset_name = get("preset")
    preset = resolve_preset(preset_name)
    question = get("q") or get("question")
    lesson = get("lesson")
    origin = get("from")
    from_tugang = origin == "tugang" or "banking-ai" in lesson
    if preset:
        tenant = tenant or preset[0]
        question = question or preset[1]
    if "banking-ai" in lesson:
        tenant = tenant or "bank"
        question = question or "活期利率是多少？"
    if tenant not in TENANTS:
        tenant = None
    return {
        "tenant": tenant,
        "question": question or None,
        "from_tugang": from_tugang,
        "step": 4 if from_tugang else None,
        "preset": preset_name or None,
    }
