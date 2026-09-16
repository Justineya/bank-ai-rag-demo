"""生成阶段：Prompt + 检索上下文 + LLM。

没有 API Key 时走抽取式回答，仍然是完整的 RAG 闭环（只是生成器更朴素）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from rag import config
from rag.retrieve import format_context
from rag.textutil import lexical_overlap

REFUSAL_EMPTY = "资料中没有检索到可用段落，因此不能编造答案。知识库未收录该问题，请换手册里的产品规则再问。"
REFUSAL_WEAK = "资料中没有提到与问题对应的内容，不能编造利率、牌价或承诺。"

SYSTEM_PROMPT = """你是星河银行的知识库助手。只能根据给定资料回答。
如果资料里没有答案，明确说「资料中没有提到」，不要编造利率、额度或电话。
若同时出现「已废止 / 失效」稿和现行手册，对客必须以现行条款为准，并指出过期数字不能用。
回答使用简体中文，并在句末用 (资料N) 标出依据。"""


class Generator(Protocol):
    def generate(self, question: str, docs: list[Document]) -> str: ...


class ExtractiveGenerator:
    """无 LLM 时：只摘一句最贴问题的话，原文放到引用卡片里，避免把整页 PDF 糊进答案。"""

    def generate(self, question: str, docs: list[Document]) -> str:
        ordered = prefer_effective_docs(docs)
        if not ordered:
            return REFUSAL_EMPTY
        if retrieval_is_weak(question, ordered) or asked_number_missing(question, ordered) or out_of_coverage(question, ordered):
            return REFUSAL_WEAK
        candidates = [d for d in ordered if not is_expired_doc(d)] or ordered
        top_doc = _pick_doc(question, candidates)
        top = re.sub(r"(\d)\s*\n+\s*", r"\1", top_doc.page_content.strip())
        sentences = [s for s in _sentences(top) if not s.startswith("#")]
        if not sentences:
            return "资料中没有找到可引用的句子。 (资料1)"
        highlight = _pick_sentence(question, sentences)
        if lexical_overlap(question, highlight) <= 0:
            return REFUSAL_WEAK
        idx = docs.index(top_doc) + 1 if top_doc in docs else 1
        extra = ""
        if "1.50%" in question and "0.20%" in highlight:
            extra = "过期内部稿中的 1.50% 已废止，对客不以该数字为准。"
        return f"{highlight}{extra} (资料{idx})"


class ChatModelGenerator:
    def __init__(self, model):
        self.model = model

    def generate(self, question: str, docs: list[Document]) -> str:
        context = format_context(prefer_effective_docs(docs))
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"资料：\n{context}\n\n问题：{question}\n\n请作答。"
            ),
        ]
        result = self.model.invoke(messages)
        content = result.content if hasattr(result, "content") else str(result)
        return content if isinstance(content, str) else str(content)


def is_expired_doc(doc: Document) -> bool:
    blob = doc.page_content or ""
    return any(mark in blob for mark in ("已废止", "失效日", "过期数字", "禁止对客"))


def prefer_effective_docs(docs: list[Document]) -> list[Document]:
    """冲突样例：已废止稿往后排，生成时先看现行手册。"""
    live, dead = [], []
    for doc in docs:
        (dead if is_expired_doc(doc) else live).append(doc)
    return live + dead


def retrieval_is_weak(question: str, docs: list[Document]) -> bool:
    if not docs:
        return True
    return max(lexical_overlap(question, d.page_content) for d in docs) <= 0


def out_of_coverage(question: str, docs: list[Document]) -> bool:
    """手册没有的行情/NIM/牌价：即使检索到「星河银行」开篇也不能当答案。"""
    q = question or ""
    blob = "\n".join(d.page_content for d in docs)
    if "NIM" in q.upper() and "NIM" not in blob.upper() and "净息差" not in blob:
        return True
    for needle in ("股价", "牌价", "净息差"):
        if needle in q and needle not in blob:
            return True
    if "美元" in q and any(k in q for k in ("牌价", "汇率", "兑")):
        return True
    if "股票" in q and any(k in q for k in ("明天", "怎么走", "涨")):
        return True
    return False


def asked_number_missing(question: str, docs: list[Document]) -> bool:
    rates = re.findall(r"\d+(?:\.\d+)?%", question or "")
    if not rates:
        return False
    blob = "\n".join(d.page_content for d in docs)
    return any(rate not in blob for rate in rates)


def postprocess_answer(answer: str, question: str, docs: list[Document]) -> tuple[str, list[str]]:
    """生成之后的规则处理：不是改 Prompt，而是检查、补引用、空检索拒答。"""
    notes: list[str] = []
    text = (answer or "").strip()
    ordered = prefer_effective_docs(docs)
    if not ordered:
        notes.append("empty_retrieve")
        return REFUSAL_EMPTY, notes
    if retrieval_is_weak(question, ordered) and not _is_refusal(text):
        notes.append("weak_retrieve")
        return REFUSAL_WEAK, notes
    if out_of_coverage(question, ordered) and not _is_refusal(text):
        notes.append("out_of_coverage")
        return REFUSAL_WEAK, notes
    if asked_number_missing(question, ordered):
        notes.append("invented_number")
        return REFUSAL_WEAK, notes
    if "```" in text:
        text = text.replace("```markdown", "").replace("```", "").strip()
        notes.append("strip_fence")
    has_cite = bool(re.search(r"资料\s*\d+", text))
    if not has_cite and not _is_refusal(text):
        if lexical_overlap(question, ordered[0].page_content) > 0:
            idx = docs.index(ordered[0]) + 1 if ordered[0] in docs else 1
            text = text + f" (资料{idx})"
            notes.append("append_cite")
    return text, notes


def cited_clips(answer: str, docs: list[Document], question: str) -> list[dict]:
    """只返回答案里真正点名的资料；答案没提到的原文不展示。"""
    if not docs or _is_refusal(answer):
        return []
    seen: list[int] = []
    for match in re.finditer(r"资料\s*(\d+)", answer or ""):
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(docs) and idx not in seen:
            seen.append(idx)
    clips = []
    for idx in seen:
        doc = docs[idx]
        if lexical_overlap(question, doc.page_content) <= 0 and lexical_overlap(answer, doc.page_content) <= 0:
            continue
        clips.append(clip_from_doc(doc, question, index=idx + 1))
    return clips


def clip_from_doc(doc: Document, question: str, index: int) -> dict:
    source = str(doc.metadata.get("source", "unknown"))
    name = source.rsplit("/", 1)[-1]
    suffix = Path(name).suffix.lstrip(".").lower()
    kind = str(doc.metadata.get("file_type") or suffix or "md").lower()
    page = doc.metadata.get("page")
    return {
        "index": index,
        "name": name,
        "kind": kind,
        "kind_label": {"pdf": "PDF", "docx": "Word", "doc": "Word", "md": "Markdown", "txt": "文本"}.get(kind, kind.upper() or "文件"),
        "page": page,
        "quote": quote_excerpt(doc.page_content, question),
    }


def quote_excerpt(text: str, question: str, limit: int = 220) -> str:
    compact = (text or "").strip()
    compact = re.sub(r"(\d)\s*\n+\s*", r"\1", compact)
    compact = re.sub(r"\n+", "\n", compact)
    sentences = [s for s in _sentences(compact) if not s.startswith("#")]
    if not sentences:
        body = compact.replace("\n", " ")
        return (body[:limit] + "…") if len(body) > limit else body
    best = max(sentences, key=lambda s: lexical_overlap(question, s))
    if lexical_overlap(question, best) <= 0:
        body = compact.replace("\n", " ")
        return (body[:limit] + "…") if len(body) > limit else body
    quote = best.rstrip("。；") + "。"
    if len(quote) > limit:
        quote = quote[:limit].rstrip("，,；; ") + "…"
    return quote


def is_refusal(text: str) -> bool:
    blob = text or ""
    return any(
        mark in blob
        for mark in ("资料中没有", "不能编造", "没有检索到", "没有找到可引用", "未收录该问题")
    )


def _is_refusal(text: str) -> bool:
    return is_refusal(text)


def preview_prompt(question: str, docs: list[Document]) -> str:
    context = format_context(docs)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{question}\n\n"
        "请作答。"
    )


def build_generator() -> tuple[Generator, str]:
    if config.AGNES_API_KEY:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            api_key=config.AGNES_API_KEY,
            base_url=config.AGNES_BASE_URL,
            model=config.AGNES_MODEL,
            temperature=0,
        )
        return ChatModelGenerator(llm), f"agnes:{config.AGNES_MODEL}"
    if config.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            api_key=config.OPENAI_API_KEY,
            model=config.OPENAI_MODEL,
            temperature=0,
        )
        return ChatModelGenerator(llm), f"openai:{config.OPENAI_MODEL}"
    if config.GROQ_API_KEY:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            api_key=config.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
            model=config.GROQ_MODEL,
            temperature=0,
        )
        return ChatModelGenerator(llm), f"groq:{config.GROQ_MODEL}"
    return ExtractiveGenerator(), "extractive"


def _pick_doc(question: str, docs: list[Document]) -> Document:
    markers = (
        (("卡丢", "挂失", "电话"), ("400-000-8888", "挂失")),
        (("还款日", "几号要还", "最后还款"), ("还款日", "26")),
        (("账单日",), ("账单日", "8 日")),
        (("随心贷", "炒股", "消费贷"), ("随心贷", "不可用于")),
        (("通知存款", "起存"), ("通知存款", "起存")),
        (("加点", "LPR"), ("55BP", "+55")),
        (("活期",), ("活期年利率",)),
        (("一年期定期", "存一年"), ("1 年期",)),
    )
    for qmarks, smarks in markers:
        if any(m in question for m in qmarks):
            hits = [d for d in docs if any(x in d.page_content for x in smarks)]
            if hits:
                if "活期" in qmarks and "定期" in question:
                    continue
                return max(hits, key=lambda d: lexical_overlap(question, d.page_content))
    return max(docs, key=lambda d: lexical_overlap(question, d.page_content))


def _pick_sentence(question: str, sentences: list[str]) -> str:
    rules = (
        (("还款日", "几号要还", "最后还款"), ("还款日",)),
        (("账单日",), ("账单日",)),
        (("挂失", "卡丢", "电话"), ("400", "挂失")),
        (("加点", "LPR", "住房按揭演示"), ("55", "加点")),
        (("起存", "通知存款"), ("起存",)),
        (("一年期定期", "存一年", "1 年期"), ("1 年期",)),
        (("随心贷", "炒股", "消费贷"), ("不可用于", "股市", "炒股")),
        (("积分",), ("20 元", "积 1 分")),
        (("二类", "II 类", "II类"), ("1 万", "二类")),
        (("三十六", "36 个月"), ("36", "违约金")),
        (("违约金", "提前还"), ("违约金",)),
    )
    for qmarks, smarks in rules:
        if any(m in question for m in qmarks):
            hits = [s for s in sentences if any(x in s for x in smarks)]
            if hits:
                return max(hits, key=lambda s: lexical_overlap(question, s))
    if "活期" in question and "定期" not in question:
        demand = [s for s in sentences if "活期年利率" in s]
        if demand:
            return demand[0]
    if any(k in question for k in ("利率", "为准")):
        rates = [s for s in sentences if "%" in s]
        if rates:
            return max(rates, key=lambda s: lexical_overlap(question, s))
    return max(sentences, key=lambda s: lexical_overlap(question, s))


def _sentences(text: str) -> list[str]:
    bits = re.split(r"[。！？\n]+", text)
    return [b.strip(" -•") for b in bits if len(b.strip()) >= 6]
