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

SYSTEM_PROMPT = """你是星河银行的知识库助手。只能根据给定资料回答。
如果资料里没有答案，明确说「资料中没有提到」，不要编造利率、额度或电话。
回答使用简体中文，并在句末用 (资料N) 标出依据。"""


class Generator(Protocol):
    def generate(self, question: str, docs: list[Document]) -> str: ...


class ExtractiveGenerator:
    """无 LLM 时：只摘一句最贴问题的话，原文放到引用卡片里，避免把整页 PDF 糊进答案。"""

    def generate(self, question: str, docs: list[Document]) -> str:
        if not docs:
            return "知识库里没有检索到相关段落，请先运行索引或换一个问法。"
        top = re.sub(r"(\d)\s*\n+\s*", r"\1", docs[0].page_content.strip())
        sentences = [s for s in _sentences(top) if not s.startswith("#")]
        if not sentences:
            return "资料中没有找到可引用的句子。 (资料1)"
        highlight = max(sentences, key=lambda s: lexical_overlap(question, s))
        if lexical_overlap(question, highlight) <= 0:
            return "资料中没有提到与问题对应的内容。"
        return f"{highlight} (资料1)"


class ChatModelGenerator:
    def __init__(self, model):
        self.model = model

    def generate(self, question: str, docs: list[Document]) -> str:
        context = format_context(docs)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"资料：\n{context}\n\n问题：{question}\n\n请作答。"
            ),
        ]
        result = self.model.invoke(messages)
        content = result.content if hasattr(result, "content") else str(result)
        return content if isinstance(content, str) else str(content)


def postprocess_answer(answer: str, question: str, docs: list[Document]) -> tuple[str, list[str]]:
    """生成之后的规则处理：不是改 Prompt，而是检查、补引用、空检索拒答。"""
    notes: list[str] = []
    text = (answer or "").strip()
    if not docs:
        notes.append("empty_retrieve")
        return "资料中没有检索到可用段落，因此不能编造答案。", notes
    if "```" in text:
        text = text.replace("```markdown", "").replace("```", "").strip()
        notes.append("strip_fence")
    has_cite = bool(re.search(r"资料\s*\d+", text))
    if not has_cite and not _is_refusal(text):
        if lexical_overlap(question, docs[0].page_content) > 0:
            text = text + " (资料1)"
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


def _is_refusal(text: str) -> bool:
    blob = text or ""
    return any(mark in blob for mark in ("资料中没有", "不能编造", "没有检索到", "没有找到可引用"))


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


def _sentences(text: str) -> list[str]:
    bits = re.split(r"[。！？\n]+", text)
    return [b.strip(" -•") for b in bits if len(b.strip()) >= 6]
