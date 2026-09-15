"""生成阶段：Prompt + 检索上下文 + LLM。

没有 API Key 时走抽取式回答，仍然是完整的 RAG 闭环（只是生成器更朴素）。
"""

from __future__ import annotations

import re
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
    """无 LLM 时：用 n-gram 挑一句摘要，并附上 top-1 chunk 原文。"""

    def generate(self, question: str, docs: list[Document]) -> str:
        if not docs:
            return "知识库里没有检索到相关段落，请先运行索引或换一个问法。"
        top = docs[0].page_content.strip()
        sentences = [s for s in _sentences(top) if not s.startswith("#")]
        highlight = ""
        if sentences:
            highlight = max(sentences, key=lambda s: lexical_overlap(question, s))
        summary = f"摘要：{highlight}\n\n" if highlight else ""
        return f"{summary}依据原文：\n{top}\n\n（未调用 LLM，以上为检索片段） (资料1)"


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
    if not has_cite:
        text = text + " (资料1)"
        notes.append("append_cite")
    names = []
    for i, doc in enumerate(docs, start=1):
        src = str(doc.metadata.get("source", "unknown"))
        names.append(f"{i}.{src.rsplit('/', 1)[-1]}")
    text = text + "\n\n【后处理·引用文件】" + "；".join(names)
    notes.append("attach_sources")
    _ = question
    return text, notes


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
