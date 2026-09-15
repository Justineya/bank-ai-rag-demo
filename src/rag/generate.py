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

SYSTEM_PROMPT = """你是星河银行的知识库助手。只能根据给定资料回答。
如果资料里没有答案，明确说「资料中没有提到」，不要编造利率、额度或电话。
回答使用简体中文，并在句末用 (资料N) 标出依据。"""


class Generator(Protocol):
    def generate(self, question: str, docs: list[Document]) -> str: ...


class ExtractiveGenerator:
    """把与问题词重叠最多的句子抽出来，用于无 Key 环境。"""

    def generate(self, question: str, docs: list[Document]) -> str:
        if not docs:
            return "知识库里没有检索到相关段落，请先运行索引或换一个问法。"
        scored: list[tuple[float, int, str]] = []
        query_tokens = _tokens(question)
        for i, doc in enumerate(docs, start=1):
            for sent in _sentences(doc.page_content):
                overlap = len(query_tokens & _tokens(sent))
                if overlap:
                    scored.append((overlap / (1 + abs(len(sent) - 24)), i, sent))
        if not scored:
            preview = docs[0].page_content.strip().replace("\n", " ")
            return f"未找到高度重合的句子，最相关片段如下：{preview[:180]} (资料1)"
        scored.sort(key=lambda x: x[0], reverse=True)
        used_sents: list[str] = []
        cites: list[int] = []
        for _, idx, sent in scored[:3]:
            if sent not in used_sents:
                used_sents.append(sent)
                cites.append(idx)
        cite = " ".join(f"(资料{i})" for i in dict.fromkeys(cites))
        return " ".join(used_sents) + f" {cite}"


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
        return result.content if hasattr(result, "content") else str(result)


def build_generator() -> tuple[Generator, str]:
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


def _tokens(text: str) -> set[str]:
    parts = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+", text.lower())
    return {p for p in parts if p.strip()}


def _sentences(text: str) -> list[str]:
    bits = re.split(r"[。！？\n]+", text)
    return [b.strip(" -•") for b in bits if len(b.strip()) >= 6]
