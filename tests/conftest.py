"""Tests never download bge models or call paid APIs."""

from __future__ import annotations

import os

os.environ["EMBEDDING_BACKEND"] = "hashed"
os.environ["RAG_RERANKER"] = "lexical"
os.environ["RAG_RETRIEVER"] = "bm25"
os.environ["RAG_TENANT"] = "bank"
os.environ["RAG_AUDIENCE"] = "all"
os.environ.setdefault("AGNES_API_KEY", "")
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from rag import config  # noqa: E402
from rag.tenants import apply_tenant  # noqa: E402

config.reload()
apply_tenant("bank")
config.AUDIENCE = "all"
