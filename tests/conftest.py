"""Tests never download bge models or call paid APIs."""

from __future__ import annotations

import os

os.environ["EMBEDDING_BACKEND"] = "hashed"
os.environ["RAG_RERANKER"] = "lexical"
os.environ["RAG_RETRIEVER"] = "bm25"
os.environ.setdefault("AGNES_API_KEY", "")
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
