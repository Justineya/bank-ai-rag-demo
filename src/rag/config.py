"""RAG 教学 Demo：配置集中在这里，方便对照流水线每一层。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT / "data" / "kb"))
UPLOAD_DIR = Path(os.getenv("RAG_UPLOAD_DIR", DATA_DIR / "uploads"))
CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", ROOT / "chroma_db"))
EVAL_PATH = Path(os.getenv("RAG_EVAL_PATH", ROOT / "data" / "eval" / "questions.json"))

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
FETCH_K = int(os.getenv("RAG_FETCH_K", "16"))
# bm25 | vector | hybrid
RETRIEVER = os.getenv("RAG_RETRIEVER", "bm25").lower()
# bge | lexical | none
RERANKER = os.getenv("RAG_RERANKER", "bge").lower()
RERANKER_MODEL = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")

# huggingface（默认中文句向量）| hashed（教学开关，不下载模型）
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "huggingface").lower()
COLLECTION_NAME = os.getenv("RAG_COLLECTION", f"bank_kb_{EMBEDDING_BACKEND}")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

TENANT = os.getenv("RAG_TENANT", "bank").lower()
AUDIENCE = os.getenv("RAG_AUDIENCE", "public").lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
AGNES_API_KEY = (os.getenv("AGNES_API_KEY") or os.getenv("AGNES_KEY") or "").strip()
AGNES_BASE_URL = os.getenv("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1").strip()
AGNES_MODEL = os.getenv("AGNES_MODEL", "agnes-2.5-flash").strip()


def reload() -> None:
    """Streamlit secrets / 环境变量变更后重新读取。"""
    global OPENAI_API_KEY, OPENAI_MODEL, GROQ_API_KEY, GROQ_MODEL
    global AGNES_API_KEY, AGNES_BASE_URL, AGNES_MODEL
    global EMBEDDING_BACKEND, COLLECTION_NAME, HF_EMBEDDING_MODEL
    global RETRIEVER, RERANKER, RERANKER_MODEL, TOP_K, FETCH_K
    global TENANT, AUDIENCE
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    AGNES_API_KEY = (os.getenv("AGNES_API_KEY") or os.getenv("AGNES_KEY") or "").strip()
    AGNES_BASE_URL = os.getenv("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1").strip()
    AGNES_MODEL = os.getenv("AGNES_MODEL", "agnes-2.5-flash").strip()
    EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "huggingface").lower()
    HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    COLLECTION_NAME = os.getenv("RAG_COLLECTION", f"bank_kb_{EMBEDDING_BACKEND}")
    RETRIEVER = os.getenv("RAG_RETRIEVER", "bm25").lower()
    RERANKER = os.getenv("RAG_RERANKER", "bge").lower()
    RERANKER_MODEL = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
    TOP_K = int(os.getenv("RAG_TOP_K", "4"))
    FETCH_K = int(os.getenv("RAG_FETCH_K", "16"))
    TENANT = os.getenv("RAG_TENANT", "bank").lower()
    AUDIENCE = os.getenv("RAG_AUDIENCE", "public").lower()
