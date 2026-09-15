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
CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", ROOT / "chroma_db"))
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "bank_kb")

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
# bm25 | vector
RETRIEVER = os.getenv("RAG_RETRIEVER", "bm25").lower()

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "hashed").lower()
HF_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
AGNES_API_KEY = (os.getenv("AGNES_API_KEY") or os.getenv("AGNES_KEY") or "").strip()
AGNES_BASE_URL = os.getenv("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1").strip()
AGNES_MODEL = os.getenv("AGNES_MODEL", "agnes-2.5-flash").strip()


def reload() -> None:
    """Streamlit secrets 写入环境变量后重新读取。"""
    global OPENAI_API_KEY, OPENAI_MODEL, GROQ_API_KEY, GROQ_MODEL
    global AGNES_API_KEY, AGNES_BASE_URL, AGNES_MODEL
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    AGNES_API_KEY = (os.getenv("AGNES_API_KEY") or os.getenv("AGNES_KEY") or "").strip()
    AGNES_BASE_URL = os.getenv("AGNES_BASE_URL", "https://apihub.agnes-ai.com/v1").strip()
    AGNES_MODEL = os.getenv("AGNES_MODEL", "agnes-2.5-flash").strip()
