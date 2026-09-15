"""索引阶段：Load → Split → Embed → Store。

对应 LangChain 官方 rag-from-scratch 的 Indexing 一课。
"""

from __future__ import annotations

import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config
from rag.embeddings import build_embeddings


def load_documents(data_dir: Path | None = None) -> list[Document]:
    directory = Path(data_dir or config.DATA_DIR)
    paths = sorted(directory.glob("**/*.md")) + sorted(directory.glob("**/*.txt"))
    docs = [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": str(path)})
        for path in paths
        if path.is_file()
    ]
    if not docs:
        raise FileNotFoundError(f"知识库为空：{directory}")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    # 先按二级标题切开，保证「活期」「随心贷」不会挤在同一个 chunk 里。
    sections: list[Document] = []
    for doc in docs:
        parts = re.split(r"(?=^## )", doc.page_content, flags=re.MULTILINE)
        for part in parts:
            text = part.strip()
            if not text:
                continue
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if len(lines) <= 1 and text.startswith("#"):
                continue
            sections.append(Document(page_content=text, metadata=doc.metadata.copy()))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    return splitter.split_documents(sections)


def get_vectorstore(reset: bool = False) -> Chroma:
    embeddings = build_embeddings()
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    if reset:
        store.delete_collection()
        store = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(config.CHROMA_DIR),
        )
    return store


def build_index(reset: bool = True, data_dir: Path | None = None) -> dict:
    docs = load_documents(data_dir)
    chunks = split_documents(docs)
    store = get_vectorstore(reset=reset)
    ids = store.add_documents(chunks)
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "ids": len(ids),
        "persist_directory": str(config.CHROMA_DIR),
        "embedding_backend": config.EMBEDDING_BACKEND,
    }
