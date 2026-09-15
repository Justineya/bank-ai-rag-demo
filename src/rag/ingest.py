"""索引阶段：Load → Split → Embed → Store。

对应 LangChain 官方 rag-from-scratch 的 Indexing 一课。
"""

from __future__ import annotations

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
    # 中文没有空格分词，所以分隔符里加入常见标点，避免整篇落进一个 chunk。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    return splitter.split_documents(docs)


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
