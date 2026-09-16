"""索引阶段：Load → Split → Embed → Store。

对应 LangChain 官方 rag-from-scratch 的 Indexing 一课。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config
from rag.embeddings import build_embeddings
from rag.loaders import SUPPORTED_SUFFIXES, load_path
from time import perf_counter


def load_documents(data_dir: Path | None = None) -> list[Document]:
    directory = Path(data_dir or config.DATA_DIR)
    upload_root = Path(os.getenv("RAG_UPLOAD_DIR", directory / "uploads"))
    roots = [directory, upload_root]
    all_files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            key = path.resolve()
            if key in seen:
                continue
            seen.add(key)
            all_files.append(path)
    office_stems = {p.stem for p in all_files if p.suffix.lower() in {".pdf", ".docx"}}
    docs: list[Document] = []
    for path in sorted(all_files):
        # 有 PDF/Word 成品时不再重复读同名 Markdown（Markdown 只作导出源）。
        if path.suffix.lower() in {".md", ".txt"} and path.stem in office_stems:
            continue
        docs.extend(load_path(path))
    if not docs:
        raise FileNotFoundError(f"知识库为空：{directory}")
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
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
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(sections)
    overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    prev_text = ""
    prev_source = ""
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        heading = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        shared = _shared_prefix_from_prev(prev_text, text) if prev_source == str(chunk.metadata.get("source")) else ""
        chunk.metadata = {
            **chunk.metadata,
            "chunk_index": i + 1,
            "section": heading[:80],
            "overlap_chars": len(shared),
            "overlap_preview": shared[:80],
            "chunk_size_setting": chunk_size or config.CHUNK_SIZE,
            "chunk_overlap_setting": overlap,
        }
        prev_text = text
        prev_source = str(chunk.metadata.get("source"))
    return chunks


def _shared_prefix_from_prev(prev: str, curr: str) -> str:
    """当前 chunk 开头与上一段结尾重合的文字（用来看见 overlap）。"""
    if not prev or not curr:
        return ""
    max_n = min(len(prev), len(curr))
    for n in range(max_n, 0, -1):
        if curr.startswith(prev[-n:]):
            return prev[-n:]
    return ""


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


def build_index(
    reset: bool = True,
    data_dir: Path | None = None,
    extra_docs: list[Document] | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    extra = extra_docs or []
    t0 = perf_counter()
    docs = load_documents(data_dir) + extra
    file_types = sorted({str(d.metadata.get("file_type", "unknown")) for d in docs})
    chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    t1 = perf_counter()
    store = get_vectorstore(reset=reset)
    ids = store.add_documents(chunks)
    t2 = perf_counter()
    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "ids": len(ids),
        "persist_directory": str(config.CHROMA_DIR),
        "embedding_backend": config.EMBEDDING_BACKEND,
        "vector_store": "chroma",
        "file_types": file_types,
        "split_ms": round((t1 - t0) * 1000, 1),
        "embed_ms": round((t2 - t1) * 1000, 1),
    }


def count_indexed() -> int:
    store = get_vectorstore(reset=False)
    raw = store.get(include=["documents"])
    return len(raw.get("documents") or [])


def ensure_index(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict:
    """仓库空时才建。用户不必先「添加」任何资料。"""
    n = count_indexed()
    if n > 0:
        return {
            "documents": len(load_documents()),
            "chunks": n,
            "ids": n,
            "persist_directory": str(config.CHROMA_DIR),
            "embedding_backend": config.EMBEDDING_BACKEND,
            "vector_store": "chroma",
            "rebuilt": False,
        }
    stats = build_index(reset=True, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    stats["rebuilt"] = True
    return stats


ALLOWED_UPLOAD_SUFFIXES = set(SUPPORTED_SUFFIXES)


def _upload_dir() -> Path:
    env = os.getenv("RAG_UPLOAD_DIR")
    folder = Path(env) if env else Path(config.DATA_DIR) / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_uploaded_file(filename: str, data: bytes) -> Path:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise ValueError(f"不支持 {suffix}，请上传 pdf / docx / md / txt / png / jpg。")
    folder = _upload_dir()
    folder.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9._\u4e00-\u9fff-]+", "_", Path(filename).stem)[:80] or "file"
    dest = folder / f"{stem}{suffix}"
    dest.write_bytes(data)
    return dest


def list_uploads() -> list[Path]:
    folder = _upload_dir()
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_UPLOAD_SUFFIXES
    )


def indexed_source_names() -> set[str]:
    store = get_vectorstore(reset=False)
    raw = store.get(include=["metadatas"])
    names: set[str] = set()
    for meta in raw.get("metadatas") or []:
        names.add(Path(str((meta or {}).get("source", ""))).name)
    return names


def uploads_missing_from_index() -> list[str]:
    indexed = indexed_source_names()
    missing = []
    for path in list_uploads():
        if path.name in indexed:
            continue
        if extracted_chars(path) == 0:
            continue
        missing.append(path.name)
    return missing


def extracted_chars(path: Path) -> int:
    return sum(len(doc.page_content.strip()) for doc in load_path(path))


def vector_inventory() -> dict:
    """按文件汇总向量库，避免预览只截前 20 条时以为上传文件没进去。"""
    store = get_vectorstore(reset=False)
    raw = store.get(include=["documents", "metadatas"])
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    buckets: dict[str, dict] = {}
    for text, meta in zip(documents, metadatas):
        meta = meta or {}
        name = Path(str(meta.get("source", ""))).name or "unknown"
        row = buckets.setdefault(
            name,
            {"source": name, "file_type": str(meta.get("file_type") or ""), "chunks": 0, "chars": 0},
        )
        row["chunks"] += 1
        row["chars"] += len(text or "")
        if not row["file_type"]:
            row["file_type"] = str(meta.get("file_type") or "")
    files = sorted(buckets.values(), key=lambda r: r["source"])
    upload_names = {p.name for p in list_uploads()}
    for row in files:
        row["uploaded"] = row["source"] in upload_names
    missing = uploads_missing_from_index()
    return {
        "total_chunks": len(documents),
        "total_files": len(files),
        "backend": config.EMBEDDING_BACKEND,
        "collection": config.COLLECTION_NAME,
        "files": files,
        "missing_uploads": missing,
    }


def preview_vectors(limit: int = 15, offset: int = 0, source: str | None = None) -> dict:
    store = get_vectorstore(reset=False)
    raw = store.get(include=["documents", "metadatas", "embeddings"])
    ids = raw.get("ids") or []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    embeddings = raw.get("embeddings")
    if embeddings is None:
        embeddings = []
    dim = len(embeddings[0]) if len(embeddings) else 0
    picked: list[int] = []
    want = (source or "").strip()
    for i, meta in enumerate(metadatas):
        name = Path(str((meta or {}).get("source", ""))).name
        if want and name != want:
            continue
        picked.append(i)
    slice_ids = picked[offset : offset + limit]
    rows = []
    for i in slice_ids:
        meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
        vec = [float(x) for x in embeddings[i]] if i < len(embeddings) else []
        text = documents[i] if i < len(documents) else ""
        norm = sum(x * x for x in vec) ** 0.5 if vec else 0.0
        rows.append(
            {
                "id": ids[i] if i < len(ids) else "",
                "source": Path(str(meta.get("source", ""))).name,
                "file_type": meta.get("file_type", ""),
                "page": meta.get("page", ""),
                "chars": len(text or ""),
                "dim": len(vec),
                "norm": round(norm, 4),
                "vector_head": [round(x, 4) for x in vec[:12]],
                "preview": (text or "").replace("\n", " ")[:160],
                "text": text,
            }
        )
    return {
        "total": len(ids),
        "filtered": len(picked),
        "dim": dim,
        "backend": config.EMBEDDING_BACKEND,
        "rows": rows,
    }
