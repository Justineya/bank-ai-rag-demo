"""把 Markdown / PDF / Word 读成 LangChain Document。"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document


def load_path(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return [
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": str(path), "file_type": suffix.lstrip("."), "page": 1},
            )
        ]
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    return []


def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs: list[Document] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": str(path), "file_type": "pdf", "page": i},
            )
        )
    return docs


def _load_docx(path: Path) -> list[Document]:
    from docx import Document as DocxFile

    parsed = DocxFile(str(path))
    blocks: list[str] = []
    current: list[str] = []
    pending_title = ""
    for para in parsed.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "") if para.style else ""
        is_heading = "Heading" in style or style.startswith("Title")
        if is_heading:
            if current:
                blocks.append("\n".join(current))
                current = []
            pending_title = text
            continue
        if pending_title:
            current.append(pending_title)
            pending_title = ""
        current.append(text)
    if pending_title:
        current.append(pending_title)
    if current:
        blocks.append("\n".join(current))
    return [
        Document(
            page_content=block,
            metadata={"source": str(path), "file_type": "docx", "page": i},
        )
        for i, block in enumerate(blocks, start=1)
        if block.strip()
    ]
