from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from rag.embeddings import HashedNgramEmbeddings
from rag.generate import ExtractiveGenerator
from rag.ingest import load_documents, split_documents
from rag.pipeline import ask
from rag.retrieve import format_context, retrieve_ranked
from rag.textutil import tokenize

KB = Path(__file__).resolve().parents[1] / "data" / "kb"


def test_domain_terms_tokenize():
    assert "随心贷" in tokenize("随心贷能用来炒股吗？")
    assert "活期" in tokenize("活期利率是多少？")


def test_load_and_split_sample_kb():
    docs = load_documents(KB)
    assert len(docs) >= 5
    chunks = split_documents(docs)
    assert len(chunks) > len(docs)
    types = {str(d.metadata.get("file_type")) for d in docs}
    assert "md" in types
    joined = " ".join(d.page_content for d in docs)
    assert "星河活期" in joined
    assert "随心贷" in joined
    assert "pdf" in types
    assert "docx" in types
    assert "违约金" in joined
    assert "400-000-8888" in joined


def test_hashed_embeddings_are_normalized_and_sensitive():
    emb = HashedNgramEmbeddings(dim=64)
    v1 = emb.embed_query("活期利率")
    v2 = emb.embed_query("活期利率")
    v3 = emb.embed_query("信用卡积分")
    assert v1 == v2
    assert len(v1) == 64
    assert abs(sum(x * x for x in v1) - 1) < 1e-6
    assert v1 != v3


def test_extractive_generator_cites_sources():
    docs = [
        Document(page_content="## 活期储蓄\n星河活期年利率：0.20%。随时存取。"),
        Document(page_content="信用卡账单日是每月 8 日。"),
    ]
    answer = ExtractiveGenerator().generate("活期年利率是多少", docs)
    assert "0.20%" in answer
    assert "资料1" in answer


def test_format_context_includes_index():
    docs = [Document(page_content="hello", metadata={"source": "a.md"})]
    text = format_context(docs)
    assert "[资料1 | a.md]" in text


def test_end_to_end_ask(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RAG_DATA_DIR", str(KB))
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashed")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("AGNES_API_KEY", "")

    from rag import config
    from rag.ingest import build_index

    config.CHROMA_DIR = tmp_path / "chroma"
    config.DATA_DIR = KB
    config.EMBEDDING_BACKEND = "hashed"
    config.RETRIEVER = "bm25"
    config.OPENAI_API_KEY = ""
    config.GROQ_API_KEY = ""
    config.AGNES_API_KEY = ""

    stats = build_index(reset=True, data_dir=KB)
    assert stats["chunks"] > 0

    result = ask("星河活期的年利率是多少？", k=4)
    assert result.sources
    assert result.generator == "extractive"
    assert "0.20%" in result.sources[0].page_content
    assert "活期" in result.sources[0].page_content

    demand = ask("活期利率是多少？", k=4)
    assert "0.20%" in demand.sources[0].page_content

    card = ask("信用卡还款日是哪天？", k=4)
    assert "26" in card.sources[0].page_content

    loan = ask("随心贷能用来炒股吗？", k=4)
    assert "随心贷" in loan.sources[0].page_content
    assert "股市" in loan.sources[0].page_content or "不可用于" in loan.sources[0].page_content

    prepay = ask("提前还房贷要不要违约金？", k=4)
    assert "违约金" in prepay.sources[0].page_content

    pdf_q = ask("满三十六个月提前还款还收违约金吗？", k=4)
    blob = " ".join(d.page_content for d in pdf_q.sources)
    assert "违约金" in blob

    ranked = retrieve_ranked("随心贷能用来炒股吗？", k=2, retriever="bm25")
    assert ranked
    assert ranked[0]["score"] > 0
    assert "随心贷" in ranked[0]["matched"]


def test_build_index_accepts_extra_docs(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_CHROMA_DIR", str(tmp_path / "chroma2"))
    from rag import config
    from rag.ingest import build_index

    config.CHROMA_DIR = tmp_path / "chroma2"
    config.EMBEDDING_BACKEND = "hashed"
    extra = [Document(page_content="## 学员补充\n大厅超时需重新取号。", metadata={"source": "user-note.md", "file_type": "md", "page": 1})]
    stats = build_index(reset=True, data_dir=KB, extra_docs=extra, chunk_size=400, chunk_overlap=40)
    assert stats["chunks"] >= 1
    ranked = retrieve_ranked("大厅超时怎么办？", k=3, retriever="bm25")
    assert ranked
    assert "超时" in ranked[0]["doc"].page_content


def test_build_generator_uses_agnes_when_key_present(monkeypatch):
    monkeypatch.setenv("AGNES_API_KEY", "sk-test-agnes")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    from rag import config
    from rag.generate import ChatModelGenerator, build_generator

    config.reload()
    gen, name = build_generator()
    assert name.startswith("agnes:")
    assert isinstance(gen, ChatModelGenerator)
