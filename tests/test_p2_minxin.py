from __future__ import annotations

import json
from pathlib import Path

from rag import config
from rag.acl import annotate_document, filter_docs, is_allowed
from rag.audit import append_event, clear, export_json, last_event
from rag.ingest import build_index, load_documents
from rag.pipeline import ask
from rag.retrieve import retrieve_ranked
from rag.tenants import apply_tenant, parse_launch_params
from rag.warmup import warmup_index
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
MINXIN_KB = ROOT / "data" / "minxin" / "kb"


def _minxin(tmp_path, monkeypatch, audience: str = "public"):
    monkeypatch.setenv("RAG_CHROMA_DIR", str(tmp_path / "chroma_minxin"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "hashed")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("AGNES_API_KEY", "")
    config.CHROMA_DIR = tmp_path / "chroma_minxin"
    config.EMBEDDING_BACKEND = "hashed"
    config.RETRIEVER = "bm25"
    config.RERANKER = "lexical"
    config.OPENAI_API_KEY = ""
    config.GROQ_API_KEY = ""
    config.AGNES_API_KEY = ""
    apply_tenant("minxin")
    config.AUDIENCE = audience
    stats = build_index(reset=True, data_dir=MINXIN_KB)
    assert stats["chunks"] > 0
    return stats


def _restore_bank():
    apply_tenant("bank")
    config.AUDIENCE = "all"


def test_frontmatter_audience_and_effective_date():
    docs = load_documents(MINXIN_KB)
    by_name = {Path(str(d.metadata.get("source", ""))).name: d for d in docs}
    assert by_name["01-group-public.md"].metadata["audience"] == "public"
    assert by_name["04-capital-internal.md"].metadata["audience"] == "internal"
    assert by_name["03-ar2025-insurance.md"].metadata.get("effective_date")
    assert by_name["05-flash-superseded.md"].metadata["status"] == "superseded"
    public_only = filter_docs(docs, "public")
    names = {Path(str(d.metadata.get("source", ""))).name for d in public_only}
    assert "04-capital-internal.md" not in names
    assert "05-flash-superseded.md" not in names
    assert "02-insurance-public.md" in names
    assert all(is_allowed(d, "internal") for d in docs)


def test_infer_audience_without_yaml():
    doc = annotate_document(
        Document(page_content="权限：对内\n状态：已废止。不得对客引用。", metadata={"source": "mail.md"})
    )
    assert doc.metadata["audience"] == "internal"


def test_public_acl_hides_internal_chunks(tmp_path, monkeypatch):
    _minxin(tmp_path, monkeypatch, audience="public")
    try:
        hits = retrieve_ranked("再保对手信用评级是什么？", k=6, retriever="bm25")
        blob = "\n".join(h["doc"].page_content for h in hits)
        names = {Path(str(h["doc"].metadata.get("source", ""))).name for h in hits}
        assert "04-capital-internal.md" not in names
        assert "标准普尔" not in blob
        assert "1,841" not in blob
    finally:
        _restore_bank()


def test_internal_acl_can_see_solvency(tmp_path, monkeypatch):
    _minxin(tmp_path, monkeypatch, audience="internal")
    try:
        hits = retrieve_ranked("再保对手信用评级是什么？", k=6, retriever="bm25")
        blob = "\n".join(h["doc"].page_content for h in hits)
        assert "A－" in blob or "A-" in blob
    finally:
        _restore_bank()


def test_minxin_extractive_official_numbers(tmp_path, monkeypatch):
    _minxin(tmp_path, monkeypatch, audience="public")
    try:
        founded = ask("闽信保险是哪一年成立的？", k=4, reranker="lexical")
        assert "1974" in founded.answer
        assert not founded.refused
        rating = ask("闽信保险的 AM Best 评级是什么？", k=4, reranker="lexical")
        assert "B++" in rating.answer
        premium = ask("2025 年闽信保险的保险收入是多少？", k=4, reranker="lexical")
        assert "19,663" in premium.answer
        assert "3 亿" not in premium.answer
        nim = ask("闽信保险净息差是多少？", k=4, reranker="lexical")
        assert nim.refused
    finally:
        _restore_bank()


def test_audit_export_json_fields(tmp_path, monkeypatch):
    _minxin(tmp_path, monkeypatch, audience="public")
    try:
        clear()
        result = ask("闽信保险是哪一年成立的？", k=4, reranker="lexical")
        event = last_event()
        assert event is not None
        assert event["question"] == "闽信保险是哪一年成立的？"
        assert event["chunk_ids"]
        assert event["model"]
        assert "refused" in event
        assert event["tenant"] == "minxin"
        dumped = json.loads(export_json(event=event))
        assert dumped["id"] == event["id"]
        assert dumped["chunk_ids"] == result.chunk_ids
        row = append_event({"question": "ping", "chunk_ids": ["x-p1-c1"], "model": "extractive", "refused": True})
        assert row["refused"] is True
    finally:
        _restore_bank()
        clear()


def test_apply_tenant_switches_collection_and_eval():
    profile = apply_tenant("minxin")
    try:
        assert config.TENANT == "minxin"
        assert "minxin" in str(config.DATA_DIR)
        assert config.COLLECTION_NAME.startswith("minxin_kb_")
        assert "minxin" in str(config.EVAL_PATH)
        assert profile["samples"]
    finally:
        _restore_bank()
        assert config.COLLECTION_NAME.startswith("bank_kb_")


def test_tugang_query_params_nim_and_demand():
    nim = parse_launch_params({"from": "tugang", "lesson": "banking-ai", "preset": "nim"})
    assert nim["from_tugang"] is True
    assert nim["tenant"] == "bank"
    assert nim["step"] == 4
    assert "净息差" in (nim["question"] or "")
    demand = parse_launch_params({"from": "tugang", "lesson": "banking-ai", "q": "活期利率是多少？"})
    assert demand["question"] == "活期利率是多少？"
    north = parse_launch_params({"preset": "northbound"})
    assert north["tenant"] == "minxin"
    assert "港车北上" in (north["question"] or "")


def test_warmup_marks_index_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_CHROMA_DIR", str(tmp_path / "chroma_warm"))
    config.CHROMA_DIR = tmp_path / "chroma_warm"
    config.EMBEDDING_BACKEND = "hashed"
    apply_tenant("minxin")
    config.AUDIENCE = "public"
    try:
        stats = warmup_index()
        assert stats.get("chunks", 0) > 0
        assert stats.get("warmed") is True
    finally:
        _restore_bank()
