from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path
from time import perf_counter

import streamlit as st
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag import config
from rag.eval import run_eval
from rag.generate import (
    REFUSAL_EMPTY,
    preview_prompt,
    build_generator,
    postprocess_answer,
    cited_clips,
    prefer_effective_docs,
)
from rag.ingest import (
    build_index,
    ensure_index,
    extracted_chars,
    list_uploads,
    load_documents,
    preview_vectors,
    save_uploaded_file,
    split_documents,
    uploads_missing_from_index,
    vector_inventory,
)
from rag.loaders import load_path
from rag.pipeline import keep_live_demand_rate
from rag.rerank import rerank_hits
from rag.retrieve import explain_hits, hybrid_table, retrieve_ranked
from rag.textutil import tokenize

STEPS = [
    ("开场", "RAG 在解决什么"),
    ("知识库", "Load：读进哪些文档"),
    ("切块", "Split：为什么要切开"),
    ("索引", "Embed + Store：写进仓库"),
    ("提问", "把问题变成检索词"),
    ("检索", "Retrieve：先多召回"),
    ("重排", "默认 bge-reranker"),
    ("生成", "结果卡 + 拒答"),
]

SAMPLE_QUESTIONS = [
    "活期利率是多少？",
    "信用卡还款日是哪天？",
    "随心贷能用来炒股吗？",
    "提前还房贷要不要违约金？",
    "PDF 细则里满 36 个月提前还款还收违约金吗？",
    "明天股价会涨吗？",
]


def main() -> None:
    st.set_page_config(page_title="星河银行 RAG 教室", page_icon="🏦", layout="wide")
    _inject_css()
    _load_secrets()
    _init_state()
    _persist_query()
    _ensure_index_ready()

    n_chunks = (st.session_state.index_stats or {}).get("chunks")
    llm_name = "Agnes" if config.AGNES_API_KEY else ("OpenAI" if config.OPENAI_API_KEY else ("Groq" if config.GROQ_API_KEY else "未接入（抽取原文）"))
    st.title("星河银行 RAG 教室")
    st.caption("建议按「下一步」慢慢走。检索仓库会在第一次打开时自动建好，你不用先添加资料。")
    if n_chunks:
        st.success(
            f"检索仓库：{n_chunks} 个 chunk。生成器：{llm_name}。"
            "索引只是把手册存成可搜索片段；大模型只在最后一步根据检索资料写答案。"
        )
    _pipeline_nav()
    st.progress((st.session_state.step + 1) / len(STEPS), text=f"第 {st.session_state.step + 1} / {len(STEPS)} 步 · {STEPS[st.session_state.step][1]}")
    if st.session_state.step != 4:
        _show_question_banner()

    step = st.session_state.step
    if step == 0:
        _step_intro()
    elif step == 1:
        _step_load()
    elif step == 2:
        _step_split()
    elif step == 3:
        _step_index()
    elif step == 4:
        _step_question()
    elif step == 5:
        _step_retrieve()
    elif step == 6:
        _step_rerank()
    else:
        _step_generate()

    _pager()


def _init_state() -> None:
    defaults = {
        "step": 0,
        "user_query": "活期利率是多少？",
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "top_k": config.TOP_K,
        "fetch_k": config.FETCH_K,
        "retriever_mode": config.RETRIEVER,
        "reranker_mode": config.RERANKER,
        "index_stats": None,
        "extra_note": "",
        "index_ready": False,
        "uploads_pending_index": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _load_secrets() -> None:
    """Streamlit Cloud / secrets.toml 里的 Key 写进环境，生成步才能调用 Agnes。"""

    def read(name: str) -> str:
        try:
            val = st.secrets[name]
        except Exception:
            return ""
        return str(val).strip()

    for src, dest in (
        ("AGNES_API_KEY", "AGNES_API_KEY"),
        ("AGNES_KEY", "AGNES_API_KEY"),
        ("AGNES_MODEL", "AGNES_MODEL"),
        ("AGNES_BASE_URL", "AGNES_BASE_URL"),
        ("OPENAI_API_KEY", "OPENAI_API_KEY"),
        ("GROQ_API_KEY", "GROQ_API_KEY"),
    ):
        val = read(src)
        if val:
            os.environ[dest] = val
    config.reload()


def _persist_query() -> None:
    """提问框只在提问步渲染。离开那一页前，先把已输入的字存进 user_query。"""
    if "query_box" in st.session_state:
        st.session_state.user_query = st.session_state.query_box


def _question() -> str:
    return (st.session_state.get("user_query") or "").strip()


def _show_question_banner() -> None:
    q = _question() or "（还没有问题）"
    left, right = st.columns((4, 1))
    with left:
        st.markdown(f"**当前问题：** {q}")
    with right:
        if st.button("自己改问题", use_container_width=True):
            st.session_state.step = 4
            st.rerun()


def _forget_hits() -> None:
    st.session_state["ranked"] = None
    st.session_state["reranked"] = None
    st.session_state["ranked_query"] = None
    st.session_state["reranked_query"] = None


def _session_extra_docs() -> list[Document]:
    note = (st.session_state.get("extra_note") or "").strip()
    if not note:
        return []
    return [
        Document(
            page_content=f"## 学员补充\n{note}",
            metadata={"source": "user-note.md", "file_type": "md", "page": 1},
        )
    ]


def _rebuild_index() -> None:
    with st.spinner("正在把知识库（含新上传文件）写入检索仓库…"):
        st.session_state.index_stats = build_index(
            reset=True,
            extra_docs=_session_extra_docs() or None,
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
        )
    st.session_state.uploads_pending_index = False
    st.session_state.index_ready = True


def _ensure_uploads_indexed() -> None:
    missing = uploads_missing_from_index()
    if missing or st.session_state.get("uploads_pending_index"):
        _rebuild_index()


def _ensure_index_ready() -> None:
    if st.session_state.get("index_ready"):
        return
    with st.spinner("正在把星河银行手册写入检索仓库（只需几秒）。这是自动的，不用你添加内容。"):
        st.session_state.index_stats = ensure_index(
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
        )
    st.session_state.index_ready = True


def _pipeline_nav() -> None:
    cols = st.columns(len(STEPS))
    for i, (short, _) in enumerate(STEPS):
        with cols[i]:
            label = f"{i + 1}. {short}"
            if st.button(label, use_container_width=True, type="primary" if i == st.session_state.step else "secondary"):
                st.session_state.step = i
                st.rerun()


def _teach(do_what: str, why: str, click_what: str) -> None:
    a, b, c = st.columns(3)
    with a:
        st.markdown(f'<div class="card teach"><h4>这一步在做什么</h4><p>{do_what}</p></div>', unsafe_allow_html=True)
    with b:
        st.markdown(f'<div class="card why"><h4>为什么需要它</h4><p>{why}</p></div>', unsafe_allow_html=True)
    with c:
        st.markdown(f'<div class="card act"><h4>你可以点什么</h4><p>{click_what}</p></div>', unsafe_allow_html=True)


def _step_intro() -> None:
    _teach(
        "大模型把世界记在参数里，但星河银行的利率、还款规则并不在里面。它不知道，就可能编。",
        "RAG 不是另一种聊天模型，而是先「查手册再回答」。查得到的段落，后面才能写进答案；查不到，就应该说资料没有。",
        "不要一次跳到检索。用底部「下一步」从知识库走到生成。顶部步骤条只是地图。",
    )
    st.markdown(
        """
#### 先记住四件事（后面每一步都在落实）

1. **知识库**是 Markdown、PDF、Word 等文件，模型事先没读过。  
2. **索引**把文件切段、向量化，写入 Chroma。加文件在知识库，写向量在索引。打开页面会自动把内置手册入库。  
3. **检索**先多捞一些候选（召回）。**重排**再按「和问题对得上吗」精排，只留几段给生成器。  
4. **生成**靠 Prompt 约束模型；**后处理**是模型写完之后的规则（补引用、空检索拒答），不是再改一遍 Prompt。
"""
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("知识库", "Markdown + PDF + Word")
    c2.metric("向量库", "Chroma", "嵌入式向量数据库")
    c3.metric("默认向量", "bge-small-zh", "哈希仅教学开关")
    st.markdown(
        """
#### 这是「能跑通的完整流水线」，不是生产系统缩小版聊天框

| | 本教室 | 生产常见做法 |
| --- | --- | --- |
| 文档 | md / 文本 PDF / docx | 再加扫描件 OCR、权限、版本 |
| 向量化 | 默认 **bge-small-zh**；哈希是教学开关 | 句向量 / 商业 Embedding |
| 存储 | **Chroma** 本地目录（就是向量库） | pgvector、Milvus、Pinecone 等 |
| 重排 | 默认 **bge-reranker**（top-16→top-4）；词重叠是对照开关 | 更大 Cross-Encoder |
| 生成 | Agnes 等 LLM + 检索上下文 | 同样，但有评测、缓存、审计 |
| 后处理 | 补引用、空检索拒答、冲突稿后置 | 忠实度检查、敏感信息过滤 |

教室仍对照「教学 vs 生产」。知识库故意放了营销页和过期 PDF（活期 1.50%、提前还款免违约金），用来演示以哪份为准。
"""
    )
    _eval_panel()
    st.markdown("#### 整条流水线")
    st.code(
        "文档 → 切块 → 写入仓库(索引) → 问题分词 → 多召回 fetch-k → 重排留 top-k → Prompt 生成 → 后处理",
        language="text",
    )
    if st.button("从知识库开始参观", type="primary"):
        st.session_state.step = 1
        st.rerun()


def _step_load() -> None:
    _teach(
        "知识库是磁盘上的文件：`data/kb` 里的 Markdown、PDF、Word，以及你现在上传的材料。Load 只负责读进来，变成文档对象。这里还没有搜索，也还没有向量。",
        "真实文件应该加在这一步，而不是索引步。索引只是把「已经在知识库里的东西」切块、向量化、写入 Chroma。放错位置会让人以为向量库是另一个文件夹。",
        "先点左侧一篇制度确认能看原文。再拖入自己的 PDF / Word / 图片：选中文件就会保存并索引，不必再点一次。扫描件会尝试 OCR。",
    )
    _upload_panel()
    docs = load_documents()
    if not docs:
        st.warning("知识库是空的。上传一份文件，或确认 `data/kb` 里有材料。")
        return
    labels = []
    for doc in docs:
        src = Path(str(doc.metadata.get("source", ""))).name
        ft = doc.metadata.get("file_type", "?")
        page = doc.metadata.get("page", "")
        labels.append(f"{src}  [{ft} · 第{page}页]")
    left, right = st.columns((1, 2))
    with left:
        picked = st.radio("点一篇打开", labels, index=0)
        st.session_state.picked_source = picked
        types = sorted({str(d.metadata.get("file_type")) for d in docs})
        st.caption(f"共 {len(docs)} 个解析单元，格式：{', '.join(types)}")
    with right:
        doc = docs[labels.index(picked)]
        st.markdown(f"**{picked}** · {len(doc.page_content)} 字")
        st.text_area("原文", doc.page_content, height=360, label_visibility="collapsed")


def _ingest_selected_uploads() -> None:
    files = st.session_state.get("kb_file_uploader") or []
    if not files:
        return
    done = st.session_state.setdefault("ingested_upload_sigs", [])
    saved = []
    blank = []
    for item in files:
        sig = f"{item.name}:{getattr(item, 'size', 0)}"
        if sig in done:
            continue
        path = save_uploaded_file(item.name, item.getvalue())
        saved.append(path)
        done.append(sig)
        if extracted_chars(path) == 0:
            blank.append(path.name)
    if not saved:
        return
    st.session_state.uploads_pending_index = True
    try:
        _rebuild_index()
        st.session_state.upload_ok = "已识别并写入检索：" + "、".join(p.name for p in saved)
        st.session_state.upload_error = ""
    except Exception as exc:
        st.session_state.upload_ok = ""
        st.session_state.upload_error = f"文件已保存，但写入检索仓库失败：{exc}"
    st.session_state.upload_blank = blank


def _upload_panel() -> None:
    st.markdown("#### 把真实文件放进知识库")
    st.caption("支持 PDF、Word（.docx）、Markdown、txt、手机拍照/截图。选中文件就会入库；扫描件会走 OCR，第一次可能要下载模型。")
    st.file_uploader(
        "拖入或选择文件（可多选）",
        type=["pdf", "docx", "md", "txt", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        key="kb_file_uploader",
        on_change=_ingest_selected_uploads,
    )
    if st.session_state.get("upload_ok"):
        st.success(st.session_state.upload_ok)
    if st.session_state.get("upload_error"):
        st.error(st.session_state.upload_error)
    if st.session_state.get("upload_blank"):
        st.error(
            "这些文件仍然抽不出字："
            + "、".join(st.session_state.upload_blank)
            + "。请换可复制文字的 PDF/Word，或更清晰的照片。"
        )
    existing = list_uploads()
    if existing:
        st.markdown("**已上传**")
        for path in existing:
            docs = load_path(path)
            text = "\n".join(d.page_content for d in docs).strip()
            left, right = st.columns((4, 1))
            left.write(f"{path.name} · {path.stat().st_size} 字节 · 抽出 {len(text)} 字")
            if right.button("删除", key=f"del_upload_{path.name}"):
                path.unlink()
                st.session_state.uploads_pending_index = True
                _rebuild_index()
                st.rerun()
            if text:
                with st.expander(f"识别出的原文 · {path.name}", expanded=len(text) < 400):
                    st.write(text[:2000] + ("…" if len(text) > 2000 else ""))
            else:
                st.caption(f"{path.name} 没有识别出文字，检索时等于不存在这份文件。")
    missing = uploads_missing_from_index()
    if missing:
        st.warning("这些上传文件还没进检索仓库：" + "、".join(missing) + "。正在补写。")
        _rebuild_index()
        st.rerun()
    st.session_state.extra_note = st.text_area(
        "可选：写一条只属于你的规定（仍是知识，会在重建索引时一并入库）",
        value=st.session_state.extra_note,
        placeholder="例如：星河银行大厅取号后超时 15 分钟需重新取号。",
        height=90,
    )


def _step_split() -> None:
    docs = load_documents()
    _teach(
        "整本手册太长，无法整本拿去比相似度。按 `##` 标题切开，过长的再按字数切，并留 overlap，避免一句话被切断。",
        "切块决定「一次能命中多大范围」。切太大，随心贷和房贷挤在一起；切太碎，一条规则裂成半句。",
        "拖动切块大小和 overlap，下面立刻用当前问题做一次内存 BM25，看命中段和重叠文字怎么变。不必先重建索引。",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.chunk_size = st.slider("chunk 大小", 80, 800, st.session_state.chunk_size, 20)
    with c2:
        st.session_state.chunk_overlap = st.slider("重叠 overlap", 0, 200, st.session_state.chunk_overlap, 10)
    chunks = split_documents(
        docs,
        chunk_size=st.session_state.chunk_size,
        chunk_overlap=st.session_state.chunk_overlap,
    )
    st.metric("当前切出的 chunk 数", len(chunks), f"来自 {len(docs)} 篇文档")
    question = _question()
    st.markdown(f"**当前问题：** `{question}` · 用这组切块立刻看命中（还没写入 Chroma）")
    hits = explain_hits(question, chunks, k=5)
    if hits:
        st.dataframe(
            [
                {
                    "chunk": row["chunk"],
                    "文件": row["source"],
                    "分数": row["score"],
                    "重叠字": row["overlap_chars"],
                    "为什么命中": row["why"],
                    "重叠预览": row["overlap_preview"] or "—",
                    "开头": row["preview"],
                }
                for row in hits
            ],
            hide_index=True,
            use_container_width=True,
        )
    for i, chunk in enumerate(chunks):
        src = Path(str(chunk.metadata.get("source", ""))).name
        title = chunk.page_content.strip().splitlines()[0][:40]
        overlap_n = chunk.metadata.get("overlap_chars") or 0
        with st.expander(f"chunk {i + 1} · {src} · 重叠 {overlap_n} 字 · {title}", expanded=(i == 0)):
            ov = chunk.metadata.get("overlap_preview") or ""
            if ov:
                st.caption("与上一段重合的开头：" + ov)
            st.write(chunk.page_content)
            st.caption(f"{len(chunk.page_content)} 字")


def _step_index() -> None:
    _teach(
        "索引 = 把知识库里已经有的文本变成向量，写入 **Chroma**（目录 `chroma_db/`）。不是再上传一份手册。打开教室时已自动把内置手册入库。",
        "专门数据库是为了按向量近邻检索。Chroma 是嵌入式向量库。没有这一步，检索就是空仓库或还是旧文件。",
        "真实文件在「知识库」保存后会自动重建。这里仍可换向量后端并手动重建。点下面预览确认新文件的 chunk 在不在。",
    )
    st.info("文件属于知识库；这一步只负责 Embed + Store。上传请点顶部「2. 知识库」。")
    if st.session_state.get("uploads_pending_index"):
        st.warning("知识库有新文件或删除尚未写入向量库，请点下面的「重建索引」。")
    use_hash = st.checkbox(
        "教学开关：用哈希向量（不下载模型，立刻能看步骤；默认是 bge-small-zh）",
        value=config.EMBEDDING_BACKEND in {"hashed", "hash", "ngram"},
    )
    if use_hash:
        config.EMBEDDING_BACKEND = "hashed"
        config.COLLECTION_NAME = f"bank_kb_{config.EMBEDDING_BACKEND}"
    else:
        config.EMBEDDING_BACKEND = "huggingface"
        config.COLLECTION_NAME = f"bank_kb_{config.EMBEDDING_BACKEND}"
    last_backend = (st.session_state.index_stats or {}).get("embedding_backend")
    if last_backend and last_backend != config.EMBEDDING_BACKEND:
        st.warning(
            f"预览正在看集合 `{config.COLLECTION_NAME}`，上次重建用的是 `{last_backend}`。"
            "换句向量/哈希后必须点重建，否则上传文件还在另一套库里。"
        )
    _ensure_uploads_indexed()
    if st.button("按当前切块重建索引", type="primary"):
        try:
            _rebuild_index()
            st.success("已重建。现在仓库里是当前切块与向量后端。")
        except ImportError as exc:
            st.error(str(exc))
    stats = st.session_state.index_stats
    if stats:
        a, b, c = st.columns(3)
        a.metric("文档", stats["documents"])
        b.metric("chunk", stats["chunks"])
        c.metric("向量库", stats.get("vector_store", "chroma"))
        if stats.get("embed_ms") is not None:
            d1, d2 = st.columns(2)
            d1.metric("切块耗时", f"{stats.get('split_ms', 0)} ms")
            d2.metric("嵌入写入耗时", f"{stats.get('embed_ms', 0)} ms")
        st.caption(
            "教学环境曾默认哈希：不下载模型、秒级看完「文本→向量」。作品集默认句向量更接近生产；慢是因为每段都过神经网络。"
        )
        types = stats.get("file_types") or []
        if types:
            st.caption("已解析格式：" + "、".join(types))
        st.caption(f"落盘目录：`{stats['persist_directory']}`")

    st.markdown("#### 预览向量库")
    st.caption("先看「按文件汇总」。下面的明细默认只翻页显示，所以上传文件如果排在后面，看起来会像没入库。")
    try:
        inventory = vector_inventory()
    except Exception as exc:
        st.warning(f"还读不出向量：{exc}")
        inventory = None
        preview = None
    if inventory:
        st.write(
            f"集合 `{inventory['collection']}` · 后端 `{inventory['backend']}` · "
            f"{inventory['total_files']} 个文件 · {inventory['total_chunks']} 条 chunk"
        )
        if inventory["missing_uploads"]:
            st.error("这些已上传文件还不在当前向量库里：" + "、".join(inventory["missing_uploads"]))
        summary = [
            {
                "文件": row["source"],
                "类型": row["file_type"] or "—",
                "chunk 数": row["chunks"],
                "总字数": row["chars"],
                "来源": "本次上传" if row["uploaded"] else "内置手册",
            }
            for row in inventory["files"]
        ]
        st.dataframe(summary, hide_index=True, use_container_width=True)
        names = ["（全部文件）"] + [row["source"] for row in inventory["files"]]
        picked = st.selectbox("查看某个文件的向量明细", names)
        if st.session_state.get("preview_source") != picked:
            st.session_state.preview_offset = 0
            st.session_state.preview_source = picked
        source = None if picked == "（全部文件）" else picked
        if "preview_offset" not in st.session_state:
            st.session_state.preview_offset = 0
        cols = st.columns((1, 1, 2))
        with cols[0]:
            if st.button("上一页") and st.session_state.preview_offset >= 20:
                st.session_state.preview_offset -= 20
                st.rerun()
        with cols[1]:
            if st.button("下一页"):
                st.session_state.preview_offset += 20
                st.rerun()
        try:
            preview = preview_vectors(limit=20, offset=st.session_state.preview_offset, source=source)
        except Exception as exc:
            st.warning(f"明细读不出：{exc}")
            preview = None
        if preview and preview["rows"]:
            st.caption(
                f"明细 {st.session_state.preview_offset + 1}–"
                f"{st.session_state.preview_offset + len(preview['rows'])} / {preview['filtered']} 条"
                "（向量只展示前 12 维）"
            )
            table = [
                {
                    "来源": row["source"],
                    "类型": row["file_type"],
                    "页": row["page"],
                    "字数": row["chars"],
                    "维度": row["dim"],
                    "模长": row["norm"],
                    "前12维": str(row["vector_head"]),
                    "原文开头": row["preview"],
                }
                for row in preview["rows"]
            ]
            st.dataframe(table, hide_index=True, use_container_width=True)
            for i, row in enumerate(preview["rows"][:8]):
                with st.expander(f"完整原文 · {row['source']} · dim={row['dim']}", expanded=(i == 0)):
                    st.write(row["text"])
                    st.code(str(row["vector_head"]) + (" …" if row["dim"] > 12 else ""))
        elif preview is not None:
            st.info("没有符合筛选的明细。")
    elif inventory is not None:
        st.info("向量库暂时是空的，请先重建索引。")


def _step_question() -> None:
    _teach(
        "原句不会整句丢进仓库。先分词，再丢掉「是、吗、多少」等停用词。`data/terms.txt` 里的产品名（随心贷、房贷、违约金）会当成一个词。",
        "分词错了，检索就会空或跑偏。比如只切出「房」和「贷」，就对不上手册里的「住房按揭」。",
        "在下面的输入框自己打字，或点现成问句。现成问句只是省事，不是只能点按钮。",
    )
    if "query_box" not in st.session_state:
        st.session_state.query_box = st.session_state.user_query
    st.text_area(
        "自己打一个问题",
        key="query_box",
        height=100,
        placeholder="例如：提前还房贷要不要付违约金？不必只用下面的现成问句。",
    )
    st.session_state.user_query = st.session_state.query_box
    if st.button("用这个问题去检索", type="primary"):
        _forget_hits()
        st.session_state.step = 5
        st.rerun()
    st.caption("点这些会填进上面的输入框，并跳到检索：")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for i, sample in enumerate(SAMPLE_QUESTIONS):
        with cols[i]:
            st.button(
                sample,
                use_container_width=True,
                on_click=_use_sample_question,
                args=(sample,),
                key=f"sample_q_{i}",
            )
    terms = tokenize(_question())
    st.markdown("**分词后用来检索的词**")
    if terms:
        chips = " ".join(f'<span class="chip">{html.escape(t)}</span>' for t in terms)
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.info("分词结果为空，检索会什么都找不到。换几个实词再试。")


def _use_sample_question(sample: str) -> None:
    # 回调在下一轮渲染、创建 text_input 之前执行，避免改已实例化的 widget key。
    st.session_state.user_query = sample
    st.session_state.query_box = sample
    st.session_state.step = 5
    _forget_hits()


def _step_retrieve() -> None:
    _teach(
        "这一步是召回：BM25、向量或 Hybrid。Hybrid 用 RRF 把两路名次合成一张表，再交给重排。",
        "检索器擅长「别漏」。相关段落如果只排在第 8 名，生成器永远看不见。",
        "选 Hybrid 看 BM25 / 向量 / 融合三列名次。哈希向量时变化可能很小，换成句向量后更明显。",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.fetch_k = st.slider("多召回几条 fetch-k", 4, 24, int(st.session_state.fetch_k))
    with c2:
        st.session_state.top_k = st.slider("重排后要留几条 top-k", 1, 8, st.session_state.top_k)
    with c3:
        modes = ["bm25", "vector", "hybrid"]
        current = st.session_state.retriever_mode if st.session_state.retriever_mode in modes else "bm25"
        st.session_state.retriever_mode = st.radio(
            "检索器",
            modes,
            index=modes.index(current),
            horizontal=True,
            format_func=lambda x: {"bm25": "BM25", "vector": "向量", "hybrid": "Hybrid"}[x],
        )
    if st.session_state.fetch_k < st.session_state.top_k:
        st.session_state.fetch_k = st.session_state.top_k
    _ensure_uploads_indexed()
    question = _question()
    st.markdown(f"当前问题：`{question}`")
    try:
        ranked = retrieve_ranked(
            question,
            k=st.session_state.top_k,
            retriever=st.session_state.retriever_mode,
            fetch_k=st.session_state.fetch_k,
        )
    except Exception as exc:
        st.error(f"检索失败：{exc}")
        st.session_state.index_ready = False
        _ensure_index_ready()
        return
    terms = tokenize(question)
    st.caption("本问用来打分的词：" + ("、".join(terms) if terms else "（空）"))
    if not ranked:
        from rag.ingest import count_indexed

        n = count_indexed()
        if n == 0:
            st.error("没有结果，因为检索仓库是空的（还没有索引），不是这句话没有答案。页面顶部会自动建库，请刷新后再问一次。")
            st.session_state.index_ready = False
            _ensure_index_ready()
        elif st.session_state.retriever_mode == "vector":
            st.warning(
                "没有可用的向量近邻。当前是教学用哈希向量，对「提前还房贷」这类句子很弱。"
                "请改回 BM25。手册里对应的说法是「提前还款」和「违约金」，在住房按揭那一节。"
            )
        else:
            from rag.ingest import indexed_source_names

            sources = "、".join(sorted(indexed_source_names())[:12]) or "（无）"
            uploads = list_uploads()
            blank = [p.name for p in uploads if extracted_chars(p) == 0]
            if blank:
                st.error(
                    "这些上传文件没有抽出文字，所以问医保/门诊也检索不到。请换成可选中复制的 PDF/Word，或先做 OCR："
                    + "、".join(blank)
                )
            else:
                st.warning(
                    "检索仓库里现在有这些文件："
                    + sources
                    + f"。问句分词是：{'、'.join(terms) or '无'}，和这些 chunk 对不上。"
                    "若刚上传，请到知识库看「抽出 N 字」是不是 0。"
                    "词和原文不一致（例如问「覆盖」、文件写「报销」）也会空。"
                )
        st.session_state["ranked"] = []
        st.session_state["reranked"] = []
        st.session_state["ranked_query"] = question
        return
    rows = []
    for i, item in enumerate(ranked, start=1):
        src = Path(str(item["doc"].metadata.get("source", ""))).name
        rows.append(
            {
                "召回名次": i,
                "分数": item["score"],
                "含义": (
                    "向量距离(越小越近)"
                    if item.get("score_kind") == "vector_distance"
                    else ("RRF 融合(越大越好)" if item.get("score_kind") == "rrf" else "BM25(越大越好)")
                ),
                "命中词": "、".join(item["matched"]) or "—",
                "来源": src,
                "开头": item["doc"].page_content.strip().splitlines()[0][:32],
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)
    if st.session_state.retriever_mode == "hybrid":
        st.markdown("#### Hybrid 融合前后名次")
        st.caption("同一段在 BM25 / 向量 / RRF 三列里的名次。空着表示这一路没进前 fetch-k。")
        try:
            table = hybrid_table(question, fetch_k=st.session_state.fetch_k)
            st.dataframe(table["rows"][:16], hide_index=True, use_container_width=True)
        except Exception as exc:
            st.caption(f"融合表暂时画不出：{exc}")
    st.session_state["ranked"] = ranked
    st.session_state["ranked_query"] = question
    st.caption("上面整张表都会进入下一步重排；生成器仍然只能看见重排后留下的几条。")
    for i, item in enumerate(ranked[:8], start=1):
        src = Path(str(item["doc"].metadata.get("source", ""))).name
        with st.expander(f"召回第 {i} 名 · 分数 {item['score']} · {src}", expanded=(i == 1)):
            st.markdown(_highlight_html(item["doc"].page_content, item["matched"]), unsafe_allow_html=True)
            st.caption("高亮 = 和问题分词重叠的词。没有高亮却被召回，多半是向量近邻，语义像但用词不同。")


def _ensure_ranked(question: str) -> list:
    ranked = st.session_state.get("ranked")
    if ranked is not None and st.session_state.get("ranked_query") == question:
        return ranked
    ranked = retrieve_ranked(
        question,
        k=st.session_state.top_k,
        retriever=st.session_state.retriever_mode,
        fetch_k=st.session_state.fetch_k,
    )
    st.session_state["ranked"] = ranked
    st.session_state["ranked_query"] = question
    return ranked


def _step_rerank() -> None:
    _teach(
        "先多召回再精排。默认用 bge-reranker 对 (问题, 段落) 打分，只留 top-k 给生成器。",
        "词重叠精排和 BM25 几乎看同一类信号，经常名次不变——所以只留作对照开关，不当默认。",
        "左右表对照召回 vs 精排。把开关拨到「词重叠」再问同一题，看名次会不会几乎不动。",
    )
    st.session_state.reranker_mode = st.radio(
        "精排后端",
        ["bge", "lexical"],
        index=0 if st.session_state.reranker_mode != "lexical" else 1,
        horizontal=True,
        format_func=lambda x: "bge-reranker（默认）" if x == "bge" else "词重叠（教学对照）",
    )
    config.RERANKER = st.session_state.reranker_mode
    if st.session_state.reranker_mode == "lexical":
        st.warning("对照模式：归一化召回分 + 词/标题重叠。这不是上线用的精排。")
    else:
        st.caption("默认：Cross-Encoder `BAAI/bge-reranker-base`，召回约 16 条再留 4 条。第一次会下载模型。")
    question = _question()
    st.markdown(f"正在精排的问题：`{question}`")
    try:
        ranked = _ensure_ranked(question)
    except Exception as exc:
        st.error(f"请先完成索引和检索：{exc}")
        return
    if not ranked:
        st.warning("召回为空，没有可重排的段落。回到检索步换一个问题。")
        st.session_state["reranked"] = []
        return
    kept = rerank_hits(
        question,
        ranked,
        keep=st.session_state.top_k,
        mode=st.session_state.reranker_mode,
    )
    kept = keep_live_demand_rate(question, ranked, kept)
    from rag.rerank import LAST_ERROR as RERANK_ERROR

    if RERANK_ERROR:
        st.warning(RERANK_ERROR)
    st.session_state["reranked"] = kept
    st.session_state["reranked_query"] = question
    left, right = st.columns(2)
    with left:
        st.markdown("#### 召回顺序（未精排）")
        recall_rows = []
        for i, item in enumerate(ranked, start=1):
            src = Path(str(item["doc"].metadata.get("source", ""))).name
            recall_rows.append(
                {
                    "召回": i,
                    "检索分": item.get("score"),
                    "来源": src,
                    "开头": item["doc"].page_content.strip().splitlines()[0][:28],
                }
            )
        st.dataframe(recall_rows, hide_index=True, use_container_width=True)
    with right:
        st.markdown("#### 重排后留下的（给生成器用）")
        keep_rows = []
        for item in kept:
            src = Path(str(item["doc"].metadata.get("source", ""))).name
            keep_rows.append(
                {
                    "精排": item.get("rerank_rank"),
                    "原召回": item.get("recall_rank"),
                    "重排分": item.get("rerank_score"),
                    "来源": src,
                    "开头": item["doc"].page_content.strip().splitlines()[0][:28],
                }
            )
        st.dataframe(keep_rows, hide_index=True, use_container_width=True)
    moved = [item for item in kept if item.get("recall_rank") != item.get("rerank_rank")]
    if moved:
        st.info("有段落的名次变了：检索负责广撒网，重排负责把更贴问题的片段抬到前面。")
    else:
        st.caption("这一问上，精排没有打乱名次。小语料 + BM25 时很常见，不代表生产里也不需要重排。")
    for item in kept[:3]:
        src = Path(str(item["doc"].metadata.get("source", ""))).name
        with st.expander(
            f"精排第 {item.get('rerank_rank')} 名（原召回第 {item.get('recall_rank')}）· {src}",
            expanded=item.get("rerank_rank") == 1,
        ):
            st.markdown(_highlight_html(item["doc"].page_content, item.get("matched") or []), unsafe_allow_html=True)


def _step_generate() -> None:
    _teach(
        "生成器只能看见精排留下的几段。本页必须给出完整结果卡：答案、引用卡片、空检索拒答，不会停在「正在回答」。",
        "没被点名的段落不该出现在结果里。PDF 按页摘一句。手册没有的问题要明确说没有，不能编利率。",
        "先看结果卡。Prompt 折在下面。需要作品集截图时，用开场页的「跑评测」。",
    )
    ranked = st.session_state.get("reranked")
    question = _question()
    if not ranked or st.session_state.get("reranked_query") != question:
        try:
            recalled = _ensure_ranked(question)
            ranked = rerank_hits(
                question,
                recalled,
                keep=st.session_state.top_k,
                mode=st.session_state.reranker_mode,
            )
            ranked = keep_live_demand_rate(question, recalled, ranked)
            st.session_state["reranked"] = ranked
            st.session_state["reranked_query"] = question
        except Exception as exc:
            st.error(f"请先完成索引和检索：{exc}")
            _render_result_card(
                question,
                REFUSAL_EMPTY,
                [],
                ["retrieve_error"],
                "none",
                [],
            )
            return
    docs = prefer_effective_docs([item["doc"] for item in ranked])
    if not docs:
        _render_result_card(question, REFUSAL_EMPTY, [], ["empty_retrieve"], "none", [])
        return
    gen, name = build_generator()
    t_gen0 = perf_counter()
    try:
        with st.spinner("正在根据检索资料生成（完成后会留下结果卡，不会停在转圈）…"):
            raw = gen.generate(question, docs)
    except Exception as exc:
        st.error(f"大模型调用失败。已回退抽取式结果卡。详情：{exc}")
        from rag.generate import ExtractiveGenerator

        raw = ExtractiveGenerator().generate(question, docs)
        name = f"{name}（调用失败，回退抽取）"
    t_gen1 = perf_counter()
    final, notes = postprocess_answer(raw, question, docs)
    t_gen2 = perf_counter()
    clips = cited_clips(final, docs, question)
    stats = st.session_state.get("index_stats") or {}
    timings = {
        "embed": stats.get("embed_ms"),
        "generate": round((t_gen1 - t_gen0) * 1000, 1),
        "postprocess": round((t_gen2 - t_gen1) * 1000, 1),
    }
    _render_result_card(question, final, clips, notes, name, docs, raw=raw, timings=timings)
    unused = len(docs) - len(clips)
    if unused > 0 and clips:
        st.caption(f"检索还拿到 {unused} 段未在答案中引用，已隐藏。")
    if name.startswith("agnes:"):
        st.success(f"本步已调用 Agnes 大模型（`{name}`）。检索仍在本地，模型只根据上面的资料作答。")
    elif name == "extractive":
        st.warning(
            "现在没有调用大模型，所以答案只是检索到的原文摘录。"
            "在 Streamlit Cloud：App → Settings → Secrets 添加 `AGNES_API_KEY`，可选 `AGNES_MODEL = \"agnes-2.5-flash\"`，然后 Reboot。"
        )


def _render_result_card(
    question: str,
    final: str,
    clips: list,
    notes: list,
    name: str,
    docs: list,
    raw: str | None = None,
    timings: dict | None = None,
) -> None:
    st.markdown("#### 结果卡")
    st.markdown(f"问题：`{question}`")
    st.markdown("##### 答案")
    st.markdown(f'<div class="answer-box">{html.escape(final).replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
    st.caption(f"生成器：`{name}`" + (f" · 后处理：{'、'.join(notes)}" if notes else ""))
    if timings:
        cols = st.columns(3)
        cols[0].metric("嵌入(上次建库)", f"{timings.get('embed') or '—'} ms")
        cols[1].metric("生成", f"{timings.get('generate')} ms")
        cols[2].metric("后处理", f"{timings.get('postprocess')} ms")
        st.caption("教学默认曾用哈希，是因为嵌入只要几毫秒、不下载模型；作品集改句向量后，这一栏才能解释「慢在嵌入/重排」。")
    st.markdown("##### 引用卡片")
    if clips:
        st.caption("资料N · 文件名 · 摘句。只显示答案点名的片段。")
        for clip in clips:
            st.markdown(_clip_html(clip), unsafe_allow_html=True)
    else:
        st.info("空检索或明确拒答：不展示原文，避免把无关段落当成依据。")
    with st.expander("发给模型的 Prompt（生成前）"):
        st.code(preview_prompt(question, docs), language="markdown")
    with st.expander("后处理做了什么"):
        st.markdown(
            """
- **空检索拒答**：没有资料就不编利率。
- **弱相关拒答**：命中段落和问题对不上，同样不编。
- **数字不在资料里**：问题或答案里的利率若手册/引用没有，直接拒答。
- **未引用则降级**：补不上 `(资料N)` 就拒答。
- **承诺类拦截**：保证收益 / 保本 / 稳赚不会输出。
- **去掉代码围栏 / 补引用**。
"""
        )
        st.caption("本轮实际执行：" + ("、".join(notes) if notes else "无"))
        if raw is not None:
            with st.expander("模型刚写完、还没后处理的原文"):
                st.write(raw)


def _eval_panel() -> None:
    st.markdown("#### 一键跑评测")
    st.caption("固定 20 题：能答 / 应拒答 / 活期与定期易混 / 跨文档 / 改写 / 过期利率冲突。截图即可贴作品集。")
    if st.button("跑评测", type="primary"):
        with st.spinner("正在按评测集逐题检索并打分…"):
            try:
                report = run_eval(
                    k=st.session_state.top_k,
                    retriever=st.session_state.retriever_mode,
                    reranker=st.session_state.reranker_mode,
                )
            except Exception as exc:
                st.error(f"评测失败：{exc}")
                return
        st.session_state.eval_report = report
    report = st.session_state.get("eval_report")
    if not report:
        return
    a, b, c, d = st.columns(4)
    a.metric("命中率", f"{report['hit_rate']*100:.0f}%", f"{report['n_answerable']} 道应答题")
    b.metric("拒答正确率", f"{report['refuse_accuracy']*100:.0f}%", f"{report['n_refuse']} 道应拒答")
    c.metric("引用点名率", f"{report['cite_named_rate']*100:.0f}%", "应答题是否写出资料N")
    d.metric("总及格率", f"{report['pass_rate']*100:.0f}%", f"{report['n']} 条含改写")
    table = [
        {
            "题号": row["id"],
            "问题": row["question"],
            "命中": "是" if row["hit"] else "否",
            "拒答正确": "是" if row["refuse_ok"] else "否",
            "点名引用": "是" if row["cited"] else "否",
            "及格": "是" if row["answer_ok"] else "否",
            "答案": (row["answer"] or "")[:80],
        }
        for row in report["rows"]
    ]
    st.dataframe(table, hide_index=True, use_container_width=True)


def _pager() -> None:
    st.divider()
    prev_col, _, next_col = st.columns((1, 3, 1))
    with prev_col:
        if st.session_state.step > 0 and st.button("← 上一步", use_container_width=True):
            st.session_state.step -= 1
            st.rerun()
    with next_col:
        if st.session_state.step < len(STEPS) - 1 and st.button("下一步 →", type="primary", use_container_width=True):
            st.session_state.step += 1
            st.rerun()


def _clip_html(clip: dict) -> str:
    kind = html.escape(str(clip.get("kind") or "md"))
    page = clip.get("page")
    bits = [html.escape(clip["name"])]
    if str(clip.get("kind")) == "pdf" and page:
        bits.append(f"第 {page} 页")
    elif page:
        bits.append(f"第 {page} 节")
    bits.append(f"资料{clip['index']}")
    quote = html.escape(clip.get("quote") or "").replace("\n", "<br>")
    return (
        f'<article class="clip clip-{kind}">'
        f'<div class="clip-badge">{html.escape(clip.get("kind_label") or "文件")}</div>'
        f'<div class="clip-main"><div class="clip-meta">{" · ".join(bits)}</div>'
        f"<blockquote>{quote}</blockquote></div></article>"
    )


def _highlight_html(text: str, terms: list[str]) -> str:
    escaped = html.escape(text)
    for term in sorted({t for t in terms if t}, key=len, reverse=True):
        pattern = re.escape(html.escape(term))
        escaped = re.sub(pattern, lambda match: f"<mark>{match.group(0)}</mark>", escaped)
    return f'<div class="passage">{escaped.replace(chr(10), "<br>")}</div>'


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .why { background: #fff7ed; border: 1px solid #fed7aa; }
        .card { border-radius: 16px; padding: 16px 18px; min-height: 150px; }
        .teach { background: #eef2ff; border: 1px solid #c7d2fe; }
        .act { background: #ecfdf5; border: 1px solid #a7f3d0; }
        .card h4 { margin: 0 0 8px 0; font-size: 0.95rem; }
        .card p { margin: 0; line-height: 1.55; color: #1f2937; }
        .chip { display: inline-block; background: #1d4ed8; color: white; border-radius: 999px;
                padding: 2px 10px; margin: 0 6px 6px 0; font-size: 0.85rem; }
        mark { background: #fde68a; padding: 0 2px; border-radius: 4px; }
        .passage { background: #fffbeb; border: 1px solid #fcd34d; border-radius: 12px; padding: 12px 14px; }
        .answer-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px;
                      padding: 18px 20px; font-size: 1.05rem; line-height: 1.7; color: #0f172a; }
        .clip { display: flex; gap: 0; margin: 12px 0 16px 0; border-radius: 12px;
                overflow: hidden; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08); }
        .clip-badge { writing-mode: vertical-rl; text-orientation: mixed; letter-spacing: 0.2em;
                      font-size: 0.75rem; font-weight: 700; padding: 12px 8px; color: white;
                      display: flex; align-items: center; justify-content: center; min-width: 36px; }
        .clip-md .clip-badge, .clip-txt .clip-badge { background: #334155; }
        .clip-pdf .clip-badge { background: #b91c1c; }
        .clip-docx .clip-badge, .clip-doc .clip-badge { background: #1d4ed8; }
        .clip-main { flex: 1; background: #fffef8; background-image:
                     repeating-linear-gradient(transparent, transparent 27px, #f1efe6 28px);
                     padding: 12px 16px 16px 16px; }
        .clip-pdf .clip-main { background: #fff7f7; }
        .clip-meta { font-size: 0.8rem; color: #64748b; margin-bottom: 8px; letter-spacing: 0.02em; }
        .clip blockquote { margin: 0; font-size: 0.95rem; line-height: 1.75; color: #1e293b; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
