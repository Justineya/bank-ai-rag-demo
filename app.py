from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag import config
from rag.generate import preview_prompt, build_generator, postprocess_answer
from rag.ingest import (
    build_index,
    ensure_index,
    list_uploads,
    load_documents,
    preview_vectors,
    save_uploaded_file,
    split_documents,
)
from rag.rerank import rerank_hits
from rag.retrieve import retrieve_ranked
from rag.textutil import tokenize

STEPS = [
    ("开场", "RAG 在解决什么"),
    ("知识库", "Load：读进哪些文档"),
    ("切块", "Split：为什么要切开"),
    ("索引", "Embed + Store：写进仓库"),
    ("提问", "把问题变成检索词"),
    ("检索", "Retrieve：先多召回"),
    ("重排", "Rerank：再精排留下几条"),
    ("生成", "Generate + 后处理"),
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
        "question": "活期利率是多少？",
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "top_k": config.TOP_K,
        "fetch_k": config.FETCH_K,
        "retriever_mode": config.RETRIEVER,
        "index_stats": None,
        "extra_note": "",
        "index_ready": False,
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
2. **索引**把文件切段、向量化，写入 Chroma。你不用再贴一遍产品——打开页面会自动入库。  
3. **检索**先多捞一些候选（召回）。**重排**再按「和问题对得上吗」精排，只留几段给生成器。  
4. **生成**靠 Prompt 约束模型；**后处理**是模型写完之后的规则（补引用、空检索拒答），不是再改一遍 Prompt。
"""
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("知识库", "Markdown + PDF + Word")
    c2.metric("向量库", "Chroma", "嵌入式向量数据库")
    c3.metric("默认向量", "哈希(快)", "可换成句向量模型")
    st.markdown(
        """
#### 这是「能跑通的完整流水线」，不是生产系统缩小版聊天框

| | 本教室 | 生产常见做法 |
| --- | --- | --- |
| 文档 | md / 文本 PDF / docx | 再加扫描件 OCR、权限、版本 |
| 向量化 | 默认字符哈希，秒级 | 句向量或商业 Embedding，分钟～小时 |
| 存储 | **Chroma** 本地目录（就是向量库） | pgvector、Milvus、Pinecone 等 |
| 重排 | 词重叠精排（教学替身） | Cross-Encoder / bge-reranker |
| 生成 | Agnes 等 LLM + 检索上下文 | 同样，但有评测、缓存、审计 |
| 后处理 | 补引用、空检索拒答 | 忠实度检查、敏感信息过滤 |

默认哈希快，是为了让你先看懂步骤。真句向量会慢，因为每段都要过神经网络。
"""
    )
    st.markdown("#### 整条流水线")
    st.code(
        "文档 → 切块 → 写入仓库(索引) → 问题分词 → 多召回 fetch-k → 重排留 top-k → Prompt 生成 → 后处理",
        language="text",
    )
    if st.button("从知识库开始参观", type="primary"):
        st.session_state.step = 1
        st.rerun()


def _step_load() -> None:
    docs = load_documents()
    _teach(
        "Load 读取 `data/kb` 里的 Markdown、PDF、Word。PDF 按页、Word 按标题切成文档对象。这里还没有搜索。",
        "生产环境还会碰到扫描件 OCR、加密 PDF、表格、页眉页脚噪音。本教室先覆盖可复制文本的 PDF/Word。",
        "左侧点文件名。请打开带 pdf / docx 标记的制度文件，确认能看到「违约金」「挂失电话」。",
    )
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


def _step_split() -> None:
    docs = load_documents()
    _teach(
        "整本手册太长，无法整本拿去比相似度。按 `##` 标题切开，过长的再按字数切，并留 overlap，避免一句话被切断。",
        "切块决定「一次能命中多大范围」。切太大，随心贷和房贷挤在一起；切太碎，一条规则裂成半句。",
        "拖动切块大小，看数量变化。点开含「提前还款」的 chunk。改完大小后，要到下一步点「重建」才会写进仓库。",
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
    for i, chunk in enumerate(chunks):
        src = Path(str(chunk.metadata.get("source", ""))).name
        title = chunk.page_content.strip().splitlines()[0][:40]
        with st.expander(f"chunk {i + 1} · {src} · {title}", expanded=(i == 0)):
            st.write(chunk.page_content)
            st.caption(f"{len(chunk.page_content)} 字")


def _step_index() -> None:
    _teach(
        "索引 = 切块后的文本变成向量，写入 **Chroma 向量数据库**（目录 `chroma_db/`）。不是再上传一份手册。打开教室时已自动入库。",
        "专门数据库是为了按向量近邻检索。Chroma 是嵌入式向量库；银行生产更多用带权限和备份的 pgvector / Milvus。没有这一步，检索就是空仓库。",
        "可以上传自己的 PDF / Word / Markdown。点下面预览：每条是一段原文 + 一串数字向量。云端上传重启可能丢失，长期请放进仓库 `data/kb/`。",
    )
    st.info("Chroma 已经是向量数据库，只是文件落在本机目录，不是「没有数据库」。")
    use_hf = st.checkbox(
        "使用句向量模型（首次下载模型会较慢；依赖已写入 requirements.txt，Cloud 需 Reboot 后才装上）",
        value=config.EMBEDDING_BACKEND == "huggingface",
    )
    if use_hf:
        config.EMBEDDING_BACKEND = "huggingface"
        config.COLLECTION_NAME = f"bank_kb_{config.EMBEDDING_BACKEND}"
    else:
        config.EMBEDDING_BACKEND = "hashed"
        config.COLLECTION_NAME = f"bank_kb_{config.EMBEDDING_BACKEND}"
    st.markdown("#### 放入真实文件")
    st.caption("支持 .pdf / .docx / .md / .txt。写入 `data/kb/uploads/`，然后请点重建索引。")
    uploaded = st.file_uploader(
        "选择文件（可多选）",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("保存到知识库", type="primary"):
        saved = []
        for item in uploaded:
            path = save_uploaded_file(item.name, item.getvalue())
            saved.append(path.name)
        st.success("已保存：" + "、".join(saved) + "。请再点「重建索引」才会进入向量库。")
    existing = list_uploads()
    if existing:
        st.markdown("**已上传（可删除）**")
        for path in existing:
            left, right = st.columns((4, 1))
            left.write(f"{path.name} · {path.stat().st_size} 字节")
            if right.button("删除", key=f"del_upload_{path.name}"):
                path.unlink()
                st.rerun()
    st.session_state.extra_note = st.text_area(
        "可选：写一条只属于你的知识（会随索引一起入库，不覆盖手册）",
        value=st.session_state.extra_note,
        placeholder="例如：星河银行大厅取号后超时 15 分钟需重新取号。",
        height=90,
    )
    if st.button("按当前切块重建索引", type="secondary"):
        extras = []
        note = st.session_state.extra_note.strip()
        if note:
            extras.append(
                Document(
                    page_content=f"## 学员补充\n{note}",
                    metadata={"source": "user-note.md", "file_type": "md", "page": 1},
                )
            )
        try:
            with st.spinner("切块、向量化、写入向量库 Chroma（句向量会较慢）…"):
                st.session_state.index_stats = build_index(
                    reset=True,
                    extra_docs=extras or None,
                    chunk_size=st.session_state.chunk_size,
                    chunk_overlap=st.session_state.chunk_overlap,
                )
            st.success("已重建。现在仓库里是当前切块与向量后端。")
        except ImportError as exc:
            st.error(str(exc))
    stats = st.session_state.index_stats
    if stats:
        a, b, c = st.columns(3)
        a.metric("文档", stats["documents"])
        b.metric("chunk", stats["chunks"])
        c.metric("向量库", stats.get("vector_store", "chroma"))
        types = stats.get("file_types") or []
        if types:
            st.caption("已解析格式：" + "、".join(types))
        st.caption(f"落盘目录：`{stats['persist_directory']}`")

    st.markdown("#### 预览向量库")
    st.caption("每条记录 = 一段原文 + 对应的浮点向量。下面只显示前 12 维，避免刷屏。")
    try:
        preview = preview_vectors(limit=20, offset=0)
    except Exception as exc:
        st.warning(f"还读不出向量：{exc}")
        preview = None
    if preview and preview["total"]:
        st.write(f"共 {preview['total']} 条，向量维度 {preview['dim']}，后端 `{preview['backend']}`")
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
        st.info("向量库暂时是空的，请先重建索引。")


def _step_question() -> None:
    _teach(
        "原句不会整句丢进仓库。先分词，再丢掉「是、吗、多少」等停用词。`data/terms.txt` 里的产品名（随心贷、房贷、违约金）会当成一个词。",
        "分词错了，检索就会空或跑偏。比如只切出「房」和「贷」，就对不上手册里的「住房按揭」。",
        "点现成问题会跳到检索。先看下面的分词芯片再跳。也可以故意问「明天股价会涨吗」看空手是什么样子。",
    )
    st.text_input("你的问题", key="question")
    st.caption("点这些会填进输入框，并跳到检索步：")
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
    terms = tokenize(st.session_state.question)
    st.markdown("**分词后用来检索的词**")
    if terms:
        chips = " ".join(f'<span class="chip">{html.escape(t)}</span>' for t in terms)
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.info("分词结果为空，检索会什么都找不到。换几个实词再试。")


def _use_sample_question(sample: str) -> None:
    # 回调在下一轮渲染、创建 text_input 之前执行，避免改已实例化的 widget key。
    st.session_state.question = sample
    st.session_state.step = 5


def _step_retrieve() -> None:
    _teach(
        "这一步是召回：用 BM25 或向量先多捞若干条，不急着只留最终的 top-k。生产里常召回 20～50 条，再交给重排。手册很小、问题很短时，也可以跳过重排、直接取 top-k。",
        "检索器擅长「别漏」：相关段落如果排在第 8 名，生成器永远看不见。先放大候选池，下一步重排再扔掉噪音。",
        "把「多召回几条」调大，对照表格。默认仍用 BM25。点开高亮看命中词。下一步才是精排。",
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state.fetch_k = st.slider("多召回几条 fetch-k", 4, 24, int(st.session_state.fetch_k))
    with c2:
        st.session_state.top_k = st.slider("重排后要留几条 top-k", 1, 8, st.session_state.top_k)
    with c3:
        st.session_state.retriever_mode = st.radio(
            "检索器",
            ["bm25", "vector"],
            index=0 if st.session_state.retriever_mode == "bm25" else 1,
            horizontal=True,
        )
    if st.session_state.fetch_k < st.session_state.top_k:
        st.session_state.fetch_k = st.session_state.top_k
    question = st.session_state.question
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
            st.warning(
                "BM25 认为这些词和仓库里的 chunk 没有足够重叠，所以分数全是 0。"
                f"当前分词：{'、'.join(terms) or '无'}。"
                "手册原文写的是「提前还款」「住房按揭 / 房贷」「违约金」。词对不上就会空。"
                "「明天股价会涨吗」这种手册里没有的问题，空结果才是正确行为。"
            )
        st.session_state["ranked"] = []
        st.session_state["reranked"] = []
        return
    rows = []
    for i, item in enumerate(ranked, start=1):
        src = Path(str(item["doc"].metadata.get("source", ""))).name
        rows.append(
            {
                "召回名次": i,
                "分数": item["score"],
                "含义": "向量距离(越小越近)" if item.get("score_kind") == "vector_distance" else "BM25(越大越好)",
                "命中词": "、".join(item["matched"]) or "—",
                "来源": src,
                "开头": item["doc"].page_content.strip().splitlines()[0][:32],
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.session_state["ranked"] = ranked
    st.caption("上面整张表都会进入下一步重排；生成器仍然只能看见重排后留下的几条。")
    for i, item in enumerate(ranked[:8], start=1):
        src = Path(str(item["doc"].metadata.get("source", ""))).name
        with st.expander(f"召回第 {i} 名 · 分数 {item['score']} · {src}", expanded=(i == 1)):
            st.markdown(_highlight_html(item["doc"].page_content, item["matched"]), unsafe_allow_html=True)
            st.caption("高亮 = 和问题分词重叠的词。没有高亮却被召回，多半是向量近邻，语义像但用词不同。")


def _step_rerank() -> None:
    _teach(
        "重排用「问题和这段话对得上的程度」再打一遍分，只留下 top-k 给生成器。教室里用词重叠 + 标题加权；生产里常换成 Cross-Encoder / bge-reranker。",
        "小知识库、BM25 已经很准时，这一步收益有限，可以不加。候选一多、向量召回噪音大时，几乎都会加。官方 rag-from-scratch 三步里没有它，所以它是可选增强，不是 RAG 定义的一部分。",
        "对照左右两张表：左边是召回顺序，右边是精排后留下的。名次若对调，说明第一轮检索把更相关的段落排到了后面。",
    )
    question = st.session_state.question
    ranked = st.session_state.get("ranked")
    if ranked is None:
        try:
            ranked = retrieve_ranked(
                question,
                k=st.session_state.top_k,
                retriever=st.session_state.retriever_mode,
                fetch_k=st.session_state.fetch_k,
            )
            st.session_state["ranked"] = ranked
        except Exception as exc:
            st.error(f"请先完成索引和检索：{exc}")
            return
    if not ranked:
        st.warning("召回为空，没有可重排的段落。回到检索步换一个问题。")
        st.session_state["reranked"] = []
        return
    kept = rerank_hits(question, ranked, keep=st.session_state.top_k)
    st.session_state["reranked"] = kept
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
        "生成器只能看见重排后留下的几段。Prompt 在调用模型之前约束它；后处理在模型写完之后检查：空检索拒答、去掉代码围栏、补 (资料N)、列出引用文件。",
        "改 Prompt 只能影响「模型怎么写」。引用列表、拒答、格式清洗属于规则，放在生成之后更稳，也不消耗一次额外的模型调用。",
        "先看 Prompt，再看「模型原文」和「后处理之后」。对照第一名原文。没有 Key 时只有抽取式摘要，后处理仍会补引用文件。",
    )
    ranked = st.session_state.get("reranked") or st.session_state.get("ranked")
    question = st.session_state.question
    if not ranked:
        try:
            recalled = retrieve_ranked(
                question,
                k=st.session_state.top_k,
                retriever=st.session_state.retriever_mode,
                fetch_k=st.session_state.fetch_k,
            )
            ranked = rerank_hits(question, recalled, keep=st.session_state.top_k)
            st.session_state["ranked"] = recalled
            st.session_state["reranked"] = ranked
        except Exception as exc:
            st.error(f"请先完成索引和检索：{exc}")
            return
    docs = [item["doc"] for item in ranked]
    if not docs:
        st.warning("检索为空，生成器没有上下文可用。回到检索步换个问题。")
        raw = "知识库里没有检索到相关段落，请先运行索引或换一个问法。"
        final, notes = postprocess_answer(raw, question, docs)
        st.write(final)
        st.caption("后处理动作：" + ("、".join(notes) if notes else "无"))
        return
    gen, name = build_generator()
    try:
        raw = gen.generate(question, docs)
    except Exception as exc:
        st.error(f"大模型调用失败（检索结果仍在右侧）。请检查 Key、模型名和 Base URL。详情：{exc}")
        from rag.generate import ExtractiveGenerator

        raw = ExtractiveGenerator().generate(question, docs)
        name = f"{name}（调用失败，回退抽取）"
    final, notes = postprocess_answer(raw, question, docs)
    left, right = st.columns(2)
    with left:
        st.markdown("#### 后处理之后的答案（用户看到的）")
        st.write(final)
        st.caption(f"当前生成器：`{name}` · 后处理：" + ("、".join(notes) if notes else "无"))
        with st.expander("模型刚写完、还没后处理的原文"):
            st.write(raw)
    with right:
        st.markdown("#### 它只看见这些资料（重排后）")
        for i, doc in enumerate(docs, start=1):
            src = Path(str(doc.metadata.get("source", ""))).name
            st.markdown(f"**资料{i} · {src}**")
            st.write(doc.page_content[:280] + ("…" if len(doc.page_content) > 280 else ""))
    with st.expander("① 发给模型的 Prompt（生成前，靠这段话锁死答案）", expanded=True):
        st.code(preview_prompt(question, docs), language="markdown")
    with st.expander("② 后处理做了什么（生成后，不是再改 Prompt）", expanded=True):
        st.markdown(
            """
- **空检索拒答**：没有资料就不让模型编利率。
- **去掉代码围栏**：有的模型喜欢包一层 markdown。
- **补引用**：正文里如果没有 `(资料N)`，补上。
- **列出文件名**：方便对照来源。生产里还可以做忠实度检查、敏感信息过滤。
"""
        )
        st.caption("本轮实际执行：" + ("、".join(notes) if notes else "无"))
    if name.startswith("agnes:"):
        st.success(f"本步已调用 Agnes 大模型（`{name}`）。检索仍在本地，模型只根据上面的资料作答。")
    elif name == "extractive":
        st.warning(
            "现在没有调用大模型，所以答案只是检索到的原文摘录。"
            "在 Streamlit Cloud：App → Settings → Secrets 添加 `AGNES_API_KEY`，可选 `AGNES_MODEL = \"agnes-2.5-flash\"`，然后 Reboot。"
            "本地可把同样内容放进 `.streamlit/secrets.toml` 或 `.env`。"
        )


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
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
