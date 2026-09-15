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
from rag.generate import preview_prompt, build_generator
from rag.ingest import build_index, ensure_index, load_documents, split_documents
from rag.retrieve import retrieve_ranked
from rag.textutil import tokenize

STEPS = [
    ("开场", "RAG 在解决什么"),
    ("知识库", "Load：读进哪些文档"),
    ("切块", "Split：为什么要切开"),
    ("索引", "Embed + Store：写进仓库"),
    ("提问", "把问题变成检索词"),
    ("检索", "Retrieve：谁排第一"),
    ("生成", "Generate：答案从哪来"),
]

SAMPLE_QUESTIONS = [
    "活期利率是多少？",
    "信用卡还款日是哪天？",
    "随心贷能用来炒股吗？",
    "提前还房贷要不要违约金？",
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
#### 先记住三件事（后面每一步都在落实）

1. **知识库**是人写好的 Markdown，模型事先没读过。  
2. **索引**把手册切成小段放进仓库，提问时才能搜。你不用往里面「添加」产品——手册已经在仓库里，打开页面时会自动入库。  
3. **检索**只取出几小段给生成器用。生成器看不见全文，所以排不到前面的段落等于不存在。
"""
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("知识库文档", "5 篇+", "储蓄 / 信用卡 / 贷款…")
    c2.metric("默认检索", "BM25", "按词匹配，适合专有名词")
    c3.metric("默认生成", "抽取原文", "没有 API Key 也能学")
    st.markdown("#### 整条流水线")
    st.code(
        "文档 → 切块 → 写入仓库(索引) → 问题分词 → 检索 top-k → 填进 Prompt → 生成答案",
        language="text",
    )
    if st.button("从知识库开始参观", type="primary"):
        st.session_state.step = 1
        st.rerun()


def _step_load() -> None:
    docs = load_documents()
    _teach(
        "Load 只是把 `data/kb` 里的 Markdown 读进内存：一篇文件 = 一个文档对象。这里还没有搜索，也没有答案。",
        "后面所有检索都只能搜到这些字。如果利率写在储蓄手册里，你却只索引了别的文件，问利率就会失败。",
        "左侧点文件名看原文。请先找到「0.20%」和「提前还款 / 违约金」分别在哪一篇，后面对照检索结果。",
    )
    names = [Path(doc.metadata["source"]).name for doc in docs]
    left, right = st.columns((1, 2))
    with left:
        picked = st.radio("点一篇打开", names, index=0)
        st.session_state.picked_source = picked
        st.caption(f"一共 {len(docs)} 篇，全部会进入下一步切块。")
    with right:
        doc = next(d for d in docs if Path(d.metadata["source"]).name == picked)
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
        "索引 = 把切好的段落放进可搜索的仓库（Chroma）。不是让你再上传一份银行资料。打开教室时已经用手册自动建过一次。",
        "没有索引时，检索对着空仓库，任何问题都是 0 条结果。这不是模型坏了，是还没有可搜的存货。",
        "只有改了切块大小、或在下面写了自己的规定时，才需要点「重建索引」。日常提问不用再点。",
    )
    st.info("你不需要「添加」星河银行手册。它已经在 `data/kb`。索引只是把这些文件做成可搜索状态。")
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
            extras.append(Document(page_content=f"## 学员补充\n{note}", metadata={"source": "user-note.md"}))
        with st.spinner("切块、向量化、写入 Chroma…"):
            st.session_state.index_stats = build_index(
                reset=True,
                extra_docs=extras or None,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
            )
        st.success("已重建。现在仓库里是当前切块设置（以及你写的补充，如果有）。")
    stats = st.session_state.index_stats
    if stats:
        a, b, c = st.columns(3)
        a.metric("文档", stats["documents"])
        b.metric("chunk", stats["chunks"])
        c.metric("embedding", stats["embedding_backend"])
        st.caption(f"落盘目录：`{stats['persist_directory']}`")
    else:
        st.caption("仓库已有内容。不必再点重建，除非你改了切块或写了补充规定。")


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
        "检索器给每个 chunk 打分，只把前 k 名交给生成器。默认 BM25 看「词是否出现」；vector 用教学哈希向量看「向量近不近」。排不到前面的段落，生成器看不见。",
        "结果为空通常有三种原因：仓库是空的（没索引）；问句用词和手册不一致；你开了 vector，而教学哈希向量对中文很弱。",
        "先保持 BM25。点开第一名看黄字命中。只有想对比时才切 vector。改 top-k 看会不会多捞到后排资料。",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.top_k = st.slider("取回几条 top-k", 1, 8, st.session_state.top_k)
    with c2:
        st.session_state.retriever_mode = st.radio(
            "检索器",
            ["bm25", "vector"],
            index=0 if st.session_state.retriever_mode == "bm25" else 1,
            horizontal=True,
        )
    question = st.session_state.question
    st.markdown(f"当前问题：`{question}`")
    try:
        ranked = retrieve_ranked(question, k=st.session_state.top_k, retriever=st.session_state.retriever_mode)
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
        return
    rows = []
    for i, item in enumerate(ranked, start=1):
        src = Path(str(item["doc"].metadata.get("source", ""))).name
        rows.append(
            {
                "名次": i,
                "分数": item["score"],
                "含义": "向量距离(越小越近)" if item.get("score_kind") == "vector_distance" else "BM25(越大越好)",
                "命中词": "、".join(item["matched"]) or "—",
                "来源": src,
                "开头": item["doc"].page_content.strip().splitlines()[0][:32],
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.session_state["ranked"] = ranked
    for i, item in enumerate(ranked, start=1):
        src = Path(str(item["doc"].metadata.get("source", ""))).name
        with st.expander(f"第 {i} 名 · 分数 {item['score']} · {src}", expanded=(i == 1)):
            st.markdown(_highlight_html(item["doc"].page_content, item["matched"]), unsafe_allow_html=True)
            st.caption("高亮 = 和问题分词重叠的词。没有高亮却被召回，多半是向量近邻，语义像但用词不同。")


def _step_generate() -> None:
    _teach(
        "生成器不能偷看手册全文，只能看见上一步挑出的几段。接上 Agnes Key 后，这一步才会调用大模型；没 Key 时只展示检索原文。",
        "答案被 Prompt 锁死。检索错了，生成再强也会错。空检索就不该编利率。",
        "展开 Prompt，对照答案和第一名原文是否一致。",
    )
    ranked = st.session_state.get("ranked")
    question = st.session_state.question
    if not ranked:
        try:
            ranked = retrieve_ranked(question, k=st.session_state.top_k, retriever=st.session_state.retriever_mode)
            st.session_state["ranked"] = ranked
        except Exception as exc:
            st.error(f"请先完成索引和检索：{exc}")
            return
    docs = [item["doc"] for item in ranked]
    if not docs:
        st.warning("检索为空，生成器没有上下文可用。回到上一步换个问题。")
        return
    gen, name = build_generator()
    try:
        answer = gen.generate(question, docs)
    except Exception as exc:
        st.error(f"大模型调用失败（检索结果仍在右侧）。请检查 Key、模型名和 Base URL。详情：{exc}")
        from rag.generate import ExtractiveGenerator

        answer = ExtractiveGenerator().generate(question, docs)
        name = f"{name}（调用失败，回退抽取）"
    left, right = st.columns(2)
    with left:
        st.markdown("#### 答案")
        st.write(answer)
        st.caption(f"当前生成器：`{name}`")
    with right:
        st.markdown("#### 它只看见这些资料")
        for i, doc in enumerate(docs, start=1):
            src = Path(str(doc.metadata.get("source", ""))).name
            st.markdown(f"**资料{i} · {src}**")
            st.write(doc.page_content[:280] + ("…" if len(doc.page_content) > 280 else ""))
    with st.expander("打开将要发给模型的 Prompt（参与感就在这里：答案被这段话锁死）", expanded=True):
        st.code(preview_prompt(question, docs), language="markdown")
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
