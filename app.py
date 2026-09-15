from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import streamlit as st
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag import config
from rag.generate import preview_prompt, build_generator
from rag.ingest import build_index, load_documents, split_documents
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
    _init_state()

    st.title("星河银行 RAG 教室")
    st.caption("每一步都能点、能改。看完一遍，你就知道 RAG 为什么不是「把文档丢给模型」那么简单。")
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
        "picked_source": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _pipeline_nav() -> None:
    cols = st.columns(len(STEPS))
    for i, (short, _) in enumerate(STEPS):
        with cols[i]:
            label = f"{i + 1}. {short}"
            if st.button(label, use_container_width=True, type="primary" if i == st.session_state.step else "secondary"):
                st.session_state.step = i
                st.rerun()


def _teach(do_what: str, click_what: str) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown(f'<div class="card teach"><h4>这一步在做什么</h4><p>{do_what}</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="card act"><h4>你可以点什么</h4><p>{click_what}</p></div>', unsafe_allow_html=True)


def _step_intro() -> None:
    _teach(
        "大模型只记得训练时见过的内容。银行利率、内部制度、你刚写的规则，它往往不知道，还可能编造。"
        "RAG 的办法是：先从本地资料里找出相关段落，再让模型「看着资料」回答。",
        "点顶部的步骤条可以跳着看；也可以一直点页面底部的「下一步」。建议先按顺序走完一遍。",
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("知识库文档", "5 篇+", "储蓄 / 信用卡 / 贷款…")
    c2.metric("默认检索", "BM25", "中文关键词更稳")
    c3.metric("默认生成", "抽取原文", "没有 API Key 也能学")
    st.markdown("#### 整条流水线")
    st.code(
        "文档 → 切块 → 写入 Chroma → 问题分词 → 检索 top-k → 填进 Prompt → 生成答案",
        language="text",
    )
    if st.button("从知识库开始参观", type="primary"):
        st.session_state.step = 1
        st.rerun()


def _step_load() -> None:
    docs = load_documents()
    _teach(
        "Load 只做一件事：把 `data/kb` 里的 Markdown 读成「一篇文档一个对象」。还没有向量，也还没有答案。",
        "左侧点文件名看原文。试着找到「随心贷」和「0.20%」分别写在哪一篇——后面检索就靠这些字。",
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
        "一篇手册太长，相似度会被稀释。所以按 `##` 标题切开，太长的段落再按字数切开，并留一点 overlap，避免一句话被拦腰截断。",
        "拖动切块大小，观察 chunk 数量变化。点开某个 chunk，看看「随心贷」是独立一段，还是和房贷挤在一起。",
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
        "每个 chunk 会变成向量（默认是教学用哈希向量），再和原文一起写入本地 Chroma。"
        "提问时默认用 BM25 在这些 chunk 上打分；向量检索可以稍后对比。",
        "先随便写一条「你自己的规定」，再点「写入索引」。下两步提问时，可以问这条内容，验证 RAG 用的是你刚放进去的资料。",
    )
    st.session_state.extra_note = st.text_area(
        "可选：写一条只属于你的知识（会随索引一起入库，不覆盖手册）",
        value=st.session_state.extra_note,
        placeholder="例如：星河银行大厅取号后超时 15 分钟需重新取号。",
        height=90,
    )
    if st.button("写入 / 重建索引", type="primary"):
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
        st.success("索引已写入。下一步可以提问了。")
    stats = st.session_state.index_stats
    if stats:
        a, b, c = st.columns(3)
        a.metric("文档", stats["documents"])
        b.metric("chunk", stats["chunks"])
        c.metric("embedding", stats["embedding_backend"])
        st.caption(f"落盘目录：`{stats['persist_directory']}`")
    else:
        st.warning("还没有索引。不点写入的话，后面检索会失败或用到旧数据。")


def _step_question() -> None:
    _teach(
        "用户原话不会直接拿去比向量。中文要先分词，并丢掉「是、吗、多少」这类停用词。产品名来自 `data/terms.txt`，所以「随心贷」会当成一个词。",
        "点下面的现成问题，或自己改一句话。试一个知识库里没有的问题（比如股价），记住分词结果，到下一步看检索会不会空手。",
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
        "检索器给每个 chunk 打分，只把前 k 名交给生成器。默认 BM25：词越稀有、命中越多、标题里出现，分越高。排不到前面的段落，后面的「模型」根本看不见。",
        "改 top-k、切换 BM25 / 向量，观察第一名会不会换人。点开一条，黄色高亮就是命中的检索词。",
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
        st.error(f"还没有可用索引：{exc}")
        st.info("回到「索引」那一步点写入。")
        return
    if not ranked:
        st.warning("没有 chunk 得分大于 0。可能是还没建索引，或问题里的词手册里都没有——这正是 RAG 该说「资料里没有」的时候。")
        return
    rows = []
    for i, item in enumerate(ranked, start=1):
        src = Path(str(item["doc"].metadata.get("source", ""))).name
        rows.append(
            {
                "名次": i,
                "分数": item["score"],
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
        "生成器不能偷看知识库全文，只能看见上一步挑出来的资料。没有 API Key 时，我们诚实地把第一名原文当作答案；有 Key 时，同一份 Prompt 会交给 LLM，并要求「资料没有就说没有」。",
        "展开 Prompt 看模型实际吃进什么。对照答案和第一名原文：如果原文错了，答案也会错——这就是 RAG 的诚实之处。",
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
    answer = gen.generate(question, docs)
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
    if name == "extractive":
        st.info("现在没有调用大模型。配上 OPENAI_API_KEY 或 GROQ_API_KEY 后重启，生成器会换成对话模型，但检索步骤完全不变。")


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
        .card { border-radius: 16px; padding: 16px 18px; min-height: 132px; }
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
