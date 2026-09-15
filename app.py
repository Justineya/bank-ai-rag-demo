from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag import config
from rag.ingest import build_index
from rag.pipeline import ask

st.set_page_config(page_title="星河银行 RAG Demo", page_icon="🏦", layout="wide")
st.title("星河银行 RAG Demo")
st.caption("教学流水线：Load → Split → Embed → Store → Retrieve → Generate")

with st.sidebar:
    st.subheader("索引")
    st.write(f"知识库目录：`data/kb`")
    st.write(f"Embedding：`{config.EMBEDDING_BACKEND}`")
    top_k = st.slider("检索条数 top-k", 1, 8, config.TOP_K)
    if st.button("重建索引", type="primary"):
        with st.spinner("正在切分并写入 Chroma…"):
            stats = build_index(reset=True)
        st.success(f"{stats['documents']} 篇文档 → {stats['chunks']} 个 chunk")

    st.subheader("示例问题")
    st.markdown(
        "- 活期利率是多少？\n"
        "- 信用卡还款日是哪天？\n"
        "- 随心贷能用来炒股吗？\n"
        "- RAG 的标准流水线有哪几步？"
    )

question = st.text_input("问知识库", placeholder="例如：提前还房贷要不要违约金？")
submit = st.button("检索并生成", type="secondary")

if submit and not question:
    st.warning("请先输入问题。")
elif question and submit:
    try:
        result = ask(question, k=top_k)
    except Exception as exc:
        st.error(f"检索失败：{exc}")
        st.info("第一次使用请先在左侧点击「重建索引」。")
    else:
        left, right = st.columns((1, 1))
        with left:
            st.markdown("### 答案")
            st.write(result.answer)
            st.caption(f"生成器：{result.generator}")
        with right:
            st.markdown("### 检索到的资料")
            for i, doc in enumerate(result.sources, start=1):
                src = Path(str(doc.metadata.get("source", ""))).name
                with st.expander(f"资料{i} · {src}", expanded=(i == 1)):
                    st.write(doc.page_content)
elif not question:
    st.info("建议路径：左侧重建索引 → 输入问题 → 对照右侧资料看答案是否忠实。")
