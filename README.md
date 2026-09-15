# 星河银行 RAG Demo

从零搭建的检索增强生成（RAG）教学项目。目标不是堆功能，而是把官方教程里的 **Indexing / Retrieval / Generation** 三步写成一份你能跑通、能改、能对照的代码。

参考：

- [langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)（Lance Martin，Indexing → Retrieval → Generation）
- 领域语料用虚构的「星河银行」产品说明，避免一上来就接真实核心系统

## 你将学到的框架

```
文档 (Markdown)
    │  Load
    ▼
切块 (RecursiveCharacterTextSplitter)
    │  Split
    ▼
向量 (Hashed n-gram 或 HuggingFace)
    │  Embed
    ▼
Chroma 向量库
    │  Store
    ▼
问题 → BM25 / 向量 top-k     Retrieve
    │
    ▼
Prompt + LLM / 抽取式回答    Generate
```

| 层 | 本 Demo 的选择 | 以后可以换成 |
| --- | --- | --- |
| 框架 | LangChain LCEL 风格的模块拆分 | LlamaIndex、自写 cosine |
| 切块 | `RecursiveCharacterTextSplitter` | 按标题 / 按 token |
| 检索 | 默认 BM25（jieba 分词） | Chroma 向量检索 `RAG_RETRIEVER=vector` |
| 向量库 | Chroma（本地目录） | FAISS、pgvector |
| 生成 | 无 Key 时抽取式；有 Key 用 OpenAI / Groq | Ollama 本地模型 |

默认走哈希 Embedding + 抽取式生成，所以 **不配 API Key 也能完成闭环**。配上 `OPENAI_API_KEY` 或 `GROQ_API_KEY` 后，生成会切换成对话模型。

## 目录

```
data/kb/          知识库：Markdown + 住房贷款 PDF + 信用卡章程 Word
scripts/          把 Markdown 导出成 PDF/Word
data/terms.txt    jieba 用户词典（产品名）
src/rag/          流水线源码，一文件一层
  config.py       路径与模型
  embeddings.py   文本 → 向量
  ingest.py       建索引
  retrieve.py     检索
  generate.py     生成
  pipeline.py     ask() 入口
cli.py            命令行
app.py            Streamlit 讲解式教室
tests/            不依赖外网模型的回归测试
```

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 可选，填入 API Key

python cli.py --rebuild
python cli.py "活期利率是多少？"
python cli.py "随心贷能用来炒股吗？"

streamlit run app.py
```

打开后是 **逐步讲解的教室**：顶部点步骤，每一步有「这一步在做什么 / 你可以点什么」。建议顺序：

1. 点开知识库文件，找到利率和随心贷写在哪  
2. 拖动切块大小，看 chunk 如何变化  
3. 可选写一条自己的规定，再点「写入索引」  
4. 点现成问题，看分词芯片  
5. 在检索页看分数、点开高亮命中词  
6. 在生成页展开 Prompt，对照答案和原文

第一次提问前需要在教室第 4 步「索引」写入向量库。

命令行仍然可用：

```bash
python cli.py --rebuild
python cli.py "活期利率是多少？"
```

## 建议的学习顺序

1. 打开 `data/kb`，看知识库长什么样。
2. 读 `src/rag/ingest.py`：切块大小 `CHUNK_SIZE=400`、重叠 `80` 会怎样影响检索。
3. 读 `src/rag/embeddings.py` 与 `src/rag/textutil.py`：先理解 BM25，再对比哈希向量。
4. 运行 `python cli.py --rebuild` 后看 `chroma_db/` 是否生成。
5. 读 `src/rag/generate.py`：对比抽取式回答和 Chat Prompt。
6. 设置 `RAG_RETRIEVER=vector` 重建后再问同一问题，看排序如何变化。
7. 索引页勾选「句向量」：Cloud 会按 `requirements.txt` 安装 `sentence-transformers`，第一次重建索引会下载模型，可能要等几分钟。
8. 换一篇自己的 Markdown 放进 `data/kb`，把产品名写入 `data/terms.txt`，重建索引。

## 测试

```bash
python -m pytest -q
```

测试使用临时 Chroma 目录和哈希向量，不调用外部 LLM。覆盖 PDF/Word 解析。

把制度稿再导出为办公格式：

```bash
python scripts/export_office_docs.py
```

## 可以部署到哪里（不改架构）

这套教室是 **Streamlit 常驻进程**（WebSocket + 本地 Chroma）。Vercel 只有无服务器函数，跑不了，也不该为此改成静态页。

能原样部署的地方：

| 平台 | 适合原因 |
| --- | --- |
| [Streamlit Community Cloud](https://share.streamlit.io) | 官方免费托管，指向本仓库的 `app.py` 即可 |
| [Hugging Face Spaces](https://huggingface.co/spaces)（SDK 选 Streamlit） | 同样跑 `app.py`，不改代码 |
| Render / Railway / Fly.io | 用仓库里的 `Dockerfile` 起容器 |

### Streamlit Cloud（最省事）

1. 把 PR 合并进 `main`，或在 Cloud 里选分支 `cursor/rag-from-scratch-demo-f62d`
2. 打开 https://share.streamlit.io → New app → 选 `Justineya/bank-ai-rag-demo`
3. Main file 填 `app.py`
4. 要用 Agnes 大模型：App → **Settings → Secrets** 填入（不要提交到 Git）

```toml
AGNES_API_KEY = "你的密钥"
AGNES_MODEL = "agnes-2.5-flash"
AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1"
```

然后 Reboot。生成那一步会显示 `agnes:agnes-2.5-flash`。检索仍在本地，Key 只用于最后根据资料写答案。


### Docker（Render / 自己的机器）

```bash
docker build -t bank-rag .
docker run -p 8501:8501 bank-rag
```

云端容器请把端口映射到平台提供的 `PORT`（镜像已读取该环境变量）。

