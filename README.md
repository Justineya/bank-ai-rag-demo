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
向量 (默认 bge-small-zh；哈希为教学开关)
    │  Embed
    ▼
Chroma 向量库
    │  Store
    ▼
问题 → BM25 / 向量 多召回 fetch-k    Retrieve
    │
    ▼
bge-reranker 精排，只留 top-k        Rerank（词重叠为对照开关）
    │
    ▼
Prompt + LLM / 抽取式回答            Generate
    │
    ▼
补引用 / 空检索拒答                  Post-process
```

| 层 | 本 Demo 的选择 | 以后可以换成 |
| --- | --- | --- |
| 框架 | LangChain LCEL 风格的模块拆分 | LlamaIndex、自写 cosine |
| 切块 | `RecursiveCharacterTextSplitter` | 按标题 / 按 token |
| 检索 | 默认 BM25（jieba 分词） | Chroma 向量检索 `RAG_RETRIEVER=vector` |
| 重排 | 默认 bge-reranker（fetch-16 → top-4） | 词重叠对照 `RAG_RERANKER=lexical` |
| 向量库 | Chroma（本地目录） | FAISS、pgvector |
| 向量 | 默认 `BAAI/bge-small-zh-v1.5` | 教学哈希 `EMBEDDING_BACKEND=hashed` |
| 生成 | 无 Key 时抽取式；有 Key 用 Agnes / OpenAI / Groq | Ollama 本地模型 |
| 后处理 | 补引用、拒答、冲突稿后置 | 忠实度打分、敏感词过滤 |
| 评测 | `data/eval/questions.json` + 教室「跑评测」 | 更大回归集 |

默认真句向量 + Cross-Encoder 重排。哈希 / 词重叠只留教学开关，方便对照「为什么教学环境曾用哈希：不下载模型、秒级看完步骤」。不配 API Key 也能完成检索 + 抽取式回答闭环。

## 目录

```
data/kb/          知识库：Markdown + 住房贷款 PDF + 信用卡章程 Word
data/kb/uploads/  上传的真实文件（默认不进 Git）
scripts/          把 Markdown 导出成 PDF/Word
data/eval/        固定评测集（能答 / 拒答 / 易混 / 跨文档 / 冲突）
data/terms.txt    jieba 用户词典（产品名）
src/rag/          流水线源码，一文件一层
  config.py       路径与模型
  embeddings.py   文本 → 向量
  ingest.py       建索引
  retrieve.py     检索（召回）
  rerank.py       重排（bge / 词重叠）
  generate.py     生成 + 后处理 + 结果卡字段
  eval.py         一键跑分
  pipeline.py     ask() 入口
  tenants.py      星河 / 闽信叙事
  acl.py          对客·对内·生效日
  audit.py        会话审计 JSON
  warmup.py       冷启动预热
cli.py            命令行（`--tenant minxin`）
app.py            Streamlit 讲解式教室
tests/            不依赖外网模型的回归测试
```

## 闽信保险叙事（P2）

星河银行仍是虚构制度稿。闽信保险教室用官网公开材料：

- 来源表：`data/minxin/SOURCES.md`
- 摘录：`data/minxin/kb/`（YAML 头：`audience` 对客/对内、`effective_date`、`status`）
- 官网 HTML 与 2025 年报 PDF 备份：`data/minxin/source/`
- 财务数字只引用官网/年报原文。官网保险收入约 **19,727 万** 与年报已审计 **19,663 万** 并存时，对客以年报为准。教学用「约 3 亿」快报标成已废止，不是官网文件。

顶部切换叙事。检索允许集默认**对客**：对内偿付/再保摘录与已废止稿不进召回。切到「对内」才能看到那部分。

生成页可导出一条审计 JSON：问题、命中 chunk id、模型、是否拒答。

第一次打开出现 **唤醒中**，会预热索引，避免 Streamlit 休眠后只剩转圈。

途港课件 `/learn/banking-ai` 可带查询参数打开教室，例如：

```
?from=tugang&lesson=banking-ai&preset=nim
?from=tugang&lesson=banking-ai&preset=demand
?tenant=minxin&preset=northbound
```

命令行：

```bash
python cli.py --tenant minxin --rebuild
python cli.py --tenant minxin "闽信保险是哪一年成立的？"
python cli.py --tenant minxin --eval
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

1. 在「知识库」打开制度，也可在这一步上传自己的 PDF / Word  
2. 拖动切块大小，看 chunk 如何变化  
3. 到「索引」点重建，把知识库写入向量库（不要在索引步找上传）  
4. 点现成问题，看分词芯片  
5. 在检索页看召回表（fetch-k 可以大于 top-k）  
6. 在重排页对照「召回顺序 vs 精排留下的」  
7. 在生成页展开 Prompt，再看后处理补了哪些引用

第一次打开页面会自动建索引。Streamlit Cloud 请把 Branch 设为功能分支，Main file 填 `app.py`。

命令行仍然可用：

```bash
python cli.py --rebuild
python cli.py "活期利率是多少？"
python cli.py --eval
```

## 建议的学习顺序

1. 打开 `data/kb`，看知识库长什么样。
2. 读 `src/rag/ingest.py`：切块大小 `CHUNK_SIZE=400`、重叠 `80` 会怎样影响检索。
3. 读 `src/rag/embeddings.py` 与 `src/rag/textutil.py`：先理解 BM25，再对比哈希向量。
4. 运行 `python cli.py --rebuild` 后看 `chroma_db/` 是否生成。
5. 读 `src/rag/generate.py`：对比抽取式回答、Chat Prompt、以及 `postprocess_answer`。
6. 读 `src/rag/rerank.py`：召回多条后再精排。小语料上名次可能不变，这是正常的。
7. 检索页选 Hybrid：一张表看 BM25 / 向量 / 融合名次。
8. 索引页默认已是句向量。勾选「教学开关：哈希」可对照不下载模型时的步骤；换后端后必须重建。
9. 开场页点「跑评测」：命中率、拒答正确率、引用点名率。命令行：`python cli.py --rebuild --eval`。
10. 切块页拖 `chunk_size` / `overlap`，立刻看当前问题命中哪一段、重叠多少字。
11. 知识库里的营销页和过期 PDF 是噪声，用来演示「以哪份为准」。

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

