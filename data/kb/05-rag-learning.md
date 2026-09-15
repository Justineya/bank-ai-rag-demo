# RAG 学习笔记（写入知识库，方便你拿 Demo 自己提问）

RAG 的全称是 Retrieval-Augmented Generation，即检索增强生成。

## 为什么需要 RAG

大模型的参数知识截止于训练数据。银行产品利率、内部制度、最新公告通常不在模型训练语料里，也不能把客户隐私拿去微调。RAG 的做法是：先从本地知识库检索相关段落，再把段落作为上下文交给模型生成答案。

## 标准流水线（本 Demo 的实现顺序）

1. Load：读取 `data/kb` 下的 Markdown 文档。
2. Split：按字符切成有重叠的 chunk，避免一段话被拦腰截断。
3. Embed：把 chunk 变成向量。本 Demo 默认用哈希向量，可切换成 HuggingFace 句向量。
4. Store：写入 Chroma 向量库，便于重复查询。
5. Retrieve：把用户问题也向量化，取最相似的 top-k 个 chunk。
6. Generate：把 chunk 填进 Prompt，让 LLM 只根据资料回答；没有 API Key 时使用抽取式回答。

## 评价一个 RAG 好不好

- 检索是否把正确段落找回来（召回）。
- 答案是否只使用检索到的内容（忠实，减少幻觉）。
- 引用是否能回到原文（可追溯）。
