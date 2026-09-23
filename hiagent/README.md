# HiAgent 控制台骨架

部署火山引擎 [HiAgent](https://www.volcengine.com/product/hiagent) **之前**的前端框架，用来对齐产品模块和页面信息架构。

这不是官方安装包，也不接火山 API。现有星河/闽信 RAG 教室（`app.py`）未被改动。

## 对照的官方能力

| 本骨架页面 | HiAgent 公开能力 |
| --- | --- |
| 工作空间 | Canvas 门户、数字员工协同入口 |
| 智能体 / 工作室 | 提示词 + 知识库 + 插件 + 工作流低代码搭建 |
| 知识库 | 文档导入、分段、召回 |
| 插件与 MCP | 插件市场、MCP Gateway、企业系统接入 |
| 工作流 | 可视化编排、分支与兜底 |
| 场景意图 | 意图识别、多智能体调度 |
| 评测 | 多维度效果评估 |
| 观测 | 调用量、延迟、用户反馈 |
| 发布 | 飞书 / 钉钉 / 企微 / API / WebSDK |
| 治理 | 权限、审计、数据不出域、模型防火墙 |

## 本地运行

```bash
cd hiagent
npm install
npm run dev
```

浏览器打开 http://localhost:5173
