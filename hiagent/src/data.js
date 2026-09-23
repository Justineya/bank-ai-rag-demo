export const agents = [
  {
    id: "it-desk",
    name: "IT 自助助手",
    status: "已发布",
    kind: "HiAgent 自建",
    scene: "运维咨询 / 权限 / 工单",
    model: "豆包·Pro",
    calls: 12840,
    resolve: "78%",
  },
  {
    id: "kb-qa",
    name: "制度问答助手",
    status: "评测中",
    kind: "HiAgent 自建",
    scene: "手册检索 + 拒答",
    model: "豆包·Lite",
    calls: 4320,
    resolve: "71%",
  },
  {
    id: "ticket",
    name: "工单预审员",
    status: "草稿",
    kind: "工作流智能体",
    scene: "严重度分级 → ITSM",
    model: "豆包·Pro",
    calls: 0,
    resolve: "—",
  },
  {
    id: "ark",
    name: "火山方舟通用助手",
    status: "已纳管",
    kind: "火山智能体",
    scene: "办公问答",
    model: "方舟托管",
    calls: 960,
    resolve: "64%",
  },
];

export const knowledgeBases = [
  { name: "IT 运维手册", docs: 86, chunks: 1240, status: "已训练", recall: 0.72 },
  { name: "权限与合规制度", docs: 24, chunks: 310, status: "已训练", recall: 0.68 },
  { name: "FAQ 常见问题", docs: 140, chunks: 520, status: "训练中", recall: 0.7 },
];

export const plugins = [
  { name: "ITSM 建单", type: "插件", scope: "内部", status: "已启用" },
  { name: "OA 审批", type: "插件", scope: "内部", status: "已启用" },
  { name: "企业通讯录", type: "插件", scope: "内部", status: "已启用" },
  { name: "飞书消息", type: "渠道", scope: "IM", status: "已启用" },
  { name: "GitLab MCP", type: "MCP", scope: "研发", status: "待审批" },
  { name: "Confluence MCP", type: "MCP", scope: "知识", status: "已启用" },
];

export const evalRows = [
  { set: "IT 咨询 80 题", hit: "86%", refuse: "91%", cite: "80%", pass: "84%" },
  { set: "权限申请 40 题", hit: "79%", refuse: "88%", cite: "74%", pass: "78%" },
  { set: "越权/幻觉 20 题", hit: "—", refuse: "95%", cite: "—", pass: "95%" },
];

export const traces = [
  { id: "tr-10421", agent: "IT 自助助手", q: "VPN 连不上怎么办", latency: "1.8s", result: "已答复", channel: "工作空间" },
  { id: "tr-10418", agent: "制度问答助手", q: "出差报销要几级审批", latency: "2.4s", result: "已引用", channel: "飞书" },
  { id: "tr-10411", agent: "IT 自助助手", q: "给我管理员权限", latency: "0.9s", result: "已拒答", channel: "工作空间" },
];
