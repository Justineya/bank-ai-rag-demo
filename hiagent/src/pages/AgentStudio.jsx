import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { agents } from "../data.js";

const DEFAULT_PROMPT = `你是企业 IT 自助助手。只能根据知识库与工具结果回答。
- 咨询类：引用制度原文，标出资料来源。
- 权限类：核验部门/职级后走预审，不得直接提权。
- 故障类：初判严重度并创建工单。
资料没有的内容明确拒答，不要编造账号或权限。`;

export default function AgentStudio() {
  const { id } = useParams();
  const nav = useNavigate();
  const agent = agents.find((a) => a.id === id) || agents[0];
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [log, setLog] = useState([
    { role: "user", text: "VPN 连不上，还要开管理员权限。" },
    { role: "bot", text: "已按意图拆成两路：VPN 故障走知识库排障；管理员权限不能直接开通，将进入预审工作流。" },
  ]);
  const [draft, setDraft] = useState("");

  function send() {
    const q = draft.trim();
    if (!q) return;
    setLog((rows) => [
      ...rows,
      { role: "user", text: q },
      { role: "bot", text: "（骨架）此处将调用 HiAgent / 方舟运行时。当前仅演示调试台布局。" },
    ]);
    setDraft("");
  }

  return (
    <div>
      <div className="page-head">
        <div>
          <h2>{agent.name} · 工作室</h2>
          <p>低代码搭建面：左侧组件，中间策略，右侧调试。正式环境再接到提示词、RAG、MCP 与评测。</p>
        </div>
        <div>
          <button className="btn ghost" onClick={() => nav("/eval")}>去评测</button>{" "}
          <button className="btn" onClick={() => nav("/publish")}>发布</button>
        </div>
      </div>
      <div className="studio">
        <div className="panel">
          <h3>组件</h3>
          <div className="comp">提示词模板 · IT 自助</div>
          <div className="comp">知识库 · IT 运维手册</div>
          <div className="comp">插件 · ITSM 建单</div>
          <div className="comp">MCP · 通讯录</div>
          <div className="comp">工作流 · 权限预审</div>
          <div className="comp">兜底 · 人工接管</div>
        </div>
        <div className="panel">
          <h3>系统提示词</h3>
          <textarea className="prompt" value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          <div className="hint">模型 {agent.model} · 温度 0.2 · 可在发布页绑定渠道</div>
        </div>
        <div className="panel">
          <h3>调试台</h3>
          <div className="chat">
            {log.map((m, i) => (
              <div key={i} className={`bubble ${m.role === "user" ? "user" : ""}`}>{m.text}</div>
            ))}
          </div>
          <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && send()}
              placeholder="输入一句试跑"
              style={{ flex: 1, border: "1px solid var(--line)", borderRadius: 8, padding: "8px 10px" }}
            />
            <button className="btn small" onClick={send}>发送</button>
          </div>
        </div>
      </div>
    </div>
  );
}
