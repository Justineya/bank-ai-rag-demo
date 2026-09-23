import { useNavigate } from "react-router-dom";
import { agents } from "../data.js";

export default function Workspace() {
  const nav = useNavigate();
  return (
    <div>
      <div className="page-head">
        <div>
          <h2>工作空间</h2>
          <p>
            对应 HiAgent 的 Canvas 门户：员工在这里和已发布的数字员工协同。正式产品里是千人千面画布；这里先用卡片入口把信息架构跑通。
          </p>
        </div>
        <button className="btn" onClick={() => nav("/agents")}>派遣智能体</button>
      </div>
      <div className="notice">骨架页。对话、权限和渠道路由都是前端示意，方便评审部署范围。</div>
      <div className="grid grid-4" style={{ marginBottom: 16 }}>
        <div className="card"><div className="metric">4 <span>已纳管</span></div><div className="hint">含自建 / 火山 / 三方</div></div>
        <div className="card"><div className="metric">18.1k <span>本周调用</span></div><div className="hint">工作空间 + IM 渠道</div></div>
        <div className="card"><div className="metric">76% <span>自动办结</span></div><div className="hint">IT 场景目标 ≥ 70%</div></div>
        <div className="card"><div className="metric">3 <span>待治理事项</span></div><div className="hint">MCP 待审批 / 越权拒答</div></div>
      </div>
      <div className="canvas">
        {agents.map((a) => (
          <div key={a.id} className="tile" onClick={() => nav(`/agents/${a.id}`)}>
            <div className="badge blue">{a.kind}</div>
            <h3 style={{ margin: "10px 0 0" }}>{a.name}</h3>
            <div className="meta">{a.scene}</div>
            <div className="meta">状态 {a.status} · 解决率 {a.resolve}</div>
          </div>
        ))}
        <div className="tile" onClick={() => nav("/intents")}>
          <div className="badge">调度</div>
          <h3 style={{ margin: "10px 0 0" }}>意图路由器</h3>
          <div className="meta">按场景把请求分给对应智能体，并保留人工接管。</div>
        </div>
      </div>
    </div>
  );
}
