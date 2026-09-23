export default function Intents() {
  const rows = [
    { scene: "IT 咨询", intent: "排障 / 账号 / VPN", agent: "IT 自助助手", chain: "意图 → 知识库 → 答复" },
    { scene: "权限申请", intent: "开通 / 扩权 / 临时权限", agent: "工单预审员", chain: "意图 → 核验 → OA" },
    { scene: "制度问答", intent: "报销 / 差旅 / 合规", agent: "制度问答助手", chain: "意图 → RAG → 拒答兜底" },
  ];
  return (
    <div>
      <div className="page-head">
        <div>
          <h2>场景与意图</h2>
          <p>运营意图描述和多智能体调度思维链，用来优化识别与协同。正式环境可 A/B 路由策略。</p>
        </div>
      </div>
      <div className="card">
        <table className="table">
          <thead><tr><th>业务场景</th><th>意图</th><th>默认智能体</th><th>调度链</th></tr></thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.scene}>
                <td>{r.scene}</td><td>{r.intent}</td><td>{r.agent}</td><td>{r.chain}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
