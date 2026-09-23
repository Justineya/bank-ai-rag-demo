import { traces } from "../data.js";

export default function Observe() {
  return (
    <div>
      <div className="page-head">
        <div>
          <h2>观测</h2>
          <p>监控性能、效果和用户反馈，沉淀生产数据反哺评测集。这里先列出调用轨迹字段。</p>
        </div>
      </div>
      <div className="grid grid-3" style={{ marginBottom: 16 }}>
        <div className="card"><div className="metric">1.9s <span>P50 延迟</span></div></div>
        <div className="card"><div className="metric">4.2% <span>转人工</span></div></div>
        <div className="card"><div className="metric">12 <span>今日拒答</span></div></div>
      </div>
      <div className="card">
        <table className="table">
          <thead><tr><th>Trace</th><th>智能体</th><th>问题</th><th>延迟</th><th>结果</th><th>渠道</th></tr></thead>
          <tbody>
            {traces.map((t) => (
              <tr key={t.id}>
                <td>{t.id}</td><td>{t.agent}</td><td>{t.q}</td><td>{t.latency}</td>
                <td><span className={`badge ${t.result === "已拒答" ? "warn" : "ok"}`}>{t.result}</span></td>
                <td>{t.channel}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
