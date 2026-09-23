import { knowledgeBases } from "../data.js";

export default function Knowledge() {
  return (
    <div>
      <div className="page-head">
        <div>
          <h2>知识库</h2>
          <p>导入制度/FAQ，分段训练后给智能体召回。正式 HiAgent 支持批量导入与召回阈值；这里先把运营字段摆出来。</p>
        </div>
        <button className="btn">批量导入</button>
      </div>
      <div className="grid grid-3">
        {knowledgeBases.map((kb) => (
          <div className="card" key={kb.name}>
            <div className={`badge ${kb.status === "已训练" ? "ok" : "warn"}`}>{kb.status}</div>
            <h3 style={{ marginTop: 10 }}>{kb.name}</h3>
            <p className="hint">{kb.docs} 篇文档 · {kb.chunks} 分段 · 召回阈值 {kb.recall}</p>
            <button className="btn ghost small">查看分段</button>
          </div>
        ))}
      </div>
    </div>
  );
}
