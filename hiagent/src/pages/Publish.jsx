const channels = [
  { name: "工作空间 / Canvas", type: "门户", status: "已开通" },
  { name: "飞书机器人", type: "IM", status: "配置中" },
  { name: "企业微信", type: "IM", status: "未配置" },
  { name: "钉钉", type: "IM", status: "未配置" },
  { name: "OpenAPI", type: "集成", status: "已开通" },
  { name: "WebSDK", type: "嵌入", status: "已开通" },
];

export default function Publish() {
  return (
    <div>
      <div className="page-head">
        <div>
          <h2>发布</h2>
          <p>官方支持飞书、钉钉、微信及 API / WebSDK。部署前先决定第一期只开工作空间还是连 IM。</p>
        </div>
      </div>
      <div className="grid grid-3">
        {channels.map((c) => (
          <div className="card" key={c.name}>
            <div className={`badge ${c.status === "已开通" ? "ok" : c.status === "配置中" ? "warn" : ""}`}>{c.status}</div>
            <h3 style={{ marginTop: 10 }}>{c.name}</h3>
            <p className="hint">{c.type}</p>
            <button className="btn ghost small">查看接入说明</button>
          </div>
        ))}
      </div>
    </div>
  );
}
