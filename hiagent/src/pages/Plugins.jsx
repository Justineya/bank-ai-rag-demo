import { plugins } from "../data.js";

export default function Plugins() {
  return (
    <div>
      <div className="page-head">
        <div>
          <h2>插件与 MCP Gateway</h2>
          <p>打通 ITSM / OA / 文档等企业系统。MCP 接入默认走审批，避免工具一接上就能写生产。</p>
        </div>
        <button className="btn">注册 MCP Server</button>
      </div>
      <div className="card">
        <table className="table">
          <thead><tr><th>名称</th><th>类型</th><th>范围</th><th>状态</th></tr></thead>
          <tbody>
            {plugins.map((p) => (
              <tr key={p.name}>
                <td>{p.name}</td>
                <td>{p.type}</td>
                <td>{p.scope}</td>
                <td><span className={`badge ${p.status === "已启用" ? "ok" : "warn"}`}>{p.status}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
