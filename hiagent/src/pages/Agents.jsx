import { useNavigate } from "react-router-dom";
import { agents } from "../data.js";

export default function Agents() {
  const nav = useNavigate();
  return (
    <div>
      <div className="page-head">
        <div>
          <h2>智能体纳管</h2>
          <p>接入 HiAgent 自建、火山引擎智能体和三方智能体。点一行进入工作室（提示词 / 知识库 / 插件 / 工作流）。</p>
        </div>
        <div>
          <button className="btn ghost" onClick={() => nav("/agents/it-desk")}>从模板创建</button>{" "}
          <button className="btn" onClick={() => nav("/agents/ticket")}>空白智能体</button>
        </div>
      </div>
      <div className="card">
        <table className="table">
          <thead>
            <tr>
              <th>名称</th><th>来源</th><th>场景</th><th>模型</th><th>调用</th><th>解决率</th><th>状态</th>
            </tr>
          </thead>
          <tbody>
            {agents.map((a) => (
              <tr key={a.id} style={{ cursor: "pointer" }} onClick={() => nav(`/agents/${a.id}`)}>
                <td>{a.name}</td>
                <td>{a.kind}</td>
                <td>{a.scene}</td>
                <td>{a.model}</td>
                <td>{a.calls.toLocaleString()}</td>
                <td>{a.resolve}</td>
                <td><span className={`badge ${a.status === "已发布" ? "ok" : a.status === "草稿" ? "warn" : "blue"}`}>{a.status}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
