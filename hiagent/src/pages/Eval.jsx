import { evalRows } from "../data.js";

export default function EvalPage() {
  return (
    <div>
      <div className="page-head">
        <div>
          <h2>评测</h2>
          <p>多维度看命中、拒答、引用。数字是示意，用来对齐上线前必须有的回归集，而不是聊天感觉。</p>
        </div>
        <button className="btn">跑评测集</button>
      </div>
      <div className="card">
        <table className="table">
          <thead><tr><th>集合</th><th>命中</th><th>拒答正确</th><th>引用</th><th>及格</th></tr></thead>
          <tbody>
            {evalRows.map((r) => (
              <tr key={r.set}>
                <td>{r.set}</td><td>{r.hit}</td><td>{r.refuse}</td><td>{r.cite}</td><td>{r.pass}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
