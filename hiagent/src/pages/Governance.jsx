export default function Governance() {
  return (
    <div>
      <div className="page-head">
        <div>
          <h2>治理与安全</h2>
          <p>对应官方的数据不出域、审计可追溯、大模型防火墙。私有化清单上这些是硬项，不是后期再补。</p>
        </div>
      </div>
      <div className="grid grid-2">
        <div className="card">
          <h3>权限</h3>
          <p className="hint">空间管理员 / 开发者 / 运营 / 普通员工。工具写入类 MCP 必须审批。</p>
        </div>
        <div className="card">
          <h3>审计</h3>
          <p className="hint">问题、命中知识、工具调用、模型、是否拒答，可导出。</p>
        </div>
        <div className="card">
          <h3>数据域</h3>
          <p className="hint">知识库与日志默认不出域。对象存储与向量库走内网。</p>
        </div>
        <div className="card">
          <h3>模型防火墙</h3>
          <p className="hint">拦截越权、凭据套取、承诺话术。与评测集里的拒答题对齐。</p>
        </div>
      </div>
    </div>
  );
}
