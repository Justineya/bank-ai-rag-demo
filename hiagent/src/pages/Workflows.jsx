export default function Workflows() {
  return (
    <div>
      <div className="page-head">
        <div>
          <h2>工作流编排</h2>
          <p>IT 自助三条主干：咨询走知识库、权限走预审、故障建工单。正式产品是可拖拽 DAG；本页用固定节点标出要落地的分支。</p>
        </div>
        <button className="btn ghost">仿真测试</button>
      </div>
      <div className="flow">
        <svg style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
          <line x1="140" y1="70" x2="300" y2="70" stroke="#9db0c7" />
          <line x1="460" y1="70" x2="620" y2="70" stroke="#9db0c7" />
          <line x1="700" y1="110" x2="700" y2="190" stroke="#9db0c7" />
          <line x1="540" y1="230" x2="380" y2="300" stroke="#9db0c7" />
          <line x1="700" y1="230" x2="700" y2="300" stroke="#9db0c7" />
          <line x1="860" y1="230" x2="860" y2="300" stroke="#9db0c7" />
        </svg>
        <div className="node start" style={{ left: 40, top: 40 }}><b>开始 · 用户请求</b>渠道：工作空间 / 飞书</div>
        <div className="node llm" style={{ left: 300, top: 40 }}><b>意图识别</b>咨询 / 权限 / 故障</div>
        <div className="node if" style={{ left: 620, top: 40 }}><b>分支</b>三条主干 + 兜底</div>
        <div className="node kb" style={{ left: 300, top: 280 }}><b>咨询 · 知识库</b>召回制度并引用</div>
        <div className="node tool" style={{ left: 620, top: 280 }}><b>权限 · 预审</b>核验职级 → OA</div>
        <div className="node tool" style={{ left: 780, top: 280 }}><b>故障 · ITSM</b>建单并回写进度</div>
        <div className="node end" style={{ left: 540, top: 400 }}><b>结束 / 人工接管</b>无法识别则升级</div>
      </div>
    </div>
  );
}
