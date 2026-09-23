import { NavLink, Outlet, useLocation } from "react-router-dom";

const NAV = [
  { group: "消费", items: [{ to: "/", label: "工作空间" }] },
  {
    group: "开发",
    items: [
      { to: "/agents", label: "智能体" },
      { to: "/knowledge", label: "知识库" },
      { to: "/plugins", label: "插件与 MCP" },
      { to: "/workflows", label: "工作流" },
      { to: "/intents", label: "场景意图" },
    ],
  },
  {
    group: "运行 / 治理",
    items: [
      { to: "/eval", label: "评测" },
      { to: "/observe", label: "观测" },
      { to: "/publish", label: "发布" },
      { to: "/governance", label: "治理与安全" },
    ],
  },
];

const TITLES = {
  "/": "工作空间",
  "/agents": "智能体纳管",
  "/knowledge": "知识库",
  "/plugins": "插件与 MCP Gateway",
  "/workflows": "工作流编排",
  "/intents": "场景与意图",
  "/eval": "评测",
  "/observe": "观测",
  "/publish": "发布渠道",
  "/governance": "治理与安全",
};

export default function Shell() {
  const { pathname } = useLocation();
  const title = pathname.startsWith("/agents/") ? "智能体工作室" : TITLES[pathname] || "HiAgent";

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">Hi</div>
          <div>
            <h1>HiAgent</h1>
            <p>控制台骨架 · 非官方包</p>
          </div>
        </div>
        {NAV.map((block) => (
          <div key={block.group}>
            <div className="nav-label">{block.group}</div>
            {block.items.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === "/"}
                className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}
              >
                {item.label}
              </NavLink>
            ))}
          </div>
        ))}
        <div className="side-foot">私有化部署预演 · 本地 mock 数据</div>
      </aside>
      <div className="main">
        <header className="topbar">
          <div className="crumb">
            数字员工中台 / <strong>{title}</strong>
          </div>
          <div className="top-actions">
            <span className="pill">环境：预演 / 未接火山 API</span>
            <span className="pill">空间：默认企业</span>
            <div className="avatar">管</div>
          </div>
        </header>
        <div className="content">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
