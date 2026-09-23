import { Navigate, Route, Routes } from "react-router-dom";
import Shell from "./layout/Shell.jsx";
import Workspace from "./pages/Workspace.jsx";
import Agents from "./pages/Agents.jsx";
import AgentStudio from "./pages/AgentStudio.jsx";
import Knowledge from "./pages/Knowledge.jsx";
import Plugins from "./pages/Plugins.jsx";
import Workflows from "./pages/Workflows.jsx";
import Intents from "./pages/Intents.jsx";
import EvalPage from "./pages/Eval.jsx";
import Observe from "./pages/Observe.jsx";
import Publish from "./pages/Publish.jsx";
import Governance from "./pages/Governance.jsx";

export default function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route path="/" element={<Workspace />} />
        <Route path="/agents" element={<Agents />} />
        <Route path="/agents/:id" element={<AgentStudio />} />
        <Route path="/knowledge" element={<Knowledge />} />
        <Route path="/plugins" element={<Plugins />} />
        <Route path="/workflows" element={<Workflows />} />
        <Route path="/intents" element={<Intents />} />
        <Route path="/eval" element={<EvalPage />} />
        <Route path="/observe" element={<Observe />} />
        <Route path="/publish" element={<Publish />} />
        <Route path="/governance" element={<Governance />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
