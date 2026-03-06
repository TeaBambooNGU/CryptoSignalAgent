import { Route, Routes } from "react-router-dom";
import { SideNav } from "./components/SideNav";
import { DashboardPage } from "./pages/DashboardPage";
import { KnowledgePage } from "./pages/KnowledgePage";
import { MemoryPage } from "./pages/MemoryPage";
import { SettingsPage } from "./pages/SettingsPage";

function App() {
  return (
    <div className="app-shell">
      <SideNav />
      <main className="app-main">
        <header className="top-strip" aria-label="页面头部">
          <p>Conversation-first interface for report chat, rewrite, and regenerate</p>
        </header>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/memory" element={<MemoryPage />} />
          <Route path="/knowledge" element={<KnowledgePage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
