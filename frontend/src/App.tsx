import { Route, Routes } from "react-router-dom";
import { SideNav } from "./components/SideNav";
import { DashboardPage } from "./pages/DashboardPage";
import { IngestPage } from "./pages/IngestPage";
import { MemoryPage } from "./pages/MemoryPage";
import { SettingsPage } from "./pages/SettingsPage";

function App() {
  return (
    <div className="app-shell">
      <SideNav />
      <main className="app-main">
        <header className="top-strip" aria-label="页面头部">
          <p>Agent-native interface for research pipeline visibility</p>
        </header>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/memory" element={<MemoryPage />} />
          <Route path="/ingest" element={<IngestPage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
