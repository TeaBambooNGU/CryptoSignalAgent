import { NavLink } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Console" },
  { to: "/memory", label: "Memory" },
  { to: "/ingest", label: "Ingest" },
  { to: "/settings", label: "Settings" },
];

export function SideNav() {
  return (
    <aside className="side-nav" aria-label="主导航">
      <div className="brand-mark">
        <span className="brand-glyph" aria-hidden="true">
          {"/*"}
        </span>
        <div>
          <p className="brand-title">Crypto Signal Agent</p>
          <p className="brand-subtitle">Cold Relay Console</p>
        </div>
      </div>

      <nav>
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              isActive ? "nav-item nav-item--active" : "nav-item"
            }
            end={item.to === "/"}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="nav-footer">
        <p className="mono-line">V1 · MCP-first</p>
        <p className="mono-line">LangGraph x Milvus</p>
      </div>
    </aside>
  );
}
