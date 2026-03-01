const runtimeItems = [
  { key: "API Base", value: import.meta.env.VITE_API_BASE || "(same-origin/proxy)" },
  { key: "Transport", value: "MCP streamable_http / stdio / sse" },
  { key: "Pipeline", value: "9-node LangGraph workflow" },
  { key: "Output", value: "report + citations + trace_id + errors + workflow_steps" },
];

const designItems = [
  { key: "Aesthetic", value: "Cold Relay Minimalism" },
  { key: "Display Font", value: "Space Grotesk" },
  { key: "Body Font", value: "Noto Sans SC" },
  { key: "Anchor", value: "Workflow Pulse Rail" },
];

export function SettingsPage() {
  return (
    <section className="stack-layout">
      <div className="panel panel--single">
        <header className="panel-head">
          <p className="panel-kicker">System</p>
          <h2>Runtime Snapshot</h2>
        </header>
        <dl className="kv-list">
          {runtimeItems.map((item) => (
            <div key={item.key}>
              <dt>{item.key}</dt>
              <dd>{item.value}</dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="panel panel--single">
        <header className="panel-head">
          <p className="panel-kicker">Design</p>
          <h2>Visual Tokens</h2>
        </header>
        <dl className="kv-list">
          {designItems.map((item) => (
            <div key={item.key}>
              <dt>{item.key}</dt>
              <dd>{item.value}</dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}
