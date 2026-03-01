import type { WorkflowNode } from "../store/useConsoleStore";

interface PulseRailProps {
  nodes: WorkflowNode[];
}

export function PulseRail({ nodes }: PulseRailProps) {
  return (
    <section className="panel panel--rail" aria-label="工作流状态">
      <header className="panel-head">
        <p className="panel-kicker">Pipeline</p>
        <h2>LangGraph Pulse Rail</h2>
      </header>
      <ol className="pulse-rail">
        {nodes.map((node, index) => (
          <li key={node.id} className={`rail-item rail-item--${node.status}`}>
            <span className="rail-index" aria-hidden="true">
              {String(index + 1).padStart(2, "0")}
            </span>
            <span className="rail-dot" aria-hidden="true" />
            <div className="rail-content">
              <p className="rail-label">{node.label}</p>
              <p className="rail-id">
                {node.id}
                {typeof node.durationMs === "number" ? ` · ${node.durationMs}ms` : ""}
              </p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
