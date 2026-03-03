import { useEffect, useState } from "react";
import type { WorkflowNode } from "../store/useConsoleStore";

interface PulseRailProps {
  nodes: WorkflowNode[];
  isPending: boolean;
}

export function PulseRail({ nodes, isPending }: PulseRailProps) {
  const [flowIndex, setFlowIndex] = useState(0);

  useEffect(() => {
    if (!isPending || nodes.length === 0) return;
    const timer = window.setInterval(() => {
      setFlowIndex((value) => (value + 1) % nodes.length);
    }, 420);
    return () => window.clearInterval(timer);
  }, [isPending, nodes.length]);

  return (
    <section className={`panel panel--rail ${isPending ? "panel--rail--pending" : ""}`} aria-label="工作流状态">
      <header className="panel-head">
        <p className="panel-kicker">Pipeline</p>
        <h2>LangGraph Pulse Rail</h2>
      </header>

      <div className={`pulse-strip ${isPending ? "pulse-strip--live" : ""}`}>
        <ol className="pulse-rail pulse-rail--horizontal">
          {nodes.map((node, index) => {
            const pendingStatus = isPending
              ? index === flowIndex
                ? "running"
                : node.status === "success" || node.status === "error"
                  ? node.status
                  : "idle"
              : node.status;
            return (
              <li
                key={node.id}
                className={`rail-item rail-item--${pendingStatus} ${isPending && index === flowIndex ? "rail-item--flow" : ""}`}
              >
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
            );
          })}
        </ol>
      </div>

      <p className="helper-line">{isPending ? "AI 正在推理，电流沿节点流转中..." : "流程已停止流转，展示当前真实节点状态。"}</p>
    </section>
  );
}
