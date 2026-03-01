import { useMutation } from "@tanstack/react-query";
import { useEffect, useMemo, useRef } from "react";
import { queryResearch } from "../api/client";
import { PulseRail } from "../components/PulseRail";
import { BASE_WORKFLOW, type WorkflowNode, useConsoleStore } from "../store/useConsoleStore";

function parseTaskContext(raw: string, symbolsText: string): Record<string, unknown> {
  const symbols = symbolsText
    .split(",")
    .map((token) => token.trim())
    .filter(Boolean);

  if (!raw.trim()) {
    return symbols.length ? { symbols } : {};
  }

  try {
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    if (symbols.length && parsed.symbols === undefined) {
      parsed.symbols = symbols;
    }
    return parsed;
  } catch {
    return symbols.length ? { symbols } : {};
  }
}

function toRunningState(step: number): WorkflowNode[] {
  return BASE_WORKFLOW.map((node, index) => {
    if (index < step) return { ...node, status: "success" };
    if (index === step) return { ...node, status: "running" };
    return { ...node, status: "idle" };
  });
}

export function DashboardPage() {
  const {
    userId,
    query,
    symbolsText,
    taskContextText,
    workflowNodes,
    result,
    errorText,
    setUserId,
    setQuery,
    setSymbolsText,
    setTaskContextText,
    setWorkflowNodes,
    setResult,
    setErrorText,
  } = useConsoleStore();

  const intervalRef = useRef<number | null>(null);

  const queryMutation = useMutation({
    mutationFn: queryResearch,
    onSuccess: (response) => {
      setWorkflowNodes(BASE_WORKFLOW.map((node) => ({ ...node, status: "success" })));
      setResult(response);
      setErrorText(response.errors.join("；"));
      if (intervalRef.current !== null) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    },
    onError: (error) => {
      setWorkflowNodes(
        BASE_WORKFLOW.map((node, index) => {
          if (index < 6) return { ...node, status: "success" };
          if (index === 6) return { ...node, status: "error" };
          return { ...node, status: "idle" };
        }),
      );
      const message = error instanceof Error ? error.message : "请求失败";
      setErrorText(message);
      if (intervalRef.current !== null) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    },
  });

  useEffect(() => {
    return () => {
      if (intervalRef.current !== null) {
        window.clearInterval(intervalRef.current);
      }
    };
  }, []);

  const taskContextPreview = useMemo(
    () => parseTaskContext(taskContextText, symbolsText),
    [taskContextText, symbolsText],
  );

  const handleSubmit = () => {
    setResult(null);
    setErrorText("");
    setWorkflowNodes(toRunningState(0));

    let step = 0;
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
    }
    intervalRef.current = window.setInterval(() => {
      step = Math.min(step + 1, BASE_WORKFLOW.length - 1);
      setWorkflowNodes(toRunningState(step));
    }, 860);

    queryMutation.mutate({
      user_id: userId,
      query,
      task_context: taskContextPreview,
    });
  };

  return (
    <section className="dashboard-shell">
      <div className="panel panel--query">
        <header className="panel-head">
          <p className="panel-kicker">Input</p>
          <h2>Research Query Composer</h2>
        </header>

        <label className="field">
          <span>User ID</span>
          <input value={userId} onChange={(e) => setUserId(e.target.value)} placeholder="u001" />
        </label>

        <label className="field">
          <span>Query</span>
          <textarea
            rows={6}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="请描述你想要的研报任务"
          />
        </label>

        <label className="field">
          <span>Symbols</span>
          <input
            value={symbolsText}
            onChange={(e) => setSymbolsText(e.target.value)}
            placeholder="BTC,ETH,SOL"
          />
        </label>

        <label className="field">
          <span>Task Context (JSON)</span>
          <textarea
            rows={4}
            value={taskContextText}
            onChange={(e) => setTaskContextText(e.target.value)}
            placeholder='{"risk_mode":"balanced"}'
          />
        </label>

        <button
          type="button"
          className="action-button"
          onClick={handleSubmit}
          disabled={queryMutation.isPending || !userId.trim() || !query.trim()}
        >
          {queryMutation.isPending ? "Generating..." : "Run Agent"}
        </button>

        <p className="helper-line">
          task_context 预览：<code>{JSON.stringify(taskContextPreview)}</code>
        </p>
      </div>

      <PulseRail nodes={workflowNodes} />

      <div className="panel panel--report">
        <header className="panel-head">
          <p className="panel-kicker">Output</p>
          <h2>Report Viewer</h2>
        </header>

        {result ? (
          <>
            <p className="trace-line">trace_id: {result.trace_id}</p>
            <article className="report-body">{result.report}</article>

            <section className="citation-block">
              <h3>Citations ({result.citations.length})</h3>
              <ul>
                {result.citations.map((citation, idx) => (
                  <li key={`${citation.source}-${idx}`}>
                    <p className="citation-source">{citation.source}</p>
                    <a href={citation.raw_ref} target="_blank" rel="noreferrer">
                      {citation.raw_ref}
                    </a>
                    <p>{citation.snippet}</p>
                  </li>
                ))}
              </ul>
            </section>
          </>
        ) : (
          <p className="empty-state">提交查询后将在这里展示报告、引用和 trace。</p>
        )}

        {errorText ? <p className="error-text">{errorText}</p> : null}
      </div>
    </section>
  );
}
