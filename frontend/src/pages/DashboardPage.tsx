import { useMutation } from "@tanstack/react-query";
import { Fragment, type ReactNode, useMemo } from "react";
import { queryResearch } from "../api/client";
import type { WorkflowStep } from "../api/types";
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

function toWorkflowFromSteps(steps: WorkflowStep[]): WorkflowNode[] {
  const stepMap = new Map(steps.map((step) => [step.node_id, step]));
  return BASE_WORKFLOW.map((node) => {
    const step = stepMap.get(node.id);
    if (!step) {
      return { ...node, status: "idle", durationMs: undefined };
    }
    return {
      ...node,
      status: step.status === "error" ? "error" : "success",
      durationMs: step.duration_ms,
    };
  });
}

function splitThinkAndReport(raw: string): { think: string; report: string } {
  const match = raw.match(/<think>([\s\S]*?)<\/think>/i);
  if (!match) {
    return { think: "", report: raw.trim() };
  }
  const think = match[1]?.trim() ?? "";
  const report = raw.replace(match[0], "").trim();
  return { think, report };
}

function isTableSeparator(line: string): boolean {
  const cells = line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

function parseTableRow(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function renderInline(text: string): ReactNode[] {
  const matches = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).filter(Boolean);
  return matches.map((part, index) => {
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={`inline-${index}`}>{part.slice(1, -1)}</code>;
    }
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={`inline-${index}`}>{part.slice(2, -2)}</strong>;
    }
    return <Fragment key={`inline-${index}`}>{part}</Fragment>;
  });
}

function renderMarkdown(report: string): ReactNode[] {
  const lines = report.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let index = 0;
  let key = 0;

  const push = (node: ReactNode) => {
    blocks.push(<Fragment key={`md-${key++}`}>{node}</Fragment>);
  };

  while (index < lines.length) {
    const line = lines[index] ?? "";
    const trimmed = line.trim();
    if (!trimmed) {
      index += 1;
      continue;
    }

    if (trimmed.startsWith("```")) {
      const language = trimmed.slice(3).trim();
      const codeLines: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith("```")) {
        codeLines.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) {
        index += 1;
      }
      push(
        <pre className="report-code">
          <code className={language ? `language-${language}` : undefined}>{codeLines.join("\n")}</code>
        </pre>,
      );
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const text = headingMatch[2].trim();
      if (level === 1) push(<h1>{renderInline(text)}</h1>);
      if (level === 2) push(<h2>{renderInline(text)}</h2>);
      if (level === 3) push(<h3>{renderInline(text)}</h3>);
      if (level === 4) push(<h4>{renderInline(text)}</h4>);
      if (level === 5) push(<h5>{renderInline(text)}</h5>);
      if (level === 6) push(<h6>{renderInline(text)}</h6>);
      index += 1;
      continue;
    }

    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      push(<hr />);
      index += 1;
      continue;
    }

    if (trimmed.includes("|") && index + 1 < lines.length && isTableSeparator(lines[index + 1])) {
      const headers = parseTableRow(lines[index]);
      const rows: string[][] = [];
      index += 2;
      while (index < lines.length) {
        const rowLine = lines[index].trim();
        if (!rowLine || !rowLine.includes("|")) {
          break;
        }
        rows.push(parseTableRow(lines[index]));
        index += 1;
      }
      push(
        <div className="report-table-wrap">
          <table className="report-table">
            <thead>
              <tr>
                {headers.map((cell, cellIndex) => (
                  <th key={`th-${cellIndex}`}>{renderInline(cell)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={`tr-${rowIndex}`}>
                  {row.map((cell, cellIndex) => (
                    <td key={`td-${rowIndex}-${cellIndex}`}>{renderInline(cell)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed) || /^[-*+]\s+/.test(trimmed)) {
      const ordered = /^\d+\.\s+/.test(trimmed);
      const listItems: string[] = [];
      while (index < lines.length) {
        const itemLine = lines[index].trim();
        if (ordered && !/^\d+\.\s+/.test(itemLine)) break;
        if (!ordered && !/^[-*+]\s+/.test(itemLine)) break;
        listItems.push(itemLine.replace(ordered ? /^\d+\.\s+/ : /^[-*+]\s+/, "").trim());
        index += 1;
      }
      if (ordered) {
        push(
          <ol>
            {listItems.map((item, itemIndex) => (
              <li key={`ol-${itemIndex}`}>{renderInline(item)}</li>
            ))}
          </ol>,
        );
      } else {
        push(
          <ul>
            {listItems.map((item, itemIndex) => (
              <li key={`ul-${itemIndex}`}>{renderInline(item)}</li>
            ))}
          </ul>,
        );
      }
      continue;
    }

    if (/^>\s?/.test(trimmed)) {
      const quoteLines: string[] = [];
      while (index < lines.length && /^>\s?/.test(lines[index].trim())) {
        quoteLines.push(lines[index].trim().replace(/^>\s?/, ""));
        index += 1;
      }
      push(<blockquote>{renderInline(quoteLines.join(" "))}</blockquote>);
      continue;
    }

    const paragraphLines: string[] = [];
    while (index < lines.length) {
      const paragraphLine = lines[index].trim();
      if (!paragraphLine) break;
      if (
        paragraphLine.startsWith("```") ||
        /^(#{1,6})\s+/.test(paragraphLine) ||
        /^(-{3,}|\*{3,}|_{3,})$/.test(paragraphLine) ||
        /^\d+\.\s+/.test(paragraphLine) ||
        /^[-*+]\s+/.test(paragraphLine) ||
        /^>\s?/.test(paragraphLine) ||
        (paragraphLine.includes("|") && index + 1 < lines.length && isTableSeparator(lines[index + 1]))
      ) {
        break;
      }
      paragraphLines.push(paragraphLine);
      index += 1;
    }
    push(<p>{renderInline(paragraphLines.join(" "))}</p>);
  }

  return blocks;
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

  const queryMutation = useMutation({
    mutationFn: queryResearch,
    onSuccess: (response) => {
      setWorkflowNodes(toWorkflowFromSteps(response.workflow_steps));
      setResult(response);
      setErrorText(response.errors.join("；"));
    },
    onError: (error) => {
      setWorkflowNodes(BASE_WORKFLOW.map((node) => ({ ...node, status: "idle", durationMs: undefined })));
      const message = error instanceof Error ? error.message : "请求失败";
      setErrorText(message);
    },
  });

  const taskContextPreview = useMemo(
    () => parseTaskContext(taskContextText, symbolsText),
    [taskContextText, symbolsText],
  );
  const reportParts = useMemo(() => splitThinkAndReport(result?.report ?? ""), [result?.report]);
  const renderedReport = useMemo(() => renderMarkdown(reportParts.report), [reportParts.report]);

  const handleSubmit = () => {
    setResult(null);
    setErrorText("");
    setWorkflowNodes(BASE_WORKFLOW.map((node) => ({ ...node, status: "idle", durationMs: undefined })));

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
            {reportParts.think ? (
              <details className="think-block">
                <summary>LLM Think（推理草稿）</summary>
                <pre>{reportParts.think}</pre>
              </details>
            ) : null}
            <article className="report-body">{renderedReport}</article>

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
          <p className="empty-state">
            {queryMutation.isPending ? "后端正在执行 9 节点流程，完成后会回填真实状态与研报。" : "提交查询后将在这里展示报告、引用和 trace。"}
          </p>
        )}

        {errorText ? <p className="error-text">{errorText}</p> : null}
      </div>
    </section>
  );
}
