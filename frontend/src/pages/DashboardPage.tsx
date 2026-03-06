import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Fragment, type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  getConversationMeta,
  getConversationReport,
  listConversationReports,
  listConversationTurns,
  sendConversationMessage,
} from "../api/client";
import type {
  ConversationAction,
  ConversationReport,
  ConversationTurnSummary,
  WorkflowStep,
} from "../api/types";
import { PulseRail } from "../components/PulseRail";
import { BASE_WORKFLOW, type WorkflowNode, useConsoleStore } from "../store/useConsoleStore";

const TURN_PAGE_SIZE = 12;
const REPORT_PAGE_SIZE = 10;

const WORKFLOW_LABELS: Record<string, string> = {
  load_user_memory: "加载记忆",
  resolve_symbols: "解析标的",
  collect_signals_via_mcp: "采集信号",
  normalize_and_index: "标准化入库",
  retrieve_knowledge_evidence: "检索知识证据",
  analyze_signals: "分析信号",
  generate_report: "生成研报",
  persist_memory: "写回记忆",
  finalize_response: "输出结果",
  chat_reply: "对话回复",
  rewrite_report: "改写报告",
};

function normalizeConversationId(rawConversationId: string, userId: string): string {
  const trimmedConversationId = rawConversationId.trim();
  if (trimmedConversationId) return trimmedConversationId;
  const trimmedUserId = userId.trim() || "u001";
  return `default:${trimmedUserId}`;
}

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
  const baseNodeIds = new Set(BASE_WORKFLOW.map((node) => node.id));

  const baseNodes: WorkflowNode[] = BASE_WORKFLOW.map((node): WorkflowNode => {
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

  const dynamicNodes: WorkflowNode[] = steps
    .filter((step) => !baseNodeIds.has(step.node_id))
    .map((step): WorkflowNode => ({
      id: step.node_id,
      label: WORKFLOW_LABELS[step.node_id] || step.node_id,
      status: step.status === "error" ? "error" : "success",
      durationMs: step.duration_ms,
    }));

  return [...baseNodes, ...dynamicNodes];
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

function formatTime(unixSeconds: number): string {
  if (!unixSeconds) return "-";
  return new Date(unixSeconds * 1000).toLocaleString("zh-CN", { hour12: false });
}

function getActionText(action: ConversationAction): string {
  if (action === "chat") return "对话";
  if (action === "rewrite_report") return "改写";
  if (action === "regenerate_report") return "重跑";
  return "自动";
}

function getStatusReport(
  selectedReport: ConversationReport | null,
  queriedReport: ConversationReport | undefined,
  latestReport: ConversationReport | undefined,
  responseReport: ConversationReport | null | undefined,
): ConversationReport | null {
  if (queriedReport) return queriedReport;
  if (selectedReport) return selectedReport;
  if (responseReport) return responseReport;
  if (latestReport) return latestReport;
  return null;
}

interface ConflictHint {
  kind: "conversation_version_conflict" | "request_in_flight";
  expectedVersion?: number;
  currentVersion?: number;
  message: string;
}

interface BranchTreeRow {
  turn: ConversationTurnSummary;
  depth: number;
  childCount: number;
  hasDetachedParent: boolean;
}

function buildBranchTreeRows(turns: ConversationTurnSummary[]): BranchTreeRow[] {
  if (!turns.length) return [];
  const turnsByVersion = [...turns].sort((left, right) => left.version - right.version);
  const turnById = new Map(turnsByVersion.map((turn) => [turn.turn_id, turn]));
  const childrenByParentId = new Map<string, string[]>();
  const roots: ConversationTurnSummary[] = [];
  const detachedParentSet = new Set<string>();

  for (const turn of turnsByVersion) {
    const parentId = (turn.parent_turn_id || "").trim();
    if (!parentId) {
      roots.push(turn);
      continue;
    }
    if (!turnById.has(parentId)) {
      roots.push(turn);
      detachedParentSet.add(turn.turn_id);
      continue;
    }
    const siblings = childrenByParentId.get(parentId) ?? [];
    siblings.push(turn.turn_id);
    childrenByParentId.set(parentId, siblings);
  }

  for (const siblings of childrenByParentId.values()) {
    siblings.sort((leftId, rightId) => {
      const left = turnById.get(leftId);
      const right = turnById.get(rightId);
      return (left?.version ?? 0) - (right?.version ?? 0);
    });
  }

  const rows: BranchTreeRow[] = [];
  const visited = new Set<string>();
  const visitNode = (turnId: string, depth: number) => {
    if (visited.has(turnId)) return;
    const turn = turnById.get(turnId);
    if (!turn) return;
    visited.add(turnId);
    const children = childrenByParentId.get(turnId) ?? [];
    rows.push({
      turn,
      depth,
      childCount: children.length,
      hasDetachedParent: detachedParentSet.has(turnId),
    });
    for (const childId of children) {
      visitNode(childId, depth + 1);
    }
  };

  for (const root of roots) {
    visitNode(root.turn_id, 0);
  }

  // 兜底处理异常链路，避免任何 turn 在树上丢失。
  for (const turn of turnsByVersion) {
    if (!visited.has(turn.turn_id)) {
      visitNode(turn.turn_id, 0);
    }
  }
  return rows;
}

function collectAnchorLineageTurnIds(
  turns: ConversationTurnSummary[],
  anchorTurnId: string,
): Set<string> {
  const target = anchorTurnId.trim();
  if (!target) return new Set<string>();
  const turnById = new Map(turns.map((turn) => [turn.turn_id, turn]));
  const result = new Set<string>();
  let cursor: string | undefined = target;
  while (cursor) {
    const turn = turnById.get(cursor);
    if (!turn) break;
    if (result.has(turn.turn_id)) break;
    result.add(turn.turn_id);
    const parentId = (turn.parent_turn_id || "").trim();
    cursor = parentId || undefined;
  }
  return result;
}

function parseConflictHint(error: unknown): ConflictHint | null {
  if (!(error instanceof ApiError) || error.status !== 409) {
    return null;
  }
  const detail = error.detail;
  if (!detail || typeof detail !== "object") {
    return {
      kind: "request_in_flight",
      message: "请求冲突（409），请稍后重试。",
    };
  }

  const detailObj = detail as {
    error?: unknown;
    message?: unknown;
    expected_version?: unknown;
    current_version?: unknown;
  };
  const errorCode = String(detailObj.error ?? "");

  if (errorCode === "conversation_version_conflict") {
    const expectedVersion =
      typeof detailObj.expected_version === "number" ? detailObj.expected_version : undefined;
    const currentVersion =
      typeof detailObj.current_version === "number" ? detailObj.current_version : undefined;
    return {
      kind: "conversation_version_conflict",
      expectedVersion,
      currentVersion,
      message:
        currentVersion !== undefined
          ? `会话版本冲突：本地 expected_version=${expectedVersion ?? "unknown"}，服务端最新版本=${currentVersion}。`
          : "会话版本冲突，请刷新到最新版本后重试。",
    };
  }

  if (errorCode === "request_in_flight") {
    return {
      kind: "request_in_flight",
      message: String(detailObj.message ?? "同一 request_id 正在处理中，请稍后重试。"),
    };
  }

  return {
    kind: "request_in_flight",
    message: error.message || "请求冲突（409），请稍后重试。",
  };
}

export function DashboardPage() {
  const queryClient = useQueryClient();
  const [traceVersionInput, setTraceVersionInput] = useState("");
  const [conflictHint, setConflictHint] = useState<ConflictHint | null>(null);
  const [newTurnId, setNewTurnId] = useState<string | null>(null);
  const [showAnchorBranchOnly, setShowAnchorBranchOnly] = useState(false);
  const dialogueListRef = useRef<HTMLOListElement | null>(null);
  const previousLatestTurnIdRef = useRef<string | null>(null);
  const {
    userId,
    conversationId,
    message,
    action,
    expectedVersion,
    targetReportId,
    fromTurnId,
    symbolsText,
    taskContextText,
    workflowNodes,
    result,
    selectedReport,
    errorText,
    setUserId,
    setConversationId,
    setMessage,
    setAction,
    setExpectedVersion,
    setTargetReportId,
    setFromTurnId,
    setSymbolsText,
    setTaskContextText,
    setWorkflowNodes,
    setResult,
    setSelectedReport,
    setErrorText,
  } = useConsoleStore();

  const resolvedConversationId = useMemo(
    () => normalizeConversationId(conversationId, userId),
    [conversationId, userId],
  );

  const conversationMetaQuery = useQuery({
    queryKey: ["conversation-meta", resolvedConversationId],
    queryFn: () => getConversationMeta(resolvedConversationId),
    enabled: Boolean(resolvedConversationId),
    retry: false,
  });

  const turnsQuery = useInfiniteQuery({
    queryKey: ["conversation-turns", resolvedConversationId],
    queryFn: ({ pageParam }) =>
      listConversationTurns(resolvedConversationId, {
        limit: TURN_PAGE_SIZE,
        beforeVersion: pageParam,
      }),
    initialPageParam: undefined as number | undefined,
    getNextPageParam: (lastPage) => {
      if (lastPage.length < TURN_PAGE_SIZE) return undefined;
      return Math.min(...lastPage.map((turn) => turn.version));
    },
    enabled: Boolean(resolvedConversationId),
  });

  const reportsQuery = useInfiniteQuery({
    queryKey: ["conversation-reports", resolvedConversationId],
    queryFn: ({ pageParam }) =>
      listConversationReports(resolvedConversationId, {
        limit: REPORT_PAGE_SIZE,
        beforeVersion: pageParam,
      }),
    initialPageParam: undefined as number | undefined,
    getNextPageParam: (lastPage) => {
      if (lastPage.length < REPORT_PAGE_SIZE) return undefined;
      return Math.min(...lastPage.map((report) => report.report_version));
    },
    enabled: Boolean(resolvedConversationId),
  });

  const reportDetailQuery = useQuery({
    queryKey: ["conversation-report", resolvedConversationId, targetReportId],
    queryFn: () => getConversationReport(resolvedConversationId, targetReportId),
    enabled: Boolean(resolvedConversationId && targetReportId.trim()),
  });

  const queryMutation = useMutation({
    mutationFn: async () => {
      return sendConversationMessage(resolvedConversationId, {
        user_id: userId,
        message,
        task_context: parseTaskContext(taskContextText, symbolsText),
        action,
        target_report_id: targetReportId.trim() || undefined,
        from_turn_id: fromTurnId.trim() || undefined,
        expected_version: typeof expectedVersion === "number" ? expectedVersion : undefined,
      });
    },
    onMutate: () => {
      setConflictHint(null);
      setErrorText("");
      setWorkflowNodes(BASE_WORKFLOW.map((node) => ({ ...node, status: "idle", durationMs: undefined })));
    },
    onSuccess: (response) => {
      setResult(response);
      setConversationId(response.conversation_id);
      setExpectedVersion(response.conversation_version);
      setTraceVersionInput(String(response.conversation_version));
      setWorkflowNodes(toWorkflowFromSteps(response.workflow_steps));
      setErrorText(response.errors.join("；"));
      setConflictHint(null);
      if (response.report) {
        setSelectedReport(response.report);
        setTargetReportId(response.report.report_id);
      }
      queryClient.invalidateQueries({ queryKey: ["conversation-meta", response.conversation_id] });
      queryClient.invalidateQueries({ queryKey: ["conversation-turns", response.conversation_id] });
      queryClient.invalidateQueries({ queryKey: ["conversation-reports", response.conversation_id] });
      queryClient.invalidateQueries({ queryKey: ["conversation-report", response.conversation_id] });
    },
    onError: (error) => {
      setWorkflowNodes(BASE_WORKFLOW.map((node) => ({ ...node, status: "idle", durationMs: undefined })));
      const hint = parseConflictHint(error);
      if (hint) {
        setConflictHint(hint);
        if (hint.kind === "conversation_version_conflict" && typeof hint.currentVersion === "number") {
          setExpectedVersion(hint.currentVersion);
        }
      } else {
        setConflictHint(null);
      }
      const messageText = hint?.message || (error instanceof Error ? error.message : "请求失败");
      setErrorText(messageText);
    },
  });

  const taskContextPreview = useMemo(
    () => parseTaskContext(taskContextText, symbolsText),
    [taskContextText, symbolsText],
  );

  const turns = useMemo(() => (turnsQuery.data?.pages ?? []).flat(), [turnsQuery.data?.pages]);
  const branchTreeRows = useMemo(() => buildBranchTreeRows(turns), [turns]);
  const anchorLineageTurnIds = useMemo(
    () => collectAnchorLineageTurnIds(turns, fromTurnId),
    [turns, fromTurnId],
  );
  const hasAnchorLineage = anchorLineageTurnIds.size > 0;
  const visibleBranchTreeRows = useMemo(() => {
    if (!showAnchorBranchOnly) return branchTreeRows;
    if (!hasAnchorLineage) return branchTreeRows;
    return branchTreeRows.filter((row) => anchorLineageTurnIds.has(row.turn.turn_id));
  }, [showAnchorBranchOnly, hasAnchorLineage, branchTreeRows, anchorLineageTurnIds]);
  const dialogueTurns = useMemo(
    () => [...turns].sort((left, right) => left.version - right.version),
    [turns],
  );
  const latestDialogueTurnId =
    dialogueTurns.length > 0 ? dialogueTurns[dialogueTurns.length - 1].turn_id : null;
  const reports = useMemo(() => (reportsQuery.data?.pages ?? []).flat(), [reportsQuery.data?.pages]);
  const latestReport = reports[0];
  const activeReport = useMemo(
    () => getStatusReport(selectedReport, reportDetailQuery.data, latestReport, result?.report),
    [selectedReport, reportDetailQuery.data, latestReport, result?.report],
  );
  const activeReportParts = useMemo(() => splitThinkAndReport(activeReport?.report ?? ""), [activeReport?.report]);
  const renderedReport = useMemo(() => renderMarkdown(activeReportParts.report), [activeReportParts.report]);

  const assistantParts = useMemo(
    () => splitThinkAndReport(result?.assistant_message ?? ""),
    [result?.assistant_message],
  );
  const renderedAssistant = useMemo(
    () => renderMarkdown(assistantParts.report),
    [assistantParts.report],
  );

  const handleSubmit = () => {
    setResult(null);
    queryMutation.mutate();
  };

  const handleTraceVersion = async () => {
    const targetVersion = Number(traceVersionInput.trim());
    if (!Number.isInteger(targetVersion) || targetVersion <= 0) {
      setErrorText("请输入有效的版本号（正整数）。");
      return;
    }

    let found = turns.find((turn) => turn.version === targetVersion);
    while (!found && turnsQuery.hasNextPage) {
      const next = await turnsQuery.fetchNextPage();
      const merged = (next.data?.pages ?? []).flat();
      found = merged.find((turn) => turn.version === targetVersion);
      if (!next.hasNextPage) break;
    }

    if (!found) {
      setErrorText(`未找到 v${targetVersion}，请继续加载更多会话轮次后重试。`);
      return;
    }

    setFromTurnId(found.turn_id);
    if (found.report_id) {
      setTargetReportId(found.report_id);
    }
    if (conversationMetaQuery.data?.latest_version) {
      setExpectedVersion(conversationMetaQuery.data.latest_version);
    }
    setErrorText(`已回溯到 v${found.version}，后续消息将从该 turn 分支续写。`);
  };

  const handleBackToLatest = () => {
    setFromTurnId("");
    setTraceVersionInput("");
    if (conversationMetaQuery.data?.latest_version) {
      setExpectedVersion(conversationMetaQuery.data.latest_version);
    }
    setErrorText("已切回最新会话上下文。");
  };

  const handleRetryAfterConflict = () => {
    if (!conflictHint) return;
    if (conflictHint.kind === "conversation_version_conflict" && typeof conflictHint.currentVersion === "number") {
      setExpectedVersion(conflictHint.currentVersion);
    }
    queryMutation.mutate();
  };

  const reportCount = reports.length;

  useEffect(() => {
    if (!latestDialogueTurnId) {
      previousLatestTurnIdRef.current = null;
      return;
    }

    const previousTurnId = previousLatestTurnIdRef.current;
    previousLatestTurnIdRef.current = latestDialogueTurnId;
    if (!previousTurnId || previousTurnId === latestDialogueTurnId) {
      return;
    }

    setNewTurnId(latestDialogueTurnId);
    const timer = window.setTimeout(() => {
      setNewTurnId((current) => (current === latestDialogueTurnId ? null : current));
    }, 560);
    return () => window.clearTimeout(timer);
  }, [latestDialogueTurnId]);

  useEffect(() => {
    if (!latestDialogueTurnId || !dialogueListRef.current) return;
    dialogueListRef.current.scrollTo({
      top: dialogueListRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [latestDialogueTurnId]);

  return (
    <section className="dashboard-shell">
      <div className="panel panel--query">
        <header className="panel-head">
          <p className="panel-kicker">Conversation</p>
          <h2>Message Composer</h2>
        </header>

        <div className="field-grid">
          <label className="field">
            <span>User ID</span>
            <input value={userId} onChange={(e) => setUserId(e.target.value)} placeholder="u001" />
          </label>
          <label className="field">
            <span>Conversation ID</span>
            <input
              value={conversationId}
              onChange={(e) => setConversationId(e.target.value)}
              placeholder={`默认自动使用 ${`default:${userId || "u001"}`}`}
            />
          </label>
        </div>

        <label className="field">
          <span>Action</span>
          <select
            className="action-select"
            value={action}
            onChange={(e) => setAction(e.target.value as ConversationAction)}
          >
            <option value="auto">auto（自动路由）</option>
            <option value="chat">chat（继续对话）</option>
            <option value="rewrite_report">rewrite_report（改写报告）</option>
            <option value="regenerate_report">regenerate_report（重跑报告）</option>
          </select>
        </label>

        <label className="field">
          <span>Message</span>
          <textarea
            rows={5}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="请输入对话消息，或描述你希望改写/重跑的要求"
          />
        </label>

        <div className="field-grid">
          <label className="field">
            <span>Symbols</span>
            <input
              value={symbolsText}
              onChange={(e) => setSymbolsText(e.target.value)}
              placeholder="BTC,ETH,SOL"
            />
          </label>
          <label className="field">
            <span>From Turn ID（可选）</span>
            <input
              value={fromTurnId}
              onChange={(e) => setFromTurnId(e.target.value)}
              placeholder="用于分支续写"
            />
          </label>
          <label className="field">
            <span>Expected Version（CAS）</span>
            <input
              value={expectedVersion ?? ""}
              onChange={(e) => {
                const raw = e.target.value.trim();
                if (!raw) {
                  setExpectedVersion(null);
                  return;
                }
                const parsed = Number(raw);
                if (Number.isInteger(parsed) && parsed >= 0) {
                  setExpectedVersion(parsed);
                }
              }}
              placeholder="留空表示不校验版本"
            />
          </label>
        </div>

        <label className="field">
          <span>Task Context (JSON)</span>
          <textarea
            rows={3}
            value={taskContextText}
            onChange={(e) => setTaskContextText(e.target.value)}
            placeholder='{"risk_mode":"balanced"}'
          />
        </label>

        {action === "rewrite_report" ? (
          <label className="field">
            <span>Target Report ID</span>
            <input
              value={targetReportId}
              onChange={(e) => setTargetReportId(e.target.value)}
              placeholder="从右侧 Version Tape 选择，或手动输入"
            />
          </label>
        ) : null}

        <button
          type="button"
          className="action-button"
          onClick={handleSubmit}
          disabled={queryMutation.isPending || !userId.trim() || !message.trim()}
        >
          {queryMutation.isPending ? "发送中..." : `发送消息（${getActionText(action)}）`}
        </button>

        <p className="helper-line">
          路由会话：<code>{resolvedConversationId}</code>
        </p>
        <p className="helper-line">
          expected_version：<code>{expectedVersion ?? "null"}</code> · task_context：<code>{JSON.stringify(taskContextPreview)}</code>
        </p>

        <div className="rollback-row">
          <input
            value={traceVersionInput}
            onChange={(e) => setTraceVersionInput(e.target.value)}
            placeholder="输入版本号，例如 12"
          />
          <button type="button" className="ghost-button" onClick={handleTraceVersion}>
            回溯到版本
          </button>
          <button type="button" className="ghost-button" onClick={handleBackToLatest}>
            回到最新
          </button>
        </div>

        <p className="helper-line">
          会话元信息：
          {conversationMetaQuery.data ? (
            <>
              {" "}
              latest_version=<code>{conversationMetaQuery.data.latest_version}</code> ·
              turn_count=<code>{conversationMetaQuery.data.turn_count}</code>
            </>
          ) : (
            " - "
          )}
        </p>

        {conflictHint ? (
          <div className="conflict-banner" role="status" aria-live="polite">
            <p>{conflictHint.message}</p>
            <div className="turn-actions">
              <button
                type="button"
                className="ghost-button"
                onClick={handleRetryAfterConflict}
                disabled={queryMutation.isPending}
              >
                按最新版本重试
              </button>
              {conflictHint.currentVersion !== undefined ? (
                <span className="helper-line">已自动同步 expected_version={conflictHint.currentVersion}</span>
              ) : null}
            </div>
          </div>
        ) : null}
      </div>

      <div className="panel panel--dialogue">
        <header className="panel-head">
          <p className="panel-kicker">Dialogue</p>
          <h2>User & Assistant Stream</h2>
        </header>

        {dialogueTurns.length ? (
          <ol className="dialogue-list" ref={dialogueListRef}>
            {dialogueTurns.map((turn) => {
              const assistantRaw = turn.assistant_message || turn.report || "";
              const assistantParts = splitThinkAndReport(assistantRaw);
              return (
                <li
                  key={`dialogue-${turn.turn_id}`}
                  className={`dialogue-turn ${fromTurnId === turn.turn_id ? "dialogue-turn--anchor" : ""} ${newTurnId === turn.turn_id ? "dialogue-turn--new" : ""}`}
                >
                  <div className="dialogue-meta">
                    <span>v{turn.version}</span>
                    <span>{turn.intent || "unknown"}</span>
                    <span>{formatTime(turn.created_at)}</span>
                  </div>

                  <article className="bubble bubble--user">
                    <p className="bubble-role">USER</p>
                    <p>{turn.query}</p>
                  </article>

                  <article className="bubble bubble--assistant">
                    <p className="bubble-role">ASSISTANT</p>
                    <div className="bubble-markdown">{renderMarkdown(assistantParts.report)}</div>
                  </article>
                </li>
              );
            })}
          </ol>
        ) : (
          <p className="empty-state empty-state--center">暂无对话记录，发送第一条消息后这里会展示完整用户/AI 对话流。</p>
        )}
      </div>

      <div className="panel panel--timeline">
        <header className="panel-head">
          <p className="panel-kicker">Timeline</p>
          <h2>Conversation Turns</h2>
        </header>

        {turnsQuery.isLoading ? <p className="helper-line">加载会话轮次中...</p> : null}

        {branchTreeRows.length ? (
          <section className="branch-tree-panel">
            <header className="branch-tree-head">
              <h3>Branch Tree</h3>
              <label className="branch-toggle">
                <input
                  type="checkbox"
                  checked={showAnchorBranchOnly}
                  onChange={(event) => setShowAnchorBranchOnly(event.target.checked)}
                />
                <span>仅看当前锚点分支</span>
              </label>
              <p className="helper-line">
                已加载 <code>{turns.length}</code> 条 turn，当前展示 <code>{visibleBranchTreeRows.length}</code> 条。
              </p>
              {showAnchorBranchOnly && !hasAnchorLineage ? (
                <p className="helper-line">未设置有效 from_turn_id，已回退展示全量分支树。</p>
              ) : null}
            </header>
            <ol className="branch-tree-list">
              {visibleBranchTreeRows.map((row) => {
                const turn = row.turn;
                const roleTag = row.childCount > 1 ? `fork×${row.childCount}` : row.childCount ? "branch" : "leaf";
                return (
                  <li
                    key={`branch-tree-${turn.turn_id}`}
                    className={`branch-node ${row.depth === 0 ? "branch-node--root" : ""} ${fromTurnId === turn.turn_id ? "branch-node--active" : ""} ${newTurnId === turn.turn_id ? "branch-node--new" : ""}`}
                    style={{ marginLeft: `${Math.min(row.depth, 10) * 22}px` }}
                  >
                    <div className="branch-node__meta">
                      <span>v{turn.version}</span>
                      <span>{turn.intent || "unknown"}</span>
                      <span>{roleTag}</span>
                      <span>{formatTime(turn.created_at)}</span>
                    </div>
                    <p className="branch-node__query">{turn.query}</p>
                    <p className="branch-node__hint">
                      {row.hasDetachedParent
                        ? `父节点 ${turn.parent_turn_id || "-"} 未加载，当前以该节点作为局部根展示。`
                        : `turn_id=${turn.turn_id}${turn.parent_turn_id ? ` · parent=${turn.parent_turn_id}` : " · root"}`}
                    </p>
                    <div className="turn-actions">
                      {turn.report_id ? (
                        <button
                          type="button"
                          className="ghost-button"
                          onClick={() => setTargetReportId(turn.report_id ?? "")}
                        >
                          选为目标报告
                        </button>
                      ) : null}
                      <button
                        type="button"
                        className="ghost-button"
                        onClick={() => {
                          setFromTurnId(turn.turn_id);
                          setTraceVersionInput(String(turn.version));
                        }}
                      >
                        设为 from_turn_id
                      </button>
                    </div>
                  </li>
                );
              })}
            </ol>
          </section>
        ) : null}

        {turns.length ? (
          <ol className="turn-list">
            {turns.map((turn) => (
              <li key={turn.turn_id} className={`turn-item ${fromTurnId === turn.turn_id ? "turn-item--active" : ""}`}>
                <div className="turn-meta">
                  <span>v{turn.version}</span>
                  <span>{turn.intent || "unknown"}</span>
                  <span>{formatTime(turn.created_at)}</span>
                </div>
                <p className="turn-query">{turn.query}</p>
                <p className="turn-answer">{turn.assistant_message || turn.report || "-"}</p>
                <div className="turn-actions">
                  {turn.report_id ? (
                    <button
                      type="button"
                      className="ghost-button"
                      onClick={() => setTargetReportId(turn.report_id ?? "")}
                    >
                      选为目标报告
                    </button>
                  ) : null}
                  <button
                    type="button"
                    className="ghost-button"
                    onClick={() => {
                      setFromTurnId(turn.turn_id);
                      setTraceVersionInput(String(turn.version));
                    }}
                  >
                    设为 from_turn_id
                  </button>
                </div>
              </li>
            ))}
          </ol>
        ) : (
          <p className="empty-state">当前会话暂无历史轮次，发送第一条消息后会自动展示。</p>
        )}

        <div className="pager-row">
          <button
            type="button"
            className="ghost-button"
            onClick={() => turnsQuery.fetchNextPage()}
            disabled={!turnsQuery.hasNextPage || turnsQuery.isFetchingNextPage}
          >
            {turnsQuery.isFetchingNextPage ? "加载中..." : turnsQuery.hasNextPage ? "加载更多轮次" : "已到底部"}
          </button>
          <span className="helper-line">已加载 {turns.length} 条</span>
        </div>
      </div>

      <div className="panel panel--report">
        <header className="panel-head">
          <p className="panel-kicker">Version Tape</p>
          <h2>Report Versions ({reportCount})</h2>
        </header>

        {reports.length ? (
          <div className="report-tape">
            {reports.map((report) => (
              <button
                key={report.report_id}
                type="button"
                className={`report-chip ${targetReportId === report.report_id ? "report-chip--active" : ""}`}
                onClick={() => {
                  setTargetReportId(report.report_id);
                  setSelectedReport(report);
                }}
              >
                <span>v{report.report_version}</span>
                <span>{report.mode}</span>
                <span>{report.report_id.slice(0, 8)}</span>
              </button>
            ))}
          </div>
        ) : (
          <p className="empty-state empty-state--center">暂无报告版本。执行 regenerate/rewrite 后会在这里累积版本资产。</p>
        )}

        <div className="pager-row">
          <button
            type="button"
            className="ghost-button"
            onClick={() => reportsQuery.fetchNextPage()}
            disabled={!reportsQuery.hasNextPage || reportsQuery.isFetchingNextPage}
          >
            {reportsQuery.isFetchingNextPage
              ? "加载中..."
              : reportsQuery.hasNextPage
                ? "加载更多报告版本"
                : "已到底部"}
          </button>
          <span className="helper-line">已加载 {reports.length} 个版本</span>
        </div>

        {result ? <p className="trace-line">trace_id: {result.trace_id}</p> : null}

        {result && result.action_taken === "chat" && !activeReport ? (
          <article className="report-body">{renderedAssistant}</article>
        ) : null}

        {activeReport ? (
          <>
            <div className="report-meta">
              <span>report_id: {activeReport.report_id}</span>
              <span>version: v{activeReport.report_version}</span>
              <span>mode: {activeReport.mode}</span>
              {activeReport.based_on_report_id ? (
                <span>based_on: {activeReport.based_on_report_id}</span>
              ) : null}
            </div>

            {activeReportParts.think ? (
              <details className="think-block">
                <summary>LLM Think（推理草稿）</summary>
                <pre>{activeReportParts.think}</pre>
              </details>
            ) : null}

            <article className="report-body">{renderedReport}</article>

            <section className="citation-block">
              <h3>Citations ({activeReport.citations.length})</h3>
              <ul>
                {activeReport.citations.map((citation, idx) => (
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
        ) : null}

        {errorText ? <p className="error-text">{errorText}</p> : null}
      </div>

      <PulseRail nodes={workflowNodes} isPending={queryMutation.isPending} />
    </section>
  );
}
