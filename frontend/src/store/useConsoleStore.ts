import { create } from "zustand";
import type {
  ConversationAction,
  ConversationMessageResponse,
  ConversationReport,
} from "../api/types";

export interface WorkflowNode {
  id: string;
  label: string;
  status: "idle" | "running" | "success" | "error";
  durationMs?: number;
}

interface ConsoleState {
  userId: string;
  conversationId: string;
  message: string;
  action: ConversationAction;
  expectedVersion: number | null;
  targetReportId: string;
  fromTurnId: string;
  symbolsText: string;
  taskContextText: string;
  workflowNodes: WorkflowNode[];
  result: ConversationMessageResponse | null;
  selectedReport: ConversationReport | null;
  errorText: string;
  setUserId: (value: string) => void;
  setConversationId: (value: string) => void;
  setMessage: (value: string) => void;
  setAction: (value: ConversationAction) => void;
  setExpectedVersion: (value: number | null) => void;
  setTargetReportId: (value: string) => void;
  setFromTurnId: (value: string) => void;
  setSymbolsText: (value: string) => void;
  setTaskContextText: (value: string) => void;
  setWorkflowNodes: (nodes: WorkflowNode[]) => void;
  setResult: (result: ConversationMessageResponse | null) => void;
  setSelectedReport: (report: ConversationReport | null) => void;
  setErrorText: (error: string) => void;
}

export const BASE_WORKFLOW: WorkflowNode[] = [
  { id: "load_user_memory", label: "加载记忆", status: "idle" },
  { id: "resolve_symbols", label: "解析标的", status: "idle" },
  { id: "collect_signals_via_mcp", label: "采集信号", status: "idle" },
  { id: "normalize_and_index", label: "标准化入库", status: "idle" },
  { id: "retrieve_context", label: "检索上下文", status: "idle" },
  { id: "analyze_signals", label: "分析信号", status: "idle" },
  { id: "generate_report", label: "生成研报", status: "idle" },
  { id: "persist_memory", label: "写回记忆", status: "idle" },
  { id: "finalize_response", label: "输出结果", status: "idle" },
];

export const useConsoleStore = create<ConsoleState>((set) => ({
  userId: "u001",
  conversationId: "default:u001",
  message: "请先给我 BTC 和 ETH 的 24 小时风险信号研报",
  action: "auto",
  expectedVersion: null,
  targetReportId: "",
  fromTurnId: "",
  symbolsText: "BTC,ETH",
  taskContextText: "{}",
  workflowNodes: BASE_WORKFLOW,
  result: null,
  selectedReport: null,
  errorText: "",
  setUserId: (value) => set({ userId: value }),
  setConversationId: (value) => set({ conversationId: value }),
  setMessage: (value) => set({ message: value }),
  setAction: (value) => set({ action: value }),
  setExpectedVersion: (value) => set({ expectedVersion: value }),
  setTargetReportId: (value) => set({ targetReportId: value }),
  setFromTurnId: (value) => set({ fromTurnId: value }),
  setSymbolsText: (value) => set({ symbolsText: value }),
  setTaskContextText: (value) => set({ taskContextText: value }),
  setWorkflowNodes: (nodes) => set({ workflowNodes: nodes }),
  setResult: (result) => set({ result }),
  setSelectedReport: (report) => set({ selectedReport: report }),
  setErrorText: (error) => set({ errorText: error }),
}));
