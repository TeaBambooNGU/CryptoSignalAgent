import { create } from "zustand";
import type { QueryResponse } from "../api/types";

export interface WorkflowNode {
  id: string;
  label: string;
  status: "idle" | "running" | "success" | "error";
  durationMs?: number;
}

interface ConsoleState {
  userId: string;
  query: string;
  symbolsText: string;
  taskContextText: string;
  workflowNodes: WorkflowNode[];
  result: QueryResponse | null;
  errorText: string;
  setUserId: (value: string) => void;
  setQuery: (value: string) => void;
  setSymbolsText: (value: string) => void;
  setTaskContextText: (value: string) => void;
  setWorkflowNodes: (nodes: WorkflowNode[]) => void;
  setResult: (result: QueryResponse | null) => void;
  setErrorText: (error: string) => void;
}

export const BASE_WORKFLOW: WorkflowNode[] = [
  { id: "load_user_memory", label: "加载记忆", status: "idle" },
  { id: "parse_intent_scope", label: "解析意图", status: "idle" },
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
  query: "请给我 BTC 和 ETH 的 24 小时风险信号研报",
  symbolsText: "BTC,ETH",
  taskContextText: "{}",
  workflowNodes: BASE_WORKFLOW,
  result: null,
  errorText: "",
  setUserId: (value) => set({ userId: value }),
  setQuery: (value) => set({ query: value }),
  setSymbolsText: (value) => set({ symbolsText: value }),
  setTaskContextText: (value) => set({ taskContextText: value }),
  setWorkflowNodes: (nodes) => set({ workflowNodes: nodes }),
  setResult: (result) => set({ result }),
  setErrorText: (error) => set({ errorText: error }),
}));
