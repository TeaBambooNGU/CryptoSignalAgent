export type JsonDict = Record<string, unknown>;

export type ConversationAction = "auto" | "chat" | "rewrite_report" | "regenerate_report";

export interface Citation {
  source: string;
  raw_ref: string;
  snippet: string;
  published_at?: string | null;
}

export interface WorkflowStep {
  node_id: string;
  status: "success" | "error";
  duration_ms: number;
}

export interface QueryRequest {
  user_id: string;
  query: string;
  task_context?: JsonDict;
  conversation_id?: string;
  turn_id?: string;
  request_id?: string;
  expected_version?: number;
}

export interface QueryResponse {
  conversation_id: string;
  turn_id: string;
  request_id: string;
  conversation_version: number;
  trace_id: string;
  report: string;
  citations: Citation[];
  errors: string[];
  workflow_steps: WorkflowStep[];
}

export interface ConversationTurnSummary {
  conversation_id: string;
  turn_id: string;
  version: number;
  request_id: string;
  user_id: string;
  query: string;
  report: string;
  assistant_message: string;
  trace_id: string;
  status: string;
  intent: string;
  turn_type: string;
  report_id?: string | null;
  created_at: number;
  updated_at: number;
}

export interface ConversationMeta {
  conversation_id: string;
  latest_version: number;
  latest_turn_id: string;
  turn_count: number;
  updated_at: number;
}

export interface ConversationReport {
  report_id: string;
  conversation_id: string;
  report_version: number;
  created_by_turn_id: string;
  based_on_report_id?: string | null;
  mode: string;
  report: string;
  citations: Citation[];
  workflow_steps: WorkflowStep[];
  status: string;
  created_at: number;
  updated_at: number;
}

export interface ConversationMessageRequest {
  user_id: string;
  message: string;
  task_context?: JsonDict;
  action?: ConversationAction;
  target_report_id?: string;
  from_turn_id?: string;
  request_id?: string;
  expected_version?: number;
}

export interface ConversationMessageResponse {
  conversation_id: string;
  turn_id: string;
  request_id: string;
  conversation_version: number;
  action_taken: ConversationAction;
  assistant_message: string;
  trace_id: string;
  errors: string[];
  workflow_steps: WorkflowStep[];
  report?: ConversationReport | null;
}

export interface PaginationOptions {
  limit?: number;
  beforeVersion?: number;
}

export interface UserPreferencesRequest {
  user_id: string;
  preference: JsonDict;
  confidence: number;
}

export interface UserPreferencesResponse {
  success: boolean;
  user_id: string;
}

export interface UserProfileResponse {
  user_id: string;
  long_term_memory: JsonDict[];
  session_memory: JsonDict[];
}

export interface IngestDocument {
  doc_id: string;
  symbol: string;
  source: string;
  published_at: string;
  text: string;
  metadata: JsonDict;
}

export interface IngestRequest {
  user_id: string;
  task_id?: string;
  documents: IngestDocument[];
}

export interface IngestResponse {
  success: boolean;
  inserted_chunks: number;
  task_id: string;
}
