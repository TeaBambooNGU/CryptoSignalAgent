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
  parent_turn_id?: string | null;
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

export interface KnowledgeDocumentCreate {
  title: string;
  source: string;
  doc_type: string;
  symbols: string[];
  tags: string[];
  text: string;
  kb_id?: string;
  language?: string;
  published_at?: string;
  metadata?: JsonDict;
}

export interface KnowledgeDocumentRequest {
  user_id: string;
  task_id?: string;
  document: KnowledgeDocumentCreate;
}

export interface KnowledgeDocumentRecord {
  doc_id: string;
  kb_id: string;
  title: string;
  source: string;
  doc_type: string;
  symbols: string[];
  tags: string[];
  language: string;
  file_name: string;
  content_type: string;
  checksum: string;
  status: string;
  chunk_count: number;
  uploaded_by: string;
  published_at?: string | null;
  created_at: number;
  updated_at: number;
}

export interface KnowledgeDocumentResponse {
  success: boolean;
  task_id: string;
  inserted_chunks: number;
  document: KnowledgeDocumentRecord;
}

export interface KnowledgeDocumentListResponse {
  items: KnowledgeDocumentRecord[];
}
