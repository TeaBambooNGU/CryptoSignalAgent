export type JsonDict = Record<string, unknown>;

export interface QueryRequest {
  user_id: string;
  query: string;
  task_context?: JsonDict;
}

export interface Citation {
  source: string;
  raw_ref: string;
  snippet: string;
  published_at?: string | null;
}

export interface QueryResponse {
  report: string;
  citations: Citation[];
  trace_id: string;
  errors: string[];
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
