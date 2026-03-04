import type {
  ConversationMeta,
  ConversationMessageRequest,
  ConversationMessageResponse,
  PaginationOptions,
  ConversationReport,
  ConversationTurnSummary,
  IngestRequest,
  IngestResponse,
  QueryRequest,
  QueryResponse,
  UserPreferencesRequest,
  UserPreferencesResponse,
  UserProfileResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE?.trim() || "";

class ApiError extends Error {
  readonly status: number;
  readonly detail: unknown;
  readonly raw: string;

  constructor(message: string, status: number, detail: unknown, raw: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.raw = raw;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });

  if (!response.ok) {
    const body = await response.text();
    let parsed: unknown = null;
    if (body) {
      try {
        parsed = JSON.parse(body) as unknown;
      } catch {
        parsed = null;
      }
    }
    const detail =
      parsed && typeof parsed === "object" && "detail" in parsed
        ? (parsed as { detail?: unknown }).detail
        : parsed;
    const detailMessage =
      detail && typeof detail === "object" && "message" in detail
        ? String((detail as { message?: unknown }).message ?? "")
        : "";
    const message = detailMessage || body || `HTTP ${response.status}`;
    throw new ApiError(message, response.status, detail, body);
  }

  return (await response.json()) as T;
}

export async function queryResearch(payload: QueryRequest): Promise<QueryResponse> {
  return request<QueryResponse>("/v1/research/query", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function sendConversationMessage(
  conversationId: string,
  payload: ConversationMessageRequest,
): Promise<ConversationMessageResponse> {
  return request<ConversationMessageResponse>(
    `/v1/conversation/${encodeURIComponent(conversationId)}/message`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

export async function getConversationMeta(conversationId: string): Promise<ConversationMeta> {
  return request<ConversationMeta>(`/v1/conversation/${encodeURIComponent(conversationId)}`);
}

function buildPaginationQuery(options: PaginationOptions = {}): string {
  const params = new URLSearchParams();
  if (typeof options.limit === "number") {
    params.set("limit", String(options.limit));
  }
  if (typeof options.beforeVersion === "number") {
    params.set("before_version", String(options.beforeVersion));
  }
  const query = params.toString();
  return query ? `?${query}` : "";
}

function buildReportPaginationQuery(options: PaginationOptions = {}): string {
  const params = new URLSearchParams();
  if (typeof options.limit === "number") {
    params.set("limit", String(options.limit));
  }
  if (typeof options.beforeVersion === "number") {
    params.set("before_report_version", String(options.beforeVersion));
  }
  const query = params.toString();
  return query ? `?${query}` : "";
}

export async function listConversationReports(
  conversationId: string,
  options: PaginationOptions = {},
): Promise<ConversationReport[]> {
  return request<ConversationReport[]>(
    `/v1/conversation/${encodeURIComponent(conversationId)}/reports${buildReportPaginationQuery(options)}`,
  );
}

export async function getConversationReport(
  conversationId: string,
  reportId: string,
): Promise<ConversationReport> {
  return request<ConversationReport>(
    `/v1/conversation/${encodeURIComponent(conversationId)}/reports/${encodeURIComponent(reportId)}`,
  );
}

export async function listConversationTurns(
  conversationId: string,
  options: PaginationOptions = {},
): Promise<ConversationTurnSummary[]> {
  return request<ConversationTurnSummary[]>(
    `/v1/conversation/${encodeURIComponent(conversationId)}/turns${buildPaginationQuery(options)}`,
  );
}

export async function updatePreferences(
  payload: UserPreferencesRequest,
): Promise<UserPreferencesResponse> {
  return request<UserPreferencesResponse>("/v1/user/preferences", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getUserProfile(userId: string): Promise<UserProfileResponse> {
  return request<UserProfileResponse>(`/v1/user/profile/${encodeURIComponent(userId)}`);
}

export async function ingestDocuments(payload: IngestRequest): Promise<IngestResponse> {
  return request<IngestResponse>("/v1/research/ingest", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export { ApiError };
