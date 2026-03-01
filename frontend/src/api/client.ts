import type {
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

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
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
    throw new ApiError(body || `HTTP ${response.status}`, response.status);
  }

  return (await response.json()) as T;
}

export async function queryResearch(payload: QueryRequest): Promise<QueryResponse> {
  return request<QueryResponse>("/v1/research/query", {
    method: "POST",
    body: JSON.stringify(payload),
  });
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
