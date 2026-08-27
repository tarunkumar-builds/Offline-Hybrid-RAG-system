import axios from "axios";
import type { BenchmarkResponse, DocumentRecord, HealthResponse, ModelsResponse, QueryResponse, SystemConfigResponse, UploadResponse } from "../types/api";

const defaultBaseUrl = "http://127.0.0.1:8000/api/v1";
const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();

export const api = axios.create({
  baseURL: configuredBaseUrl || defaultBaseUrl,
  timeout: 90_000,
  headers: { "X-Request-ID": crypto.randomUUID() },
});

export const ragApi = {
  health: () => api.get<HealthResponse>("/health").then(({ data }) => data),
  models: () => api.get<ModelsResponse>("/system/models").then(({ data }) => data),
  config: () => api.get<SystemConfigResponse>("/system/config").then(({ data }) => data),
  documents: () => api.get<DocumentRecord[]>("/documents").then(({ data }) => data),
  deleteDocument: (id: string) => api.delete(`/documents/${encodeURIComponent(id)}`),
  uploadDocuments: (files: File[], onProgress?: (progress: number) => void) => {
    const form = new FormData(); files.forEach((file) => form.append("files", file));
    return api.post<UploadResponse>("/documents/upload", form, { onUploadProgress: (event) => onProgress?.(event.total ? Math.round((event.loaded / event.total) * 100) : 0) }).then(({ data }) => data);
  },
  query: (payload: { question: string; top_k: number; evaluation_enabled: boolean }) => api.post<QueryResponse>("/query", payload).then(({ data }) => data),
  uploadBenchmark: (file: File) => { const form = new FormData(); form.append("file", file); return api.post<BenchmarkResponse>("/evaluation/benchmark/upload", form).then(({ data }) => data); },
};
