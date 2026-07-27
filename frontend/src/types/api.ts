export type ApiStatus = "ready" | "missing" | "unavailable" | "healthy" | "degraded";

export interface HealthResponse {
  status: ApiStatus;
  ollama_status: ApiStatus;
  embedding_model: string;
  reranker_model: string;
  indexed_documents: number;
  vector_count: number;
  database_status: ApiStatus;
}

export interface ModelsResponse { embedding_model: string; reranker_model: string; llm_model: string; }
export interface SystemConfigResponse { configuration: Record<string, unknown>; }
export interface DocumentRecord { document_id: string; document_name: string; ingestion_time: string; chunk_count: number; }
export interface UploadResponse { documents: DocumentRecord[]; indexed_chunks: number; }
export interface Citation { document_name: string; page_number: number; chunk_number: number; snippet: string; }
export interface RetrievedChunk { chunk_id: string; document_name: string; page_number: number; chunk_number: number; text: string; retrieval_score: number; rerank_score: number; rank: number; }
export interface GenerationMetrics { precision: number | null; recall: number | null; f1: number | null; generation_time: number; answer_available: boolean; }
export interface EvaluationResult { generation: GenerationMetrics; citations: { citation_coverage: number }; performance: { total_pipeline_time: number }; }
export interface QueryResponse { answer: string; citations: Citation[]; retrieved_chunks: RetrievedChunk[]; model_name: string; generation_time: number; processing_time: number; evaluation: EvaluationResult | null; }
export interface BenchmarkSummary { total_queries: number; successful_queries: number; success_rate: number; average_latency: number; average_citation_coverage: number; best_question: string | null; worst_question: string | null; }
export interface BenchmarkResponse { results: EvaluationResult[]; summary: BenchmarkSummary; processing_time: number; }
export interface ApiError { detail?: string | Array<{ msg?: string }>; }
