# Offline Hybrid RAG — Phase 1

Phase 1 implements an entirely local PDF ingestion pipeline. It extracts text with PyMuPDF, cleans it, creates sentence-aware chunks, generates normalized BGE embeddings, stores vectors in FAISS, and stores metadata in SQLite. No cloud API is used at runtime.

## Prerequisites

- Python 3.12
- A locally available `BAAI/bge-small-en-v1.5` Sentence Transformers model

The embedding generator is deliberately configured with `local_files_only=True`. Before running in a disconnected environment, make the model available in the local Hugging Face cache through an approved internal/offline distribution process.

## Installation

```powershell
cd C:\Users\Tarun kumar\OneDrive\Documents\Hybrid_offline_RAG
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set any local path or model overrides in `.env`. All paths are relative to the project directory unless made absolute.

## Ingest documents

Place PDFs anywhere outside the generated storage directories, then index one file or a directory:

```powershell
python -m app.ingestion.cli C:\path\to\report.pdf
python -m app.ingestion.cli C:\path\to\pdf-folder
```

The pipeline hashes every source PDF using SHA256. A file whose content was already indexed is safely skipped. New documents are copied to `documents/raw/`, their cleaned chunk payload is saved to `documents/processed/`, vectors are written to `vector_store/documents.faiss`, and metadata is stored in `metadata/metadata.db`.

## Hybrid retrieval

Phase 2 adds a fully local hybrid retriever. It embeds a query once using the cached local BGE model, searches normalized FAISS vectors for cosine similarity, searches SQLite-backed chunk text with BM25, and combines both ranked lists using Reciprocal Rank Fusion (RRF): `score = sum(1 / (60 + rank))`.

Use it from Python after indexing at least one PDF:

```python
from app.config import Settings
from app.retrieval import HybridRetriever, SearchRequest

retriever = HybridRetriever(Settings())
results = retriever.search(SearchRequest(query="What are the key findings?", limit=3))
for result in results:
    print(result.document_name, result.page_number, result.rrf_score, result.text)
```

`SearchRequest` accepts optional `SearchFilters` for `document_name`, `document_id`, or `page_number`. Query embeddings and recent search responses use bounded LRU caches. The retriever reports embedding, FAISS, BM25, fusion, and total latency through Loguru.

## Cross-encoder reranking

Phase 3 rescoring improves the retrieval quality before any generation happens. Hybrid retrieval is fast: its bi-encoder represents the query and chunks independently for FAISS, while BM25 supplies complementary keyword matches. A cross-encoder instead reads each query-and-chunk pair together, producing a more precise relevance score for the small candidate set.

The local `BAAI/bge-reranker-base` model is loaded once, scores candidates in batches, and returns the best context chunks in descending `rerank_score` order. It is offline-only (`local_files_only=True`), uses configurable input/output limits, device, and cache size, and accepts Phase 2 `HybridResult` values directly.

```python
from app.config import Settings
from app.reranker import RerankingPipeline
from app.retrieval import HybridRetriever, SearchRequest

settings = Settings()
candidates = HybridRetriever(settings).search(SearchRequest(query="key findings", limit=20))
contexts = RerankingPipeline.from_settings(settings).rerank("key findings", candidates)
```

## Local answer generation

Phase 4 completes the offline pipeline: hybrid retrieval selects broad candidates, the cross-encoder narrows them to the most relevant chunks, and Ollama generates a grounded answer from a YAML-built prompt. The default model is configurable through `RAG_OLLAMA_MODEL` and supports `gemma3`, `llama3`, `mistral`, `phi4`, or any locally installed Ollama-compatible model.

Install and start Ollama, then download the chosen model once while connected to your approved model source:

```powershell
ollama serve
ollama pull gemma3
```

The available prompt styles are `default`, `concise`, and `detailed` in `app/generation/prompt_templates/`. Set `RAG_GENERATION_PROMPT_TEMPLATE` in `.env` to choose one. The engine uses `RAG_GENERATION_TEMPERATURE`, `RAG_GENERATION_TOP_P`, `RAG_GENERATION_MAX_TOKENS`, `RAG_GENERATION_TIMEOUT_SECONDS`, and `RAG_GENERATION_STREAMING` for local model behavior.

```python
from app.config import Settings
from app.generation import GenerationPipeline

answer = GenerationPipeline.from_settings(Settings()).answer("What are the key findings?")
print(answer.answer)
for citation in answer.citations:
    print(citation.document_name, citation.page_number, citation.chunk_number)
```

## Offline evaluation

Phase 5 evaluates the complete RAG result without needing cloud metrics. It computes retrieval volume, score, duplicate, document, and context measures; answer availability and optional reference-answer metrics (exact match, ROUGE-L, BLEU, precision, recall, F1); citation coverage and duplication; and normalized stage timings.

Use `EvaluationPipeline` for a completed query or `BenchmarkRunner` with JSON, YAML, or CSV records containing `question`, optional `reference_answer`, and optional `expected_documents`. Reports support JSON, CSV, and Markdown, selected through `RAG_EVALUATION_REPORT_FORMAT`; output defaults to `reports/`.

```python
from app.evaluation import EvaluationInput, EvaluationPipeline
from app.config import Settings

evaluation = EvaluationPipeline.from_settings(Settings())
result = evaluation.evaluate_and_report(EvaluationInput(
    question="What are the key findings?",
    retrieved_chunks=tuple(retrieved_chunks),
    reranked_chunks=tuple(reranked_chunks),
    generated_answer=generated_answer,
))
```

## Tests

```powershell
pytest -q
```

## Layout

```text
app/
  config/       Runtime configuration
  database/     SQLite document and chunk metadata
  ingestion/    PDF loading, cleaning, chunking, embedding, FAISS, CLI
  models/       Pydantic data contracts
  utils/        Logging, hashes, and domain errors
  api/          Reserved for Phase 3+
  retrieval/    Dense search, BM25, filters, fusion, caching, orchestration
  reranker/     Cross-encoder config, scoring, cache, and pipeline
  generation/   YAML prompts, Ollama client, citations, and answer orchestration
  evaluation/   Metrics, datasets, reports, single-query and benchmark workflows
documents/raw/          Original indexed PDFs
documents/processed/    Cleaned chunk records
vector_store/           Persistent FAISS index
metadata/               SQLite database
logs/                   Rotating ingestion logs
tests/                  Unit tests
```

## Next phase

Phase 6 can provide a thin local service or user interface that exposes ingestion, search, answers, and evaluation reports without changing these reusable core modules.
