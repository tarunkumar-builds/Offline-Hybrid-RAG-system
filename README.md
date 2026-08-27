# Offline Hybrid RAG

Offline Hybrid RAG is a local-first document question-answering system built with FastAPI, SQLite, FAISS, SentenceTransformers, a local reranker, Ollama, and a React/Vite frontend. It ingests PDFs, builds a local vector and metadata index, retrieves evidence with hybrid search, reranks results, and generates grounded answers with citations.

## What Happens On A Fresh Clone

On a fresh clone, the repository contains source code, configuration templates, and empty placeholder directories only.

- `documents/raw/` starts empty and receives copied source PDFs after ingestion.
- `documents/processed/` starts empty and receives chunked document JSON after ingestion.
- `metadata/metadata.db` does not exist yet and is created after the first successful ingestion.
- `vector_store/documents.faiss` does not exist yet and is created after the first successful ingestion.
- `logs/application.jsonl` is created when the backend or ingestion CLI runs.

The backend can start on a fresh clone, but querying is only useful after you ingest at least one PDF and generate the local corpus artifacts.

## Prerequisites

- Python 3.12
- Node.js and npm
- Ollama installed locally
- Local access to the required Hugging Face models before offline use

Default runtime models in the current code:

- Embedding model: `BAAI/bge-small-en-v1.5`
- Reranker model: `BAAI/bge-reranker-base`
- Ollama generation model: `qwen3:latest`

Both Hugging Face models are loaded with `local_files_only=True`, so they must already exist in the local Hugging Face cache before you run the system fully offline.

## First-Run Setup

1. Clone the repository and enter the project directory.

```powershell
git clone <your-repo-url>
cd Hybrid_offline_RAG
```

2. Create and activate the Python virtual environment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install backend dependencies.

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Create your backend environment file from the template.

```powershell
Copy-Item .env.example .env
```

5. Install frontend dependencies.

```powershell
cd frontend
npm install
cd ..
```

6. Review the environment templates.

- `.env.example` defines backend defaults such as the FAISS path, SQLite path, Ollama URL, and allowed frontend origins.
- `frontend/.env.example` defines `VITE_API_BASE_URL` for the frontend.

Create a frontend env file if you want to override the default local API URL:

```powershell
Copy-Item frontend/.env.example frontend/.env.local
```

## Prepare Models For Offline Use

### Ollama Model

Install and start Ollama, then pull the exact default backend model:

```powershell
ollama serve
ollama pull qwen3:latest
```

If you want to use a different Ollama model, update `RAG_OLLAMA_MODEL` in `.env` to match the model you pulled locally.

### Hugging Face Embedding And Reranker Models

The backend expects these local models to be available before offline execution:

- `BAAI/bge-small-en-v1.5`
- `BAAI/bge-reranker-base`

Because the code uses `local_files_only=True`, these models are not downloaded automatically at runtime. Prepare them in your local Hugging Face cache while you still have approved network access, or distribute them through your offline model delivery process. After they exist in the local cache, the application can load them offline.

## Ingest Documents

Add your own PDFs and ingest either a single file or a directory:

```powershell
python -m app.ingestion.cli C:\path\to\report.pdf
python -m app.ingestion.cli C:\path\to\pdf-folder
```

During ingestion, the pipeline:

- hashes the source PDF to detect duplicates
- copies the original PDF into `documents/raw/`
- cleans and chunks document text into `documents/processed/`
- stores vectors in `vector_store/documents.faiss`
- stores metadata in `metadata/metadata.db`

These generated artifacts are local runtime data and are intentionally not committed to Git.

## Start The Backend

Run the FastAPI server from the repository root:

```powershell
uvicorn app.api.main:app --reload
```

The default backend API base URL is `http://127.0.0.1:8000/api/v1`.

## Start The Frontend

Run the Vite development server from the `frontend/` directory:

```powershell
cd frontend
npm run dev
```

The frontend uses `VITE_API_BASE_URL` when it is set. Otherwise, it falls back to `http://127.0.0.1:8000/api/v1`.

## Expected Behavior Before Ingestion

Before you ingest any documents:

- system and health pages can still show backend and Ollama status
- document lists will be empty
- `metadata/metadata.db` and `vector_store/documents.faiss` may not exist yet
- query results will remain unavailable until a local corpus has been ingested

This is normal for a new workspace.

## Retrieval And Generation Overview

The retrieval pipeline embeds the query once with the local embedding model, performs dense FAISS search and sparse BM25 search, and merges results with Reciprocal Rank Fusion.

The reranker then uses the local `BAAI/bge-reranker-base` cross-encoder to select the most relevant chunks before answer generation.

Generation uses Ollama plus YAML prompt templates stored in `app/generation/prompt_templates/`. Available templates are `default`, `concise`, and `detailed`.

## Evaluation

The evaluation workflow supports local single-query evaluation and uploaded or file-based benchmark datasets in JSON, YAML, or CSV format. Reports are written to `reports/` using the configured format in `.env`.

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
  api/          FastAPI routes, schemas, and service wiring
  retrieval/    Dense search, BM25, filters, fusion, caching, orchestration
  reranker/     Cross-encoder config, scoring, cache, and pipeline
  generation/   YAML prompts, Ollama client, citations, and answer orchestration
  evaluation/   Metrics, datasets, reports, single-query and benchmark workflows
documents/raw/          Original indexed PDFs
documents/processed/    Cleaned chunk records
vector_store/           Persistent FAISS index
metadata/               SQLite database
logs/                   Rotating application logs
tests/                  Unit tests
```
