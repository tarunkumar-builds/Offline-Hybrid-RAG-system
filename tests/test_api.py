"""API route tests using dependency-injected offline service fakes."""

from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.dependencies import get_services
from app.api.main import create_app
from app.evaluation.evaluator import Evaluator
from app.evaluation.models import EvaluationInput
from app.generation.models import Citation, GeneratedAnswer
from app.models import DocumentRecord
from app.reranker.models import RerankedChunk


def test_document_upload_and_list() -> None:
    services = FakeServices()
    client = _client(services)

    response = client.post("/api/v1/documents/upload", files=[("files", ("report.pdf", b"%PDF-1.7", "application/pdf"))])

    assert response.status_code == 201
    assert response.json()["indexed_chunks"] == 2
    assert client.get("/api/v1/documents").json()[0]["document_name"] == "report.pdf"


def test_query_and_evaluation_routes_return_structured_json() -> None:
    services = FakeServices()
    client = _client(services)

    query = client.post("/api/v1/query", json={"question": "What changed?", "evaluation_enabled": True})
    evaluation = client.post("/api/v1/evaluation/query", json={"evaluation_input": _evaluation_input().model_dump(mode="json")})

    assert query.status_code == 200
    assert query.json()["answer"] == "The policy changed. [S1]"
    assert query.json()["evaluation"]["generation"]["answer_available"] is True
    assert evaluation.status_code == 200
    assert evaluation.json()["result"]["question"] == "What changed?"


def test_health_and_system_routes() -> None:
    client = _client(FakeServices())

    health = client.get("/api/v1/health")
    models = client.get("/api/v1/system/models")

    assert health.status_code == 200
    assert health.json()["indexed_documents"] == 1
    assert models.json()["llm_model"] == "gemma3"


def _client(services: "FakeServices") -> TestClient:
    app = create_app()
    app.dependency_overrides[get_services] = lambda: services
    return TestClient(app)


class FakeIngestion:
    def __init__(self, services: "FakeServices") -> None:
        self._services = services

    def ingest(self, path: Path) -> int:
        self._services.document = DocumentRecord(
            document_id="doc-1",
            document_name=path.name,
            file_path=path,
            file_hash="hash",
            ingestion_time=datetime.now(timezone.utc),
        )
        return 2


class FakeEvaluation:
    def evaluate(self, evaluation_input: EvaluationInput):
        return Evaluator().evaluate(evaluation_input)

    def evaluate_and_report(self, evaluation_input: EvaluationInput, name: str):
        return Evaluator().evaluate(evaluation_input)


class FakeSettings:
    embedding_model = "BAAI/bge-small-en-v1.5"
    reranker_model = "BAAI/bge-reranker-base"
    ollama_model = "gemma3"
    evaluation_dataset_path = None
    max_upload_files = 10
    max_upload_size_bytes = 25 * 1024 * 1024

    def model_dump(self, mode: str = "python") -> dict[str, object]:
        return {"ollama_model": self.ollama_model}


class FakeServices:
    def __init__(self) -> None:
        self.settings = FakeSettings()
        self.document = DocumentRecord(
            document_id="doc-1",
            document_name="report.pdf",
            file_path=Path("report.pdf"),
            file_hash="hash",
            ingestion_time=datetime.now(timezone.utc),
        )
        self.ingestion = FakeIngestion(self)
        self.evaluation = FakeEvaluation()

    def list_documents(self):
        return [self.document]

    def get_document(self, document_id: str):
        return self.document if document_id == self.document.document_id else None

    def delete_document(self, document_id: str) -> bool:
        return document_id == self.document.document_id

    def ingest_document(self, path: Path):
        indexed_chunks = self.ingestion.ingest(path)
        return self.document, indexed_chunks

    def answer(self, question: str, top_k: int, model_name, temperature, template) -> GeneratedAnswer:
        return _answer()

    def health(self):
        return "ready", 1, 1, "ready"

    def document_chunk_counts(self) -> dict[str, int]:
        return {self.document.document_id: 2}


def _chunk() -> RerankedChunk:
    return RerankedChunk(
        chunk_id="chunk-1", document_name="report.pdf", page_number=1, chunk_number=0,
        text="The policy changed.", retrieval_score=0.5, rerank_score=0.9, rank=1,
    )


def _answer() -> GeneratedAnswer:
    citation = Citation(document_name="report.pdf", page_number=1, chunk_number=0, snippet="The policy changed.")
    return GeneratedAnswer(
        answer="The policy changed. [S1]", citations=(citation,), source_documents=("report.pdf",),
        retrieved_chunks=(_chunk(),), model_name="gemma3", generation_time=0.1, prompt_tokens=4, response_tokens=4,
    )


def _evaluation_input() -> EvaluationInput:
    return EvaluationInput(question="What changed?", reranked_chunks=(_chunk(),), generated_answer=_answer())
