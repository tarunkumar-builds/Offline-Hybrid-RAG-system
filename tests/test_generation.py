"""Unit tests for the offline Phase 4 generation engine."""

from pathlib import Path

from app.generation.answer_generator import AnswerGenerator
from app.generation.citation_builder import CitationBuilder
from app.generation.config import GenerationConfig
from app.generation.ollama_client import OllamaClient
from app.generation.pipeline import GenerationPipeline
from app.generation.prompt_builder import PromptBuilder
from app.generation.prompt_loader import PromptLoader
from app.reranker.models import RerankedChunk


def test_prompt_loader_and_builder_include_context_and_rules(tmp_path: Path) -> None:
    (tmp_path / "custom.yaml").write_text(
        "system_instructions: Use context only.\nresponse_rules: Be direct.\ncitation_instructions: Cite [S1].\n",
        encoding="utf-8",
    )
    builder = PromptBuilder(PromptLoader(tmp_path))

    prompt = builder.build("What changed?", [_chunk()], "custom")

    assert "SYSTEM INSTRUCTIONS:\nUse context only." in prompt
    assert "[S1] Document: report.pdf" in prompt
    assert "USER QUESTION:\nWhat changed?" in prompt


def test_citation_builder_selects_referenced_chunks_and_snippets() -> None:
    chunks = [_chunk("first"), _chunk("second", page_number=2, chunk_number=1)]

    citations = CitationBuilder().build("The evidence is in [S2].", chunks)

    assert len(citations) == 1
    assert citations[0].page_number == 2
    assert citations[0].snippet == "second"


def test_ollama_client_sends_configured_local_request() -> None:
    client = FakeHttpClient(FakeResponse({"response": "Grounded answer [S1]."}))
    config = GenerationConfig(model_name="gemma3", temperature=0.1, max_tokens=100)

    answer = OllamaClient(config, client=client).generate("test prompt")

    assert answer == "Grounded answer [S1]."
    assert client.requests[0][0] == "/api/generate"
    assert client.requests[0][1]["model"] == "gemma3"
    assert client.requests[0][1]["options"]["num_predict"] == 100


def test_answer_generator_returns_structured_answer_with_citations() -> None:
    generator = AnswerGenerator(
        GenerationConfig(),
        prompt_builder=FakePromptBuilder(),
        ollama_client=FakeOllamaClient("The report confirms the update [S1]."),
    )

    result = generator.generate("What changed?", [_chunk("The report confirms the update.")])

    assert result.answer.endswith("[S1].")
    assert result.citations[0].document_name == "report.pdf"
    assert result.model_name == "gemma3"
    assert result.prompt_tokens == 2


def test_generation_pipeline_calls_retrieval_reranking_and_generation() -> None:
    retriever = FakeRetriever([object()])
    reranker = FakeReranker([_chunk()])
    generator = FakeGenerator("answer")
    pipeline = GenerationPipeline(retriever, reranker, generator, reranker_input_limit=4)

    result = pipeline.answer("What changed?")

    assert retriever.request.limit == 4
    assert reranker.question == "What changed?"
    assert generator.context == [_chunk()]
    assert result == "answer"


class FakeResponse:
    """Minimal successful httpx response substitute."""

    def __init__(self, payload: dict[str, str]) -> None:
        self._payload = payload
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return self._payload


class FakeHttpClient:
    """Captures non-streaming Ollama calls."""

    def __init__(self, response: FakeResponse) -> None:
        self._response = response
        self.requests: list[tuple[str, dict[str, object]]] = []

    def post(self, path: str, json: dict[str, object]) -> FakeResponse:
        self.requests.append((path, json))
        return self._response


class FakePromptBuilder:
    """Deterministic prompt collaborator for answer generator testing."""

    def build(self, question: str, chunks: list[RerankedChunk], template_name: str) -> str:
        return "test prompt"

    def estimate_tokens(self, text: str) -> int:
        return len(text.split())


class FakeOllamaClient:
    """Returns a predefined local-model response."""

    def __init__(self, answer: str) -> None:
        self._answer = answer

    def generate(self, prompt: str) -> str:
        return self._answer


class FakeRetriever:
    """Captures the requested Phase 2 candidate limit."""

    def __init__(self, results: list[object]) -> None:
        self._results = results
        self.request = None

    def search(self, request):
        self.request = request
        return self._results


class FakeReranker:
    """Returns controlled context chunks without model inference."""

    def __init__(self, chunks: list[RerankedChunk]) -> None:
        self._chunks = chunks
        self.question = ""

    def rerank(self, question: str, candidates: list[object]) -> list[RerankedChunk]:
        self.question = question
        return self._chunks


class FakeGenerator:
    """Captures final generation input for pipeline testing."""

    def __init__(self, result: str) -> None:
        self._result = result
        self.context: list[RerankedChunk] = []

    def generate(self, question: str, context: list[RerankedChunk]) -> str:
        self.context = context
        return self._result


def _chunk(text: str = "first", page_number: int = 1, chunk_number: int = 0) -> RerankedChunk:
    return RerankedChunk(
        chunk_id=f"chunk-{page_number}-{chunk_number}",
        document_name="report.pdf",
        page_number=page_number,
        chunk_number=chunk_number,
        text=text,
        retrieval_score=0.1,
        rerank_score=0.9,
        rank=1,
    )
