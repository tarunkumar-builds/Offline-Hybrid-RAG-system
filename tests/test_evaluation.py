"""Unit tests for offline Phase 5 metrics, reports, datasets, and benchmarks."""

import json
from pathlib import Path

from app.evaluation.benchmark import BenchmarkRunner
from app.evaluation.dataset import DatasetLoader
from app.evaluation.evaluator import Evaluator
from app.evaluation.models import DatasetRecord, EvaluationInput
from app.evaluation.pipeline import EvaluationPipeline
from app.evaluation.config import EvaluationConfig
from app.evaluation.report_generator import ReportGenerator
from app.generation.models import Citation, GeneratedAnswer
from app.reranker.models import RerankedChunk
from app.retrieval.models import HybridResult


def test_evaluator_calculates_retrieval_generation_citation_and_reference_metrics() -> None:
    result = Evaluator().evaluate(_evaluation_input())

    assert result.retrieval.retrieved_chunk_count == 1
    assert result.retrieval.average_retrieval_score == 0.5
    assert result.generation.exact_match == 1.0
    assert result.citations.citation_coverage == 1.0
    assert result.performance.total_pipeline_time == 0.5


def test_dataset_loader_reads_json_records(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"records": [{"question": "What changed?", "expected_documents": ["report.pdf"]}]}), encoding="utf-8")

    records = DatasetLoader().load(path)

    assert records == [DatasetRecord(question="What changed?", expected_documents=("report.pdf",))]


def test_report_generator_writes_json_csv_and_markdown(tmp_path: Path) -> None:
    generator = ReportGenerator(tmp_path)
    result = Evaluator().evaluate(_evaluation_input())

    paths = [generator.write_evaluation(result, report_format, report_format) for report_format in ("json", "csv", "markdown")]

    assert all(path.is_file() for path in paths)
    assert "What changed?" in paths[0].read_text(encoding="utf-8")
    assert "RAG Evaluation Report" in paths[2].read_text(encoding="utf-8")


def test_benchmark_runner_aggregates_success_and_latency() -> None:
    records = [DatasetRecord(question="first"), DatasetRecord(question="second")]
    benchmark = BenchmarkRunner().run(records, lambda record: _evaluation_input(record.question))

    assert benchmark.summary.total_queries == 2
    assert benchmark.summary.success_rate == 1.0
    assert benchmark.summary.average_latency == 0.5


def test_evaluation_pipeline_evaluates_and_writes_report(tmp_path: Path) -> None:
    pipeline = EvaluationPipeline(Evaluator(), ReportGenerator(tmp_path), EvaluationConfig(output_directory=tmp_path))

    result = pipeline.evaluate_and_report(_evaluation_input(), name="single")

    assert result.question == "What changed?"
    assert (tmp_path / "single.json").is_file()


def _evaluation_input(question: str = "What changed?") -> EvaluationInput:
    retrieved = HybridResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        document_name="report.pdf",
        page_number=1,
        chunk_number=0,
        text="The policy changed.",
        rrf_score=0.5,
    )
    reranked = RerankedChunk(
        chunk_id="chunk-1",
        document_name="report.pdf",
        page_number=1,
        chunk_number=0,
        text="The policy changed.",
        retrieval_score=0.5,
        rerank_score=0.9,
        rank=1,
    )
    citation = Citation(document_name="report.pdf", page_number=1, chunk_number=0, snippet="The policy changed.")
    answer = GeneratedAnswer(
        answer="The policy changed.",
        citations=(citation,),
        source_documents=("report.pdf",),
        retrieved_chunks=(reranked,),
        model_name="gemma3",
        generation_time=0.5,
        prompt_tokens=10,
        response_tokens=4,
    )
    return EvaluationInput(
        question=question,
        retrieved_chunks=(retrieved,),
        reranked_chunks=(reranked,),
        generated_answer=answer,
        reference_answer="The policy changed.",
    )
