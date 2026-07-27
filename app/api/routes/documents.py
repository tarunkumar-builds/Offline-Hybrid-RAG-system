"""Versioned document ingestion and lifecycle endpoints."""

import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Path as PathParameter, UploadFile, status

from app.api.dependencies import ApiServices, get_services
from app.api.schemas.documents import DocumentResponse, UploadResponse
from app.models import DocumentRecord


router = APIRouter(prefix="/documents", tags=["Documents"])

_ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf"}
_FILENAME_UNSAFE_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_documents(
    files: list[UploadFile] = File(...), services: ApiServices = Depends(get_services)
) -> UploadResponse:
    """Validate, safely stage, and index one or more PDF documents."""
    if not files:
        raise HTTPException(status_code=400, detail="At least one PDF is required")
    if len(files) > services.settings.max_upload_files:
        raise HTTPException(status_code=400, detail="Too many files in one upload request")

    indexed_chunks = 0
    documents: list[DocumentResponse] = []
    try:
        with tempfile.TemporaryDirectory() as directory:
            staged_documents = []
            for uploaded in files:
                filename = _safe_pdf_filename(uploaded)
                contents = await uploaded.read(services.settings.max_upload_size_bytes + 1)
                if len(contents) > services.settings.max_upload_size_bytes:
                    raise HTTPException(status_code=413, detail="Uploaded PDF exceeds the configured size limit")
                if not contents or b"%PDF-" not in contents[:1024]:
                    raise HTTPException(status_code=415, detail="Uploaded content is not a valid PDF")
                temporary_path = Path(directory) / filename
                temporary_path.write_bytes(contents)
                staged_documents.append(temporary_path)

            for temporary_path in staged_documents:
                result = services.ingest_document(temporary_path)
                if result is None:
                    raise HTTPException(status_code=409, detail="An identical document has already been indexed")
                document, chunk_count = result
                indexed_chunks += chunk_count
                documents.append(_document_response(document, chunk_count))
    finally:
        for uploaded in files:
            await uploaded.close()
    return UploadResponse(documents=documents, indexed_chunks=indexed_chunks)


@router.get("", response_model=list[DocumentResponse])
def list_documents(services: ApiServices = Depends(get_services)) -> list[DocumentResponse]:
    """List public metadata for all indexed documents."""
    chunk_counts = services.document_chunk_counts()
    return [_document_response(item, chunk_counts.get(item.document_id, 0)) for item in services.list_documents()]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: str = PathParameter(min_length=1, max_length=128),
    services: ApiServices = Depends(get_services),
) -> DocumentResponse:
    """Return public metadata for one indexed document."""
    document = services.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return _document_response(document, services.document_chunk_counts().get(document.document_id, 0))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str = PathParameter(min_length=1, max_length=128),
    services: ApiServices = Depends(get_services),
) -> None:
    """Delete one document and its associated retrieval artifacts."""
    if not services.delete_document(document_id):
        raise HTTPException(status_code=404, detail="Document not found")


def _safe_pdf_filename(uploaded: UploadFile) -> str:
    if not uploaded.filename or Path(uploaded.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported")
    if uploaded.content_type and uploaded.content_type.lower() not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported")
    filename = Path(uploaded.filename).name
    stem = _FILENAME_UNSAFE_CHARACTERS.sub("_", Path(filename).stem).strip("._")
    if not stem:
        raise HTTPException(status_code=400, detail="Uploaded filename is invalid")
    return f"{stem[:120]}.pdf"


def _document_response(document: DocumentRecord, chunk_count: int) -> DocumentResponse:
    return DocumentResponse(
        document_id=document.document_id,
        document_name=document.document_name,
        ingestion_time=document.ingestion_time,
        chunk_count=chunk_count,
    )
