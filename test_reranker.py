from sentence_transformers import CrossEncoder

print("Loading model...")

model = CrossEncoder(
    "BAAI/bge-reranker-base",
    local_files_only=True,
)

print("Model loaded successfully!")