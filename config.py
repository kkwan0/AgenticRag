import os

# Database configuration
DB_NAME = os.getenv("DB_NAME", "vector_db")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

# Model paths
LLM_MODEL_PATH = os.getenv(
    "LLM_MODEL_PATH",
    "/home/freen/.cache/llama_index/models/llama-2-13b-chat.Q4_0.gguf"
)

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# LLM settings
LLM_TEMPERATURE = 0.1
LLM_MAX_NEW_TOKENS = 256
LLM_CONTEXT_WINDOW = 3900
LLM_GPU_LAYERS = 40

# Chunking settings
CHUNK_SIZE = 1024

# Retrieval settings
SIMILARITY_TOP_K = 3

# Vector store table name
VECTOR_TABLE_NAME = "ug_bulletins"

# Document paths
DOCUMENT_PATHS = [
    "/mnt/c/Users/freen/Documents/chatdku/supportingDocuments/ug_bulletin2025-26.pdf"
]