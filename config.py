import os
# Database
DB_NAME = "vector_db"
DB_HOST = "localhost"
DB_PASSWORD = "password"
DB_PORT = "5432"
DB_USER = "user"
TABLE_NAME = "ug_bulletins"

# Models
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DEVICE = "cuda"
EMBED_DIM = 384
LM_NAME = "ollama_chat/llama2:13b" # DSPY LLM name



# Chunking
CHUNK_SIZE = 1024
BATCH_SIZE = 32
# Retrieval
SIMILARITY_TOP_K = 3
TOP_K = 3

#Other

FILE_PATHS = [
    "/mnt/c/Users/freen/Documents/AgenticRag/supportingDocuments/ug_bulletin2025-26.pdf",
    "/mnt/c/Users/freen/Documents/AgenticRag/supportingDocuments/ug_bulletin2024-25.pdf",
    "/mnt/c/Users/freen/Documents/AgenticRag/supportingDocuments/ug_bulletin2023-24.pdf",
]