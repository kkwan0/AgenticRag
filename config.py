# Database
DB_NAME = "vector_db"
DB_HOST = "localhost"
DB_PASSWORD = "password"
DB_PORT = "5432"
DB_USER = "user"
TABLE_NAME = "ug_bulletins"

# Models
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBED_DEVICE = "cuda"
EMBED_DIM = 768 # 384 for minilm
RERANK_MODEL_NAME_FLAG = "BAAI/bge-reranker-v2-m3"
RERANK_MODEL_NAME_SENTENCE_TRANSFORMER = "mixedbread-ai/mxbai-rerank-base-v2"
LM_NAME = "ollama_chat/qwen3:8b" # DSPY LLM name
LM_TESTS = [
    "ollama_chat/llama3.1:8b",
    "ollama_chat/llama3.2:1b",
    "ollama_chat/qwen3:0.6b",
    "ollama_chat/qwen3:4b",
    "ollama_chat/qwen3:8b",
    "ollama_chat/qwen3:14b",
]


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