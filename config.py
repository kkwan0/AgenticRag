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
EMBED_DIM = 384
LLM_NAME = "llama2:13b"
LLM_TEMPERATURE = 0.1
LLM_TIMEOUT = 120.0
LLM_KWARGS = {
    # "n_gpu_layers": 40,
    # Add other model-specific kwargs here   
    }
# LLM_PATH = "/home/freen/.cache/llama_index/models/llama-2-13b-chat.Q4_0.gguf"

# Chunking
CHUNK_SIZE = 1024

# Retrieval
SIMILARITY_TOP_K = 3