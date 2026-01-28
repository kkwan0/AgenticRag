# models.py
from sentence_transformers import SentenceTransformer
from llama_index.llms.llama_cpp import LlamaCPP
from config import EMBED_MODEL_NAME, LLM_PATH

_embed_model = None
_llm = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("Loading embedding model...")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model

def get_llm():
    global _llm
    if _llm is None:
        print("Loading LLM...")
        _llm = LlamaCPP(
            model_url=LLM_PATH,
            model_kwargs={"n_gpu_layers": 40},
            temperature=0.1,
            max_new_tokens=256,
            context_window=3900,
            verbose=True,
        )
    return _llm