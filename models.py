import dspy
from sentence_transformers import SentenceTransformer
from llama_index.llms.llama_cpp import LlamaCPP
from config import EMBED_MODEL_NAME, LLM_NAME, LLM_TEMPERATURE, LLM_TIMEOUT, LLM_KWARGS
from llama_index.llms.ollama import Ollama

_embed_model = None
_lm = None
def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("Loading embedding model...")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model
def get_lm():
    global _lm
    if _lm is None:
        print("Loading DSPY LLM...")
        _lm = dspy.LM('ollama_chat/llama2:13b', api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=_lm)
    return _lm