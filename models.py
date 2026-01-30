import dspy
from sentence_transformers import SentenceTransformer
from llama_index.llms.llama_cpp import LlamaCPP
from config import EMBED_MODEL_NAME
from llama_index.llms.ollama import Ollama
from config import LM_NAME
_embed_model = None
_lm = None
def get_embed_model(): # initialize embedding model once
    global _embed_model
    if _embed_model is None:
        print("Loading embedding model...")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model
def get_lm(): # initialize LLM once
    global _lm
    if _lm is None:
        print("Loading DSPY LLM...")
        _lm = dspy.LM(LM_NAME, api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=_lm)
    return _lm
def get_named_lm(name: str): # initialize LLM once
    global _lm
    if _lm is None:
        print(f"Loading {name} LLM...")
        _lm = dspy.LM(name, api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=_lm)
    return _lm