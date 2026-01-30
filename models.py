import dspy
import time
from sentence_transformers import SentenceTransformer
from llama_index.llms.llama_cpp import LlamaCPP
from config import EMBED_MODEL_NAME
from llama_index.llms.ollama import Ollama
from config import LM_NAME
_embed_model = None
_lm = None
_lm_cache = {}
_load_times = {}
def get_embed_model(): # initialize embedding model once
    global _embed_model, _load_times
    if _embed_model is None:
        print("Loading embedding model...")
        start = time.time()
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        _load_times["embed"] = time.time() - start
    return _embed_model
def get_lm(): # initialize LLM once
    global _lm, _load_times
    if _lm is None:
        print("Loading DSPY LLM...")
        start = time.time()
        _lm = dspy.LM(LM_NAME, api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=_lm)
        _load_times["lm"] = time.time() - start
    return _lm
def get_named_lm(name: str): # initialize LLM from list
    global _lm_cache, _load_times
    if name not in _lm_cache:
        print(f"Loading {name} LLM...")
        _lm_cache[name] = dspy.LM(name, api_base='http://localhost:11434', api_key='')
    else:
        print(f"{name} LLM already loaded.")
    return _lm_cache[name]
def print_model_load_times():
    if not _load_times:
        print("No models have been loaded yet.")
        return
    for model_type, load_time in _load_times.items():
        print(f"{model_type.capitalize()} model load time: {load_time:.2f} seconds")
    print("--- End of model load times ---\n")