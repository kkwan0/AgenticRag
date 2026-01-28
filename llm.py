from llama_index.llms.llama_cpp import LlamaCPP

from config import (
    LLM_MODEL_PATH,
    LLM_TEMPERATURE,
    LLM_MAX_NEW_TOKENS,
    LLM_CONTEXT_WINDOW,
    LLM_GPU_LAYERS,
)


def get_llm() -> LlamaCPP:
    """Load and return the LlamaCPP model."""
    return LlamaCPP(
        model_url=LLM_MODEL_PATH,
        model_kwargs={"n_gpu_layers": LLM_GPU_LAYERS},
        temperature=LLM_TEMPERATURE,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        context_window=LLM_CONTEXT_WINDOW,
        verbose=True,
    )