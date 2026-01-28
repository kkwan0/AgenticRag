from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME


def get_embed_model() -> SentenceTransformer:
    """Load and return the sentence transformer embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)