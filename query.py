from retriever import VectorDBRetriever
from db import connect_vector_store
from models import get_embed_model, get_lm
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate
from rag import RAG

def query(question: str) -> str:
    get_lm()  # Ensure LLM is loaded
    rag = RAG(k=3)
    answer = rag(question=question)
    return answer.answer
    