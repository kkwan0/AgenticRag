import dspy
from db import connect_vector_store
from models import get_embed_model, get_llm
from llama_index.core.vector_stores import VectorStoreQuery

class GenerateAnswer(dspy.Signature):
    context = dspy.InputField(desc='retrieved passages from vector store')
    question = dspy.InputField()
    answer = dspy.OutputField()