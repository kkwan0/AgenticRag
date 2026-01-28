from retriever import VectorDBRetriever
from db import connect_vector_store
from models import get_embed_model, get_llm
from llama_index.core.query_engine import RetrieverQueryEngine

def query(query_str: str):
    print("loading models")
    embed_model = get_embed_model()
    llm = get_llm()
    
    print("connecting to vector store")
    vector_store = connect_vector_store()
    
    print("setting up retriever and query engine")
    retriever = VectorDBRetriever(
        vector_store=vector_store,
        embed_model=embed_model,
        similarity_top_k=2,
    )
    
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
    
    print("running query")
    response = query_engine.query(query_str)
    print("Response:")
    print(response)
    