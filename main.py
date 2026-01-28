from llama_index.core.query_engine import RetrieverQueryEngine
from db import create_database, connect_vector_store
from retriever import VectorDBRetriever
from documents import embed_nodes, load_documents, chunk_documents
from sentence_transformers import SentenceTransformer
from llama_index.llms.llama_cpp import LlamaCPP
from models import get_embed_model, get_llm
import psycopg2

def ingest(file_paths: list[str]):
    print("initialize database")
    create_database()
    
    print("loading models")
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("loading and chunking documents")
    documents = load_documents(file_paths)
    nodes = chunk_documents(documents)
    nodes = embed_nodes(nodes, embed_model)
    
    print(f"Adding {len(nodes)} nodes to vector store...")
    connect_vector_store().add(nodes) # type: ignore
    print("Ingestion complete!")
    
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
    
def main():
    ingest(["/mnt/c/Users/freen/Documents/AgenticRag/supportingDocuments/ug_bulletin2025-26.pdf", "/mnt/c/Users/freen/Documents/AgenticRag/supportingDocuments/ug_bulletin2024-25.pdf", "/mnt/c/Users/freen/Documents/AgenticRag/supportingDocuments/ug_bulletin2023-24.pdf"])
    answer = query("How would I request a transcript?")
    print(answer)
    
    
    
if __name__ == "__main__":
    main()