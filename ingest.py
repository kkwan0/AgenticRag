import torch

from sentence_transformers import SentenceTransformer
from db import create_database, connect_vector_store
from documents import load_documents, chunk_documents, embed_nodes


def ingest(file_paths: list[str]):
    print("initialize database")
    create_database()
    
    print("loading models")
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
    
    print("loading and chunking documents")
    documents = load_documents(file_paths)
    nodes = chunk_documents(documents)
    nodes = embed_nodes(nodes, embed_model)
    
    print(f"Adding {len(nodes)} nodes to vector store...")
    connect_vector_store().add(nodes) # type: ignore
    print("Ingestion complete!")
    