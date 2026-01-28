
from sentence_transformers import SentenceTransformer
from db import create_database, connect_vector_store
from documents import load_documents, chunk_documents, embed_nodes
from config import EMBED_MODEL_NAME, EMBED_DEVICE, EMBED_DIM

def ingest(file_paths: list[str]):
    print("initialize database")
    create_database() 
    
    print("loading models")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)
    
    print("loading and chunking documents")
    documents = load_documents(file_paths)
    nodes = chunk_documents(documents)
    nodes = embed_nodes(nodes, embed_model)
    
    print(f"Adding {len(nodes)} nodes to vector store...")
    connect_vector_store().add(nodes) # type: ignore
    print("Ingestion complete!")
    