"""
Main entry point for the RAG pipeline.

Usage:
    python main.py                          # Run with default query
    python main.py "Your question here"     # Run with custom query
    python main.py --rebuild                # Rebuild the database and index
"""

import argparse

from llama_index.core.query_engine import RetrieverQueryEngine

from config import SIMILARITY_TOP_K
from db import create_database, get_vector_store
from embeddings import get_embed_model
from llm import get_llm
from document_loader import process_documents
from retriever import VectorDBRetriever


def build_index(embed_model) -> None:
    """Create database and populate with embedded documents."""
    print("Creating database...")
    create_database()
    
    print("Connecting to vector store...")
    vector_store = get_vector_store()
    
    print("Processing documents...")
    nodes = process_documents(embed_model)
    
    print(f"Inserting {len(nodes)} nodes into vector store...")
    vector_store.add(nodes)
    
    print("Index built successfully!")


def get_query_engine(embed_model, llm):
    """Create and return the query engine."""
    vector_store = get_vector_store()
    
    retriever = VectorDBRetriever(
        vector_store=vector_store,
        embed_model=embed_model,
        query_mode="default",
        similarity_top_k=SIMILARITY_TOP_K,
    )
    
    return RetrieverQueryEngine.from_args(retriever, llm=llm)


def query(query_str: str, query_engine) -> str:
    """Run a query and return the response."""
    response = query_engine.query(query_str)
    return str(response)


def main():
    parser = argparse.ArgumentParser(description="RAG Query System")
    parser.add_argument(
        "query",
        nargs="?",
        default="How would I request a transcript?",
        help="The question to ask"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the database and index before querying"
    )
    args = parser.parse_args()

    print("Loading embedding model...")
    embed_model = get_embed_model()

    if args.rebuild:
        build_index(embed_model)

    print("Loading LLM...")
    llm = get_llm()

    print("Creating query engine...")
    query_engine = get_query_engine(embed_model, llm)

    print(f"\n=== QUERY ===\n{args.query}\n")
    
    response = query(args.query, query_engine)
    
    print("\n=== ANSWER ===\n")
    print(response)


if __name__ == "__main__":
    main()