from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from sentence_transformers import SentenceTransformer

from config import CHUNK_SIZE, DOCUMENT_PATHS


def load_documents(file_paths: List[str] = DOCUMENT_PATHS):
    """Load documents from the specified file paths."""
    return SimpleDirectoryReader(input_files=file_paths).load_data()


def chunk_documents(documents, chunk_size: int = CHUNK_SIZE) -> tuple[list, list]:
    """Split documents into chunks and track their source document indices."""
    text_parser = SentenceSplitter(chunk_size=chunk_size)
    
    text_chunks = []
    doc_idxs = []

    for doc_idx, doc in enumerate(documents):
        chunks = text_parser.split_text(doc.text)
        text_chunks.extend(chunks)
        doc_idxs.extend([doc_idx] * len(chunks))

    return text_chunks, doc_idxs


def create_nodes(
    documents,
    text_chunks: list,
    doc_idxs: list,
    embed_model: SentenceTransformer
) -> List[TextNode]:
    """Create TextNodes with embeddings from chunks."""
    nodes = []

    for idx, chunk in enumerate(text_chunks):
        node = TextNode(text=chunk)
        node.metadata = documents[doc_idxs[idx]].metadata

        # Embed the node content
        node.embedding = embed_model.encode(
            node.get_content(metadata_mode="all"),
            normalize_embeddings=True
        ).tolist()

        nodes.append(node)

    return nodes


def process_documents(
    embed_model: SentenceTransformer,
    file_paths: List[str] = DOCUMENT_PATHS,
    chunk_size: int = CHUNK_SIZE
) -> List[TextNode]:
    """Full pipeline: load, chunk, and create embedded nodes."""
    documents = load_documents(file_paths)
    text_chunks, doc_idxs = chunk_documents(documents, chunk_size)
    nodes = create_nodes(documents, text_chunks, doc_idxs, embed_model)
    return nodes