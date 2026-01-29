from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, MetadataMode
from sqlalchemy import text
from config import CHUNK_SIZE, BATCH_SIZE
import re

def load_documents(file_paths: List[str]):
    return SimpleDirectoryReader(input_files=file_paths, file_metadata=get_metadata).load_data()

def get_metadata(filename: str) -> dict:
    return {
        "source": filename
    }
    
def chunk_documents(documents) -> List[TextNode]:
    text_parser = SentenceSplitter(chunk_size=CHUNK_SIZE)
    
    nodes = []
    for doc in documents:
        clean_text = doc.text.replace('\x00', '')
        chunks = text_parser.split_text(clean_text)
        doc_metadata = doc.get_metadata()

        for chunk in chunks:
            node = TextNode(text=chunk)
            node.metadata = doc_metadata
            nodes.append(node)
    
    return nodes


def embed_nodes(nodes: List[TextNode], embed_model) -> List[TextNode]:
    texts = [node.get_content(metadata_mode=MetadataMode.ALL) for node in nodes]
    embeddings = embed_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE
    ).tolist()
    for node, embedding in zip(nodes, embeddings):
        node.embedding = embedding.tolist()
    return nodes
    