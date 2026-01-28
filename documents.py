from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, MetadataMode
from config import CHUNK_SIZE
import re

def load_documents(file_paths: List[str]):
    return SimpleDirectoryReader(input_files=file_paths, file_metadata=get_metadata).load_data()

def get_metadata(filename: str) -> dict:
    match = re.search(r'(\d{4})-(\d{2})', filename)
    if match:
        start_year = match.group(1)                    # "2025"
        end_year = start_year[:2] + match.group(2)    # "2026"
        year = f"{start_year}-{end_year}"             # "2025-2026"
    else:
        year = "unknown"
    return {
        "source": filename,
        "year": year
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
    for node in nodes:
        node.embedding = embed_model.encode(
            node.get_content(metadata_mode=MetadataMode.ALL),
            normalize_embeddings=True
        ).tolist()
    return nodes