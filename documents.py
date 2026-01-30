from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, MetadataMode
from sqlalchemy import text
from config import CHUNK_SIZE, BATCH_SIZE
import time

_document_timings = {
    'load_docs' : [],
    'chunk_docs' : [],
    'embed_nodes' : []
}

def print_document_timings():
    for timing_type, timings in _document_timings.items():
        total_time = sum(timings)
        print(f"{timing_type} total time over {len(timings)} runs: {total_time:.2f} seconds")
    print("--- End of document processing timings ---\n")

def get_document_timings():
    return _document_timings

def load_documents(file_paths: List[str]):
    return SimpleDirectoryReader(input_files=file_paths).load_data()


    
def chunk_documents(documents) -> List[TextNode]:
    chunk_time = time.time()
    text_parser = SentenceSplitter(chunk_size=CHUNK_SIZE)
    
    nodes = []
    for doc in documents:
        clean_text = doc.text.replace('\x00', '')
        chunks = text_parser.split_text(clean_text)
        doc_metadata = doc.metadata

        for chunk in chunks:
            node = TextNode(text=chunk)
            node.metadata = doc_metadata
            nodes.append(node)
    _document_timings['chunk_docs'].append(time.time() - chunk_time)
    return nodes


def embed_nodes(nodes: List[TextNode], embed_model) -> List[TextNode]:
    embed_time = time.time()
    _document_timings['embed_nodes'].append(time.time() - embed_time)
    texts = [node.get_content(metadata_mode=MetadataMode.ALL) for node in nodes]
    embeddings = embed_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE
    ).tolist()
    for node, embedding in zip(nodes, embeddings):
        node.embedding = embedding
    _document_timings['embed_nodes'].append(time.time() - embed_time)
    return nodes
    