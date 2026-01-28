from typing import List
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, MetadataMode
from config import CHUNK_SIZE


def load_documents(file_paths: List[str]):
    return SimpleDirectoryReader(input_files=file_paths).load_data()


def chunk_documents(documents) -> List[TextNode]:
    text_parser = SentenceSplitter(chunk_size=CHUNK_SIZE)
    
    nodes = []
    for doc in documents:
        clean_text = doc.text.replace('\x00', '')
        chunks = text_parser.split_text(clean_text)
        
        for chunk in chunks:
            node = TextNode(text=chunk)
            node.metadata = doc.metadata
            nodes.append(node)
    
    return nodes


def embed_nodes(nodes: List[TextNode], embed_model) -> List[TextNode]:
    for node in nodes:
        node.embedding = embed_model.encode(
            node.get_content(metadata_mode=MetadataMode.ALL),
            normalize_embeddings=True
        ).tolist()
    return nodes