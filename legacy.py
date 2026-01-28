import psycopg2
from typing import Any, List, Optional

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.llms.llama_cpp import LlamaCPP
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from llama_index.vector_stores.postgres import PGVectorStore


llm = LlamaCPP(
    model_url="/home/freen/.cache/llama_index/models/llama-2-13b-chat.Q4_0.gguf",
    model_kwargs={"n_gpu_layers": 40},
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    verbose=True,
)

# llm = LlamaCPP(
#     model_url="/home/freen/.cache/llama_index/models/llama-2-13b-chat.Q4_0.gguf",
#     temperature=0.1,
#     max_new_tokens=256,
#     context_window=3900,
#     model_kwargs={"n_gpu_layers": 1},
#     verbose=True,
# )


embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


db_name = "vector_db"
host = "localhost"
password = "password"
port = "5432"
user = "user"

# Create database fresh
conn = psycopg2.connect(
    dbname="postgres",
    host=host,
    password=password,
    port=port,
    user=user,
)
conn.autocommit = True

with conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")

conn.close()

# Connect PGVectorStore
vector_store = PGVectorStore.from_params(
    database=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
    table_name="ug_bulletins",
    embed_dim=384,
)

documents = SimpleDirectoryReader(
    input_files=[
        # "/mnt/c/Users/freen/Documents/chatdku/supportingDocuments/ug_bulletin2023-24.pdf",
        # "/mnt/c/Users/freen/Documents/chatdku/supportingDocuments/ug_bulletin2024-25.pdf",
        "/mnt/c/Users/freen/Documents/chatdku/supportingDocuments/ug_bulletin2025-26.pdf"
    ]
).load_data()

text_parser = SentenceSplitter(chunk_size=1024) # maybe try larger chunks or gpu 

text_chunks = []
doc_idxs = []

for doc_idx, doc in enumerate(documents):
    chunks = text_parser.split_text(doc.text)
    text_chunks.extend(chunks)
    doc_idxs.extend([doc_idx] * len(chunks))



nodes = []

for idx, chunk in enumerate(text_chunks):
    node = TextNode(text=chunk)
    node.metadata = documents[doc_idxs[idx]].metadata

    # embed
    node.embedding = embed_model.encode(
        node.get_content(metadata_mode="all"),
        normalize_embeddings=True )

    nodes.append(node)

vector_store.add(nodes)


class VectorDBRetriever(BaseRetriever):
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 2,
    ) -> None:
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_embedding = embed_model.encode(
            query_bundle.query_str,
            normalize_embeddings=True
            )


        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )

        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores


retriever = VectorDBRetriever(
    vector_store=vector_store,
    embed_model=embed_model,
    query_mode="default",
    similarity_top_k=3,
)


query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)

query_str = "How would I request a transcript?"
response = query_engine.query(query_str)

print("\n=== ANSWER ===\n")
print(str(response))
