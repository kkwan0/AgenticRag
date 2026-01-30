import dspy
from db import connect_vector_store
from models import get_embed_model, get_rerank_model
from llama_index.core.vector_stores import VectorStoreQuery
from config import TOP_K
import time
class GenerateAnswer(dspy.Signature): # Define signature, tells DSPY what inputs/outputs are expected
    context = dspy.InputField() # List of retrieved passages
    question = dspy.InputField() # User Input
    answer = dspy.OutputField()  # Generated Answer
    
class PGVectorRetriever(dspy.Module): # DSPy has own module for retrievers
    def __init__(self, k=TOP_K):
        self.vector_store = connect_vector_store() # From db.py
        self.embed_model = get_embed_model() # From models.py
        self.k = k # number of top similar chunks to retrieve
        self.rerank_top_k = k
        self.rerank_model = get_rerank_model() # Reranker model
        
        self.last_embedding_time = 0 # timing metrics
        self.last_query_time = 0
        self.total_time = 0
        self.last_rerank_time = 0
        
        
        super().__init__()

    def forward(self, query: str) -> dspy.Prediction: # takes query, returns prediction object
            total_start = time.time() # time total retrieval
            
            embed_time = time.time() # time embedding step
            query_embedding = self.embed_model.encode(
                query, normalize_embeddings=True
            ).tolist() #convert to list for JSON serialization
            self.last_embedding_time = time.time() - embed_time
            
            query_time = time.time() # time query step
            results = self.vector_store.query( # queries vector store to find k most similar chunks
                VectorStoreQuery(
                    query_embedding=query_embedding,
                    similarity_top_k=self.k
                )
            )
            self.last_query_time = time.time() - query_time # end query time
            if results.nodes:
                passages = [node.get_content() for node in results.nodes]
                sources = [{**node.metadata, "text": node.get_content()} for node in results.nodes]
                rerank_start = time.time()
                pairs = [[query, passage] for passage in passages]
                scores = self.rerank_model.compute_score(pairs)
                if isinstance(scores, list):
                    scored_items = list(zip(scores, passages, sources))
                else:
                    scored_items = list(zip([scores], passages, sources))
                
                scored_items.sort(key=lambda x: x[0], reverse=True)
                scored_items = scored_items[:self.rerank_top_k]
                
                # Extract reranked passages and sources
                passages = [item[1] for item in scored_items]
                sources = [item[2] for item in scored_items]
                
                self.last_rerank_time = time.time() - rerank_start
            else:
                passages = []
                sources = []
            self.total_time = time.time() - total_start
            return dspy.Prediction(passages=passages, sources=sources) # return passages and their metadata
    
class RAG(dspy.Module): # RAG module combining retriever and generator
    def __init__(self, k=TOP_K):
        self.retriever = PGVectorRetriever(k=k) 
        self.generate = dspy.ChainOfThought(GenerateAnswer) # uses DSPY's CoT for generation, has that step by step style
        
        self.last_retrieval_time = 0 # timing metrics
        self.last_generation_time = 0
        self.total_time = 0
        
    def forward(self, question: str): # takes user question
            total_start = time.time()
            
            retrieval_start = time.time()
            result = self.retriever(question) # retrieve relevant passages
            self.last_retrieval_time = time.time() - retrieval_start 
            context = result.passages # retrieved passages
            sources = result.sources

            generation_start = time.time()
            answer = self.generate(context=context, question=question)
            self.last_generation_time = time.time() - generation_start
            answer.sources = sources
            self.total_time = time.time() - total_start
            return answer
