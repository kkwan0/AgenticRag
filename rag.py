import dspy
from db import connect_vector_store
from models import get_embed_model
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
        
        self.last_embedding_time = 0 # timing metrics
        self.last_query_time = 0
        self.total_time = 0
        
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
