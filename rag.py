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
        self.k = k * 5 # number of top similar chunks to retrieve, overretrieve more for reranking
        self.rerank_top_k = k
        self.rerank_model = get_rerank_model() # Reranker model
        
        
        super().__init__()

    def forward(self, query: str) -> dspy.Prediction: # takes query, returns prediction object
            
            query_embedding = self.embed_model.encode(
                query, normalize_embeddings=True
            ).tolist() #convert to list for JSON serialization
            
            results = self.vector_store.query( # queries vector store to find k most similar chunks
                VectorStoreQuery(
                    query_embedding=query_embedding,
                    similarity_top_k=self.k
                )
            )
            if results.nodes:
                passages = [node.get_content() for node in results.nodes]
                sources = [{**node.metadata, "text": node.get_content()} for node in results.nodes]
                pairs = [[query, passage] for passage in passages]
                scores = self.rerank_model.compute_score(pairs) # for FlagReranker
                # scores = self.rerank_model.predict(pairs) # SentenceTransformer CrossEncoder
                if isinstance(scores, list):
                    scored_items = list(zip(scores, passages, sources))
                else:
                    scored_items = list(zip([scores], passages, sources))
                
                scored_items.sort(key=lambda x: x[0], reverse=True)
                scored_items = scored_items[:self.rerank_top_k]
                
                # Extract reranked passages and sources
                passages = [item[1] for item in scored_items]
                sources = [item[2] for item in scored_items]
                
            else:
                passages = []
                sources = []
            return dspy.Prediction(passages=passages, sources=sources) # return passages and their metadata
    
class RAG(dspy.Module): # RAG module combining retriever and generator
    def __init__(self, k=TOP_K):
        self.retriever = PGVectorRetriever(k=k) 
        self.generate = dspy.ChainOfThought(GenerateAnswer) # uses DSPY's CoT for generation, has that step by step style 
    def forward(self, question: str): # takes user question
            result = self.retriever(question) # retrieve relevant passages
            context = result.passages # retrieved passages
            sources = result.sources

            answer = self.generate(context=context, question=question)
            answer.sources = sources
            return answer
