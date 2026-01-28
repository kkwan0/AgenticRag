import dspy
from db import connect_vector_store
from models import get_embed_model, get_lm
from llama_index.core.vector_stores import VectorStoreQuery

class GenerateAnswer(dspy.Signature):
    context = dspy.InputField(desc='retrieved passages from vector store')
    question = dspy.InputField()
    answer = dspy.OutputField()
    
class PGVectorRetriever(dspy.Module):
    def __init__(self, k=3):
        self.vector_store = connect_vector_store()
        self.embed_model = get_embed_model()
        self.k = k
        super().__init__()

    def forward(self, query: str) -> dspy.Prediction:
            query_embedding = self.embed_model.encode(
                query, normalize_embeddings=True
            ).tolist()

            results = self.vector_store.query(
                VectorStoreQuery(
                    query_embedding=query_embedding,
                    similarity_top_k=self.k
                )
            )

            # Step 3: Extract text from results
            passages = [node.get_content() for node in results.nodes] if results.nodes else []
            
            # Step 4: Return as DSPy Prediction
            return dspy.Prediction(passages=passages)
    
class RAG(dspy.Module):
    def __init__(self, k=3):
        self.retriever = PGVectorRetriever(k=k)
        self.generate = dspy.ChainOfThought(GenerateAnswer)
    def forward(self, question: str):
            # Step 1: Retrieve relevant passages
            context = self.retriever(question).passages
            
            # Step 2: Generate answer using LLM
            answer = self.generate(context=context, question=question)
            
            return answer