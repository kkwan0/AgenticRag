from models import get_lm, get_named_lm
from rag import RAG
from config import TOP_K, LM_TESTS
from dataclasses import dataclass
from questionBank import QUERY_QUESTIONS
@dataclass
class QueryResult():
    answer: str
    sources: list[dict]
    
    def formatted_sources(self) -> str:
        lines = []
        for source in self.sources:
            text = source.get('text', '')
            preview = text[:150] + "..." if len(text) > 150 else text
            lines.append(f"- {source.get('file_name')} | Page: {source.get('page_label')}\n") #  \"{preview}\" for the actual line pulled
        return "\n".join(lines)
        
_rag = None

def _get_rag():
    global _rag
    if _rag is None:
        get_lm()
        _rag = RAG(k=TOP_K)
    return _rag
def query(question: str) -> QueryResult:
    result = _get_rag()(question=question)
    return QueryResult(answer=result.answer, sources=result.sources)

def test_models():
    for model_name in LM_TESTS:
        get_named_lm(model_name)
        rag = RAG(k=TOP_K)
        for question in QUERY_QUESTIONS:
            result = rag(question=question)
            print("========================================")
            print(f"Question: {question}")
            print(f"Answer: {result.answer}")
            print("Sources:")
            for source in result.sources:
                print(f"  - {source.get('file_name')} | Page: {source.get('page_label')}")
            print("========================================\n")
    
   