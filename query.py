import dspy
from models import get_lm, get_named_lm, print_model_load_times
from rag import RAG
from config import TOP_K, LM_TESTS
from dataclasses import dataclass
from questionBank import QUERY_QUESTIONS
import time
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

def query_and_time(question: str) -> QueryResult:
    start_time = time.time()
    result = query(question)
    end_time = time.time()
    print_model_load_times()
    
    print(f"Time taken for query: {end_time - start_time} seconds")
    return result

def test_models():
    for model_name in LM_TESTS:
        run_id = str(int(time.time()))  # Unique per run
        timelist = []
        print(f"Testing model: {model_name}")
        lm = get_named_lm(model_name)
        dspy.configure(lm=lm)
        rag = RAG(k=TOP_K)
        for question in QUERY_QUESTIONS:
            start_question = time.time()
            unique_question = f"{question} (Run ID: {run_id})"
            result = rag(question=unique_question)
            print("========================================")
            print(f"Model: {model_name}")
            print(f"Question: {question}")
            print(f"Answer: {result.answer}")
            print("Sources:")
            for source in result.sources:
                print(f"  - {source.get('file_name')} | Page: {source.get('page_label')}")
            end_question = time.time()
            timelist.append(end_question - start_question)
            print(f"Time taken for this question: {end_question - start_question} seconds")
            print("========================================\n")
        print(f"Average time for each model: {model_name}")
        print(f"Total time: {sum(timelist)} seconds")
        print(f"Average time per question: {sum(timelist) / len(timelist)} seconds")
        
def test_questions():
    rag = _get_rag()
    run_id = str(int(time.time()))  # Unique per run
    timelist = []
    for question in QUERY_QUESTIONS:
        start_question = time.time()
        unique_question = f"{question} (Run ID: {run_id})"
        result = rag(question=unique_question)
        print("========================================")
        print(f"Question: {question}")
        print(f"Answer: {result.answer}")
        print("Sources:")
        for source in result.sources:
            print(f"  - {source.get('file_name')} | Page: {source.get('page_label')}")
        end_question = time.time()
        timelist.append(end_question - start_question)
        print(f"Time taken for this question: {end_question - start_question} seconds")
        print("========================================\n")
    print(f"Total time: {sum(timelist)} seconds")
    print(f"Average time per question: {sum(timelist) / len(timelist)} seconds")