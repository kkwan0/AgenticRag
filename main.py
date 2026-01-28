from query import query
from ingest import ingest
import argparse
from config import FILE_PATHS
from rag import GenerateAnswer, PGVectorRetriever, RAG
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the vector store by ingesting documents')
    args = parser.parse_args()
    if args.rebuild:
        ingest(FILE_PATHS)
    #already ingested
    answer = query("How do I audit a course?")
    print(answer)
    
    
if __name__ == "__main__":
    main()