from query import query
from ingest import ingest
import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the vector store by ingesting documents')
    args = parser.parse_args()
    if args.rebuild:
        ingest(["/mnt/c/Users/freen/Documents/AgenticRag/supportingDocuments/ug_bulletin2025-26.pdf", "/mnt/c/Users/freen/Documents/AgenticRag/supportingDocuments/ug_bulletin2024-25.pdf", "/mnt/c/Users/freen/Documents/AgenticRag/supportingDocuments/ug_bulletin2023-24.pdf"])
    #already ingested
    answer = query("How do I audit a course?")
    print(answer)
    
    
if __name__ == "__main__":
    main()