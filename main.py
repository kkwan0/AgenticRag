from query import query, test_models
from ingest import ingest
import argparse
from config import FILE_PATHS


def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the vector store by ingesting documents')
    # Parse the command-line arguments, add rebuild argument
    args = parser.parse_args()
    if args.rebuild:
        ingest(FILE_PATHS)
        #if flag set to rebuild, ingest documents
    #already ingested
    # answer = query("Can I do a literature review for my signature work?")
    # print(answer.answer)
    # print(answer.formatted_sources())
    test_models()
    
    
if __name__ == "__main__":
    main()