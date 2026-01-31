from query import query, query_and_time, test_models
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
    answer = query_and_time("What are the graduation requirements for a Bachelor of Science degree?")
    print(answer.answer)
    # print(answer.formatted_sources())
    
    
    
    
    # test_models()
    
    
if __name__ == "__main__":
    main()