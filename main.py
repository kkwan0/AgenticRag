import time
from query import query, query_and_time, test_models, test_questions
from ingest import ingest
import argparse
from config import FILE_PATHS


def main():
    end_to_end_latency_start = time.time()
    parser = argparse.ArgumentParser() 
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the vector store by ingesting documents')
    # Parse the command-line arguments, add rebuild argument
    args = parser.parse_args()
    if args.rebuild:
        ingest(FILE_PATHS)
        #if flag set to rebuild, ingest documents
    #already ingested
    # answer = query("Can I do a literature review for my signature work?")
    # run_id = str(int(time.time()))  # Unique per run
    # question = "How can I find the campus events calendar?"
    # unique_question = f"{question} (Run ID: {run_id})"
    # answer = query_and_time(unique_question) # gets query time
    # print(answer.answer)
    # print(answer.formatted_sources())
    test_questions()
    end_to_end_latency = time.time() - end_to_end_latency_start
    print(f"End-to-end latency: {end_to_end_latency} seconds")
    # test_models()
    
    
if __name__ == "__main__":
    main()