from config import DEFAULT_DATASET
from loader import load_queries_and_qrels
from services.online_vectorizers.bm25 import bm25_search

def main():
    # Load the default dataset
    dataset_name = DEFAULT_DATASET
       
    # Load the dataset using the loader
    queries, qrels = load_queries_and_qrels(dataset_name)
    
    # Search using BM25
    bm25_search(dataset_name, queries[0].text, 10)
       
    # # Print query details
    # print("\nQuery:", queries[2])
       
    # # Find relevant documents for this query
    # relevant_docs = [qrel for qrel in qrels if qrel.query_id == queries[2].query_id]
    # print("\nRelevant documents for this query:")
    # for qrel in relevant_docs:
    #     print(f"Doc ID: {qrel.doc_id}, Relevance: {qrel.relevance}")
           
    

if __name__ == "__main__":
    main()