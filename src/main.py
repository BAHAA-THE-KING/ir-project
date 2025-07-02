import time
import dill
from config import DEFAULT_DATASET
from loader import load_dataset_with_queries
# from services.online_vectorizers.embedding import Embedding_online
from services.processing.preprocessing import preprocess_text

import sys
import os
from services.online_vectorizers.tfidf import tfidf_search
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    

def main():
    dataset_name = DEFAULT_DATASET

    docs, queries, qrels = load_dataset_with_queries('antique')
    
        #Search with TF-IDF
    search_query_1 = "Because it is run by politicians."
    print(f"\n--- Searching for: '{search_query_1}' ---")

    start_time = time.time()
    results_1 = tfidf_search(dataset_name, search_query_1, 10, True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Function execution time: {elapsed_time:.4f} seconds")
    
    print("\nSearch Results (Ranked by Relevance):")
    for res in results_1:
        print(f"  Doc ID: {res[0]}, Score: {res[1]:.4f}, Text: '{res[2]}'")
    print("-" * 60)


    docs, queries, qrels = load_dataset_with_queries('quora')
    Embedding_online.evaluate_embedding('quora',queries, qrels)
    

if __name__ == "__main__":
    main()

