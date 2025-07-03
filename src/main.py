import time
from config import DEFAULT_DATASET
from loader import load_dataset_with_queries
from services.online_vectorizers.bm25 import BM25_online
from services.online_vectorizers.tfidf import TFIDF_online
from services.processing.preprocessing import preprocess_text

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    

def main():
    dataset_name = 'quora'
    docs, queries, qrels = load_dataset_with_queries(dataset_name)
    retriever = TFIDF_online()

    preprocess_text("hbd")
    retriever.__loadInstance__(dataset_name)
    retriever.__loadInvertedIndex__(dataset_name)
    print("search started")
    
    retriever.evaluateNDCG(dataset_name, queries, qrels, docs, 10)
    # retriever.evaluateMAP()

    start_time = time.time()
    results = retriever.search(dataset_name, "saddam", 10, True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Function execution time: {elapsed_time:.4f} seconds")

    start_time = time.time()
    results = retriever.search(dataset_name, "politicians", 10, True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Function execution time: {elapsed_time:.4f} seconds")


    start_time = time.time()
    results = retriever.search(dataset_name, "please don't", 10)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Function execution time: {elapsed_time:.4f} seconds")


    print("\nSearch Results (Ranked by Relevance):")
    for res in results:
        print(f"  Doc ID: {res[0]}, Score: {res[1]:.4f}, Text: '{res[2]}'")
    print("-" * 60)

    
    retriever = BM25_online()
    MRR = retriever.evaluateMRR(dataset_name, queries, qrels)
    MAP = retriever.evaluateMAP(dataset_name, queries, qrels)

    print(f"MAP={MAP}")
    print(f"MRR={MRR}")
    
if __name__ == "__main__":
    main()

