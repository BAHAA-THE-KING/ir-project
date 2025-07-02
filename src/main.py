from config import DEFAULT_DATASET
from loader import load_dataset_with_queries
from services.online_vectorizers.bm25 import BM25_online

import sys
import os
from services.online_vectorizers.tfidf import tfidf_search
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    

def main():
    dataset_name = DEFAULT_DATASET
    docs, queries, qrels = load_dataset_with_queries(dataset_name)
    retriever = BM25_online()
    retriever.evaluateNDCG(dataset_name, queries, qrels, docs)
    

if __name__ == "__main__":
    main()

