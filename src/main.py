from config import DEFAULT_DATASET
from loader import load_dataset_with_queries
from services.online_vectorizers.embedding import Embedding_online
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    

def main():
    docs, queries, qrels = load_dataset_with_queries('quora')
    Embedding_online.evaluate_embedding('quora',queries, qrels)
    

if __name__ == "__main__":
    main()

