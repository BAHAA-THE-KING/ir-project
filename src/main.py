from config import DATASETS, DEFAULT_DATASET
from loader import load_dataset
from services.online_vectorizers.bm25 import bm25_search
from services.offline_vectorizers.bm25 import bm25_train

def main():
    try:
        # Load the default dataset
        dataset_name = DEFAULT_DATASET
        
        # Load the dataset using the loader
        docs, queries, qrels = load_dataset(dataset_name)
        
        # Search using BM25
        bm25_search(dataset_name, queries[1].text, 10)
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")

if __name__ == "__main__":
    main()