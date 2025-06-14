from config import DATASETS, DEFAULT_DATASET
from loader import load_dataset

def main():
    try:
        # Load the default dataset
        dataset_name = DEFAULT_DATASET
        print(f"Loading dataset: {dataset_name}")
        print(f"Description: {DATASETS[dataset_name]['description']}")
        
        # Load the dataset using the loader
        docs, queries, qrels = load_dataset(dataset_name)
        
        # Print some basic statistics
        print(f"\nDataset Statistics:")
        print(f"Number of documents: {len(docs)}")
        print(f"Number of queries: {len(queries)}")
        print(f"Number of relevance judgments: {len(qrels)}")
        
        # Print a sample document
        if docs:
            print("\nSample Document:")
            print(docs[1])
            
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")

if __name__ == "__main__":
    main()