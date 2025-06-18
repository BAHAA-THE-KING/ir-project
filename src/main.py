from config import DATASETS, DEFAULT_DATASET
from loader import load_dataset,load_dataset_with_queries
from services.online_vectorizers.bm25 import bm25_search
from services.offline_vectorizers.bm25 import bm25_train
from services.processing.preprocessing import TextPreprocessor

def main():
    # Load the default dataset
    dataset_name = DEFAULT_DATASET
       
    # Load the dataset using the loader
    docs, queries, qrels = load_dataset_with_queries(dataset_name)

    document=docs[0]
    text=document.text
    processedText = TextPreprocessor().preprocess_text(text)

    print(processedText)
       
    # Search using BM25
    # bm25_search(dataset_name, queries[2].text, 10)
       
    # # Print query details
    # print("\nQuery:", queries[2])
       
    # # Find relevant documents for this query
    # relevant_docs = [qrel for qrel in qrels if qrel.query_id == queries[2].query_id]
    # print("\nRelevant documents for this query:")
    # for qrel in relevant_docs:
    #     print(f"Doc ID: {qrel.doc_id}, Relevance: {qrel.relevance}")
           
    

if __name__ == "__main__":
    main()