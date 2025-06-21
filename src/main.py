from config import DATASETS, DEFAULT_DATASET
from loader import load_dataset
from services.offline_vectorizers.tfidf import tfidf_train
from services.online_vectorizers.tfidf import tfidf_search
from services.online_vectorizers.bm25 import bm25_search
from services.offline_vectorizers.bm25 import bm25_train
from services.processing.preprocessing import TextPreprocessor

def main():
    # Load the default dataset
    dataset_name = DEFAULT_DATASET
       
    # Load the dataset using the loader
    docs, queries, qrels = load_dataset(dataset_name)

    document=docs[0]
    text=document.text
    processedText = TextPreprocessor().preprocess_text(text)

    # Train the TF-IDF Model
    print(f"--- Training TF-IDF Model for '{dataset_name}' ---")
    tfidf_train([document], dataset_name)
    print("-" * 50)
        

    #Search with TF-IDF
    search_query_1 = "Saddam Hussien politician"
    print(f"\n--- Searching for: '{search_query_1}' ---")
    results_1 = tfidf_search(search_query_1, dataset_name, 10)

    print("\nSearch Results (Ranked by Relevance):")
    for res in results_1:
        print(f"  Doc ID: {res['doc_id']}, Score: {res['score']:.4f}, Text: '{res['text_snippet']}'")
    print("-" * 60)


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