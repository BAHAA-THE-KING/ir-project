from config import DEFAULT_DATASET
from loader import load_dataset_with_queries
from services.offline_vectorizers.tfidf import tfidf_train
from services.online_vectorizers.bm25 import bm25_search
from services.online_vectorizers.tfidf import tfidf_search
from gui.gui import IRMainWindow
from PyQt6.QtWidgets import QApplication

import math

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def ndcj(reli, rank):
    return ((2** reli)-1)/math.log10(rank+1)

def evaluate_bm25(dataset_name, queries, qrels):
    NDCG = []

    for i in range(len(queries)):
        query = queries[i]
        
        # Search using BM25
        results=bm25_search(dataset_name, query.text, 37, True)

        # Find relevant documents for this query
        relevant_qrels = [qrel for qrel in qrels if qrel.query_id == query.query_id]
        relevant_qrels = sorted(relevant_qrels, key=lambda x: x.relevance, reverse=True)
        
        nDCG = [
            ndcj(
                list(
                    filter(
                        lambda qrel: qrel.doc_id == doc[0], relevant_qrels
                        )
                    )[0].relevance if list(
                    filter(
                        lambda qrel: qrel.doc_id == doc[0], relevant_qrels
                        )
                    ) else 0
                , i+1
            ) for i, doc in enumerate(results)]
        
        iDCG = [ndcj(qrel.relevance, i+1) for i, qrel in enumerate(relevant_qrels)]
        
        res = sum(nDCG) 
        ires = sum(iDCG) 
        
        print("")
        print(f"query: {i}")
        print(f"query: {i}")
        print(f"nDCG: {res}")
        print(f"iDCG: {ires}")
        print(f"NDCG: {res/ires*100}%")
        NDCG.append(res/ires)
    
    print(f"Average NDCG: {sum(NDCG)/len(NDCG)*100}%")
    

def main():
    # Load the default dataset
    dataset_name = DEFAULT_DATASET
       
    # Load the dataset using the loader
    docs, queries, qrels = load_dataset_with_queries(dataset_name)
    
    # Train the TF-IDF Model
    print(f"--- Training TF-IDF Model for '{dataset_name}' ---")
    tfidf_train([docs[0], docs[1], docs[2]], dataset_name)
    print("-" * 50)

        #Search with TF-IDF
    search_query_1 = "Saddam Hussien"
    print(f"\n--- Searching for: '{search_query_1}' ---")
    results_1 = tfidf_search(dataset_name, search_query_1, 10)

    print("\nSearch Results (Ranked by Relevance):")
    for res in results_1:
        print(f"  Doc ID: {res[0]}, Score: {res[1]:.4f}, Text: '{res[2]}'")
    print("-" * 60)

    # Search using BM25
    # results=bm25_search(dataset_name, queries[10].text, 10, True)

    # results_ids = [res[0] for res in results]

    # print(results_ids)

    # # Print query details
    # print("\nQuery:", queries[10])
       
    # # Find relevant documents for this query
    # relevant_docs = [qrel for qrel in qrels if qrel.query_id == queries[10].query_id]
    # print("\nRelevant documents for this query:")
    # for qrel in relevant_docs:
    #     print(f"Doc ID: {qrel.doc_id}, Relevance: {qrel.relevance}")
    
    # ev = [res for res in results_ids if res in [rev.doc_id for rev in relevant_docs]]

    # print(ev)

    # print(f"results: {len(results_ids)}")
    # print(f"qrel: {len(relevant_docs)}")
    # print(f"ev: {len(ev)}")
    

if __name__ == "__main__":
    main()

# app = QApplication(sys.argv)
# window = IRMainWindow()
# window.show()
# sys.exit(app.exec())

