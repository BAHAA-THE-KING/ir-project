from config import DEFAULT_DATASET
from loader import load_queries_and_qrels
from services.online_vectorizers.bm25 import bm25_search
from gui.gui import IRMainWindow
from PyQt6.QtWidgets import QApplication

import math

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def ndcj(reli, rank):
    return ((2** reli)-1)/math.log10(rank+1)

def main():
    # Load the default dataset
    dataset_name = DEFAULT_DATASET
       
    # Load the dataset using the loader
    queries, qrels = load_queries_and_qrels(dataset_name)
    
    # Search using BM25
    results=bm25_search(dataset_name, queries[30].text, 37, True)

    results_ids = [res[0] for res in results]

    print(results_ids)

    # Print query details
    print("\nQuery:", queries[30])
       
    # Find relevant documents for this query
    relevant_docs = [qrel for qrel in qrels if qrel.query_id == queries[30].query_id]
    print("\nRelevant documents for this query:")
    for qrel in relevant_docs:
        print(f"Doc ID: {qrel.doc_id}, Relevance: {qrel.relevance}")
    
    nDCG = [
        ndcj(
            list(
                filter(
                    lambda qrel: qrel[1] == id, relevant_docs
                    )
                )[0][2]
            if list(
                filter(
                    lambda qrel: qrel[1] == id, relevant_docs
                    )
                )
            else 0
            , i+1
        ) for i, (id, score, text) in enumerate(results)]
    
    iDCG = [
        ndcj(
            relevance
            , i+1
        ) for i, (query_id, doc_id, relevance, iteration) in enumerate(sorted(relevant_docs, key=lambda x: x.relevance, reverse=True))]
    
    res = sum(nDCG) 
    ires = sum(iDCG) 
    
    print(f"nDCG: {res}")
    print(f"iDCG: {ires}")

    print(f"NDCG: {res/ires}")
   

if __name__ == "__main__":
    main()

# app = QApplication(sys.argv)
# window = IRMainWindow()
# window.show()
# sys.exit(app.exec())

