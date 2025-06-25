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
    queries, qrels = load_queries_and_qrels(dataset_name)

    evaluate_bm25(dataset_name, queries, qrels)

if __name__ == "__main__":
    main()

# app = QApplication(sys.argv)
# window = IRMainWindow()
# window.show()
# sys.exit(app.exec())

