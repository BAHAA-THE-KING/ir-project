from config import DEFAULT_DATASET
from loader import load_dataset_with_queries, load_queries_and_qrels
from services.offline_vectorizers.tfidf import tfidf_train
from services.online_vectorizers.tfidf import tfidf_search
from services.online_vectorizers.bm25 import bm25_search, evaluate_bm25
from gui.gui import IRMainWindow
from PyQt6.QtWidgets import QApplication

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    

def main():
    docs, queries, qrels = load_dataset_with_queries('quora')
    evaluate_bm25('quora',queries, qrels)
    

if __name__ == "__main__":
    main()

# app = QApplication(sys.argv)
# window = IRMainWindow()
# window.show()
# sys.exit(app.exec())

