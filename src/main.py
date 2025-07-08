import time
from config import DEFAULT_DATASET
from loader import load_dataset_with_queries, load_dataset
from services.online_vectorizers.bm25 import BM25_online
from services.offline_vectorizers.bm25 import BM25_offline
from services.online_vectorizers.inverted_index import InvertedIndex
from services.processing.text_preprocessor import TextPreprocessor
import dill
import time

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    

def main():
    dataset_name = DEFAULT_DATASET
    docs, queries, qrels = load_dataset_with_queries(dataset_name)
    print('Dataset loaded')
    # st = time.time()
    # BM25_offline().bm25_train(docs, dataset_name)
    # print(f"{time.time() - st}s")

    # st = time.time()
    # index = InvertedIndex()
    # for i, doc in enumerate([TextPreprocessor.getInstance().preprocess_text(doc.text) for doc in docs]):
    #     index.add_document(i, doc)
    # with open('data/antique/inverted_index.dill', 'wb') as f:
    #     dill.dump(index, f)
    # print(f"{time.time() - st}s")
    
    # st = time.time()
    # print(BM25_online().evaluateMAP(dataset_name, queries, qrels, docs))
    # print(f"{time.time() - st}s")

    # st = time.time()
    # print(BM25_online().evaluateMRR(dataset_name, queries, qrels))
    # print(f"{time.time() - st}s")

    # st = time.time()
    # print(BM25_online().evaluateNDCG(dataset_name, queries, qrels, docs))
    # print(f"{time.time() - st}s")

    BM25_online().search(dataset_name, "hi man, how are you ?")

if __name__ == "__main__":
    main()

