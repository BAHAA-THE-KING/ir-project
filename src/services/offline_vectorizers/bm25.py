import os
import joblib
from rank_bm25 import BM25Okapi
from src.services.processing.preprocessing import preprocess_text

def bm25_train(docs, dataset_name):
    # Train the model
    bm25 = BM25Okapi([doc.text for doc in docs], tokenizer=preprocess_text)

    # Save the model and the documents
    path = f"data/{dataset_name}/bm25_model.joblib"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'a').close()
    joblib.dump(bm25, path)

    print("✅ Model trained and saved")