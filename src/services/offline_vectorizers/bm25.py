from rank_bm25 import BM25Okapi
import joblib
from services.processing.preprocessing import TextPreprocessor

def bm25_train(docs, dataset_name):
    preprocessor = TextPreprocessor()

    # Train the model
    bm25 = BM25Okapi([doc.text.split() for doc in docs], tokenizer=preprocessor.preprocess_text)
    

    # Save the model and the documents
    joblib.dump(bm25, f"data/{dataset_name}/bm25_model.joblib")

    print("✅ Model trained and saved")