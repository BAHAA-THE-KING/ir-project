from rank_bm25 import BM25Okapi
import joblib

def bm25_train(docs, dataset_name):
    tokenized_docs = [doc.text.lower().split() for doc in docs]

    # Train the model
    bm25 = BM25Okapi(tokenized_docs)

    # Save the model and the documents
    joblib.dump(bm25, f"data/{dataset_name}/bm25_model.joblib")
    joblib.dump(docs, f"data/{dataset_name}/docs_list.joblib")

    print("✅ Model trained and saved")