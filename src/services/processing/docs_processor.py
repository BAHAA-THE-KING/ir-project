from preprocessing import preprocess_text
import joblib

def process_docs(docs, dataset_name):
    tokenized_docs = [preprocess_text(doc.text).split() for doc in docs]
    joblib.dump(tokenized_docs, f"data/{dataset_name}/docs_list.joblib")