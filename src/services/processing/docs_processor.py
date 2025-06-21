from preprocessing import preprocess_text
import joblib
import os

def process_docs(docs, dataset_name):
    tokenized_docs = [preprocess_text(doc.text) for doc in docs]
    
    path = f"data/{dataset_name}/docs_list.joblib"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'a').close()
    joblib.dump(tokenized_docs, path)