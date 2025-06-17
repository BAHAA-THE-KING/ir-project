from preprocessing import TextPreprocessor
import joblib

def process_docs(docs, dataset_name):
    preprocessor = TextPreprocessor()
    tokenized_docs = [preprocessor.preprocess_text(doc.text).split() for doc in docs]
    joblib.dump(tokenized_docs, f"data/{dataset_name}/docs_list.joblib")