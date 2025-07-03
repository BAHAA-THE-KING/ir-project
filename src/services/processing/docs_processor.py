import os
import dill
from .text_preprocessor import TextPreprocessor

def process_docs(docs, dataset_name):
    tokenized_docs = [TextPreprocessor.getInstance().preprocess_text(doc.text) for doc in docs]
    
    path = f"data/{dataset_name}/docs_list.dill"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'a').close()
    with open(path, "wb") as f:
        dill.dump(tokenized_docs, f)