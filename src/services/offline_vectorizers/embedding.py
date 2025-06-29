from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
from services.processing.preprocessing import TextPreprocessor
import os

def download_bert():
    MODEL_NAME = 'all-MiniLM-L6-v2'
    MODEL_PATH = f'models/{MODEL_NAME}'
    if not os.path.exists(MODEL_PATH):
        print(f"Model '{MODEL_NAME}' not found locally. Downloading...")
        model = SentenceTransformer(MODEL_NAME)
        model.save(MODEL_PATH)
        print(f"✅ Model downloaded and saved to '{MODEL_PATH}'")
    else:
        print(f"✅ Model '{MODEL_NAME}' already exists locally at '{MODEL_PATH}'")


def embedding_train(docs, dataset_name):
    preprocessor = TextPreprocessor()
    print("Loading Sentence-BERT model for training...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_texts = [preprocessor.preprocess_text(doc.text) for doc in docs]
    print(f"Generating embeddings for {len(doc_texts)} documents...")
    document_embeddings = model.encode(doc_texts, show_progress_bar=True)
    embedding_path = f"data/{dataset_name}/bert_embeddings.npy"
    np.save(embedding_path, document_embeddings)
    docs_list_path = f"data/{dataset_name}/embedding_docs.joblib"
    joblib.dump(docs, docs_list_path)
    print(f"Embeddings model generated and saved to {embedding_path}")
    print(f"Documents list saved to {docs_list_path}")