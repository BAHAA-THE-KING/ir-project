import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import joblib
from services.processing.text_preprocessor import TextPreprocessor
import os

def download_bert():
    MODEL_NAME = 'all-MiniLM-L6-v2'
    MODEL_PATH = f'data/models/{MODEL_NAME}'
    if not os.path.exists(MODEL_PATH):
        print(f"Model '{MODEL_NAME}' not found locally. Downloading...")
        model = SentenceTransformer(MODEL_NAME)
        model.save(MODEL_PATH)
        print(f"✅ Model downloaded and saved to '{MODEL_PATH}'")
    else:
        print(f"✅ Model '{MODEL_NAME}' already exists locally at '{MODEL_PATH}'")


def embedding_train(docs, dataset_name):
    preprocessor = TextPreprocessor.getInstance()
    print("Loading Sentence-BERT model for training...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_texts = [" ".join(preprocessor.preprocess_text(doc.text)) for doc in docs]
    print(f"Generating embeddings for {len(doc_texts)} documents...")
    document_embeddings = model.encode(doc_texts, show_progress_bar=True)
    embedding_path = f"data/{dataset_name}/bert_embeddings.npy"
    np.save(embedding_path, document_embeddings)
    docs_list_path = f"data/{dataset_name}/embedding_docs.joblib"
    joblib.dump(docs, docs_list_path)
    print(f"Embeddings model generated and saved to {embedding_path}")
    print(f"Documents list saved to {docs_list_path}")



def populate_vector_store(docs, dataset_name):
    print("--- Starting to Populate Vector Store ---")
    model_path = 'data/models/all-MiniLM-L6-v2'
    print(f"Loading model from: {model_path}...")
    model = SentenceTransformer(model_path)
    preprocessor = TextPreprocessor.getInstance()
    print("Preprocessing texts...")
    doc_texts = [" ".join(preprocessor.preprocess_text(doc.text)) for doc in docs]

    print("Generating embeddings")
    embeddings = model.encode(doc_texts, show_progress_bar=True)

    client = chromadb.PersistentClient(path="chroma_db")
    
    collection = client.get_or_create_collection(name=f"{dataset_name}_embeddings")
    
    doc_ids = [doc.doc_id for doc in docs]
    metadatas = [{'text': doc.text} for doc in docs]

    batch_size = 5000
    print(f"Adding {len(doc_ids)} documents to Chroma collection in batches of {batch_size}...")
    for i in range(0, len(doc_ids), batch_size):
        collection.add(
            ids=doc_ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size].tolist(), 
            metadatas=metadatas[i:i+batch_size]
        )
        print(f"Added batch {i//batch_size + 1}...")

    print("Vector store populated successfully!")
    print(f"Total items in collection: {collection.count()}")