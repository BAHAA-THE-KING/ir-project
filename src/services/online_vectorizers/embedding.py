from sentence_transformers import SentenceTransformer, util
import numpy as np
import joblib
import torch 
from loader import load_dataset
from sentence_transformers import SentenceTransformer


def embedding_search(dataset_name: str, query: str, top_k: int = 10):
   
    print("Loading pre-trained models and data...")
    model_path = 'data/models/all-MiniLM-L6-v2' 
    print(f"Loading Sentence-BERT model from local path: {model_path}...")
    model = SentenceTransformer(model_path)
    
    embedding_path = f"data/{dataset_name}/bert_embeddings.npy"
    document_embeddings = np.load(embedding_path)


    docs = load_dataset(dataset_name)

    print("Encoding query...")
    query_embedding = model.encode(query)

    cos_scores = util.cos_sim(torch.tensor(query_embedding), torch.tensor(document_embeddings))[0]
    top_results = torch.topk(cos_scores, k=top_k)

    results = []
    print(f"\nTop {top_k} results for query: '{query}'")
    for score, idx in zip(top_results[0], top_results[1]):
        doc_id = docs[idx].doc_id
        doc_text = docs[idx].text[:100] + "..." 
        results.append((doc_id, score.item(), doc_text))
        print(f"Doc ID: {doc_id}, Score: {score.item():.4f}, Text: {doc_text}")
    
    return results
