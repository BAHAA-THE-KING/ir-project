from rank_bm25 import BM25Okapi
import joblib
from services.processing.preprocessing import TextPreprocessor

def bm25_search(dataset_name: str, query: str, top_k: int = 10):
    # Load the model and the documents
    bm25:BM25Okapi = joblib.load(f"data/{dataset_name}/bm25_model.joblib") 
    docs = joblib.load(f"data/{dataset_name}/docs_list.joblib")

    # Execute the query
    preprocessor = TextPreprocessor()
    query_tokens = preprocessor.preprocess_text(query).split()
    scores = bm25.get_scores(query_tokens)

    # Sort the results
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []

    # Display the results
    for idx in top_indices:
        text = docs[idx].text[:40] + "..." if len(docs[idx].text) > 40 else docs[idx].text
        results.append((docs[idx].doc_id, scores[idx], text))
    
    return results