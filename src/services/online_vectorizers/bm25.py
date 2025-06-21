import joblib
from rank_bm25 import BM25Okapi
from src.services.online_vectorizers.inverted_index import InvertedIndex
from src.services.processing.preprocessing import preprocess_text

def bm25_search(dataset_name: str, query: str, top_k: int = 10):
    # Load the model and the documents
    bm25:BM25Okapi = joblib.load(f"data/{dataset_name}/bm25_model.joblib") 
    # docs = joblib.load(f"data/{dataset_name}/docs_list.joblib")
    inverted_index = InvertedIndex.load(dataset_name)

    # Execute the query
    query_tokens = preprocess_text(query)
    documents_sharing_terms_with_query = inverted_index.get_documents_sharing_terms_with_query(query_tokens)
    scores = bm25.get_batch_scores(query_tokens, documents_sharing_terms_with_query)

    print(scores)

    # # Sort the results
    # top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []

    # # Display the results
    # for idx in top_indices:
    #     text = docs[idx].text[:40] + "..." if len(docs[idx].text) > 40 else docs[idx].text
    #     results.append((docs[idx].doc_id, scores[idx], text))
    
    return results