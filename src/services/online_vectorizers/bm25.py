import dill
from rank_bm25 import BM25Okapi
from services.online_vectorizers.inverted_index import InvertedIndex
from services.processing.preprocessing import preprocess_text

def bm25_search(dataset_name: str, query: str, top_k: int = 10):
    # Load the model and the documents
    with open(f"data/{dataset_name}/bm25_model.dill", "rb") as f:
        bm25:BM25Okapi = dill.load(f) 
    # docs = dill.load(f"data/{dataset_name}/docs_list.dill")
    with open(f"data/{dataset_name}/inverted_index.dill", "rb") as f:
        inverted_index:InvertedIndex = dill.load(f)
        print(inverted_index.get_documents_sharing_terms_with_query.__code__)

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