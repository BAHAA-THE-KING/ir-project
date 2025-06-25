import dill
from rank_bm25 import BM25Okapi
from services.online_vectorizers.inverted_index import InvertedIndex
from services.processing.preprocessing import preprocess_text
from loader import load_dataset

def bm25_search(dataset_name: str, query: str, top_k: int = 10, with_inverted_index: bool = False) -> list[tuple[int, float, str]]:
    # Load the model and the documents
    with open(f"data/{dataset_name}/bm25_model.dill", "rb") as f:
        bm25 : BM25Okapi = dill.load(f) 
    # with open(f"data/{dataset_name}/docs_list.dill", "rb") as f:
    #     docs = dill.load(f)
    docs=load_dataset('antique')
    if with_inverted_index:
        with open(f"data/{dataset_name}/inverted_index.dill", "rb") as f:
            inverted_index = InvertedIndex()
            ii = dill.load(f)
            inverted_index.index = ii.index
            inverted_index.doc_lengths = ii.doc_lengths
            inverted_index.N = ii.N

    # Execute the query
    query_tokens = preprocess_text(query)
    if with_inverted_index:
        documents_sharing_terms_with_query = inverted_index.get_documents_sharing_terms_with_query(query_tokens)
        scores = bm25.get_batch_scores(query_tokens, documents_sharing_terms_with_query)
    else:
        scores = bm25.get_scores(query_tokens)

    # Sort the results
    if with_inverted_index:
        top_indices = sorted(list(enumerate(documents_sharing_terms_with_query)), key=lambda  elm: scores[elm[0]], reverse=True)[:top_k]
    else:
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []

    # Display the results
    if with_inverted_index:
        for elm in top_indices:
            text = docs[elm[1]].text[:40] + "..." if len(docs[elm[1]].text) > 40 else docs[elm[1]].text
            results.append((docs[elm[1]].doc_id, scores[elm[0]], text))
    else:
        for idx in top_indices:
            text = docs[idx].text[:40] + "..." if len(docs[idx].text) > 40 else docs[idx].text
            results.append((docs[idx].doc_id, scores[idx], text))
    
    return results