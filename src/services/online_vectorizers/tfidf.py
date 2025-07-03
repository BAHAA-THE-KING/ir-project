import time
import dill
import joblib
import numpy as np
from src.services.processing.text_preprocessor import TextPreprocessor
from sklearn.metrics.pairwise import cosine_similarity

from services.online_vectorizers.inverted_index import InvertedIndex

def tfidf_search(dataset_name, query, top_k, with_index = False):

    # Load the model and the documents
    vectorizer = joblib.load(f"data/{dataset_name}/tfidf_vectorizer.joblib")
    docs = joblib.load(f"data/{dataset_name}/docs_list.joblib")
    docs_tfidf_matrix = joblib.load(f"data/{dataset_name}/tfidf_matrix.joblib")
    
    with open(f"data/{dataset_name}/inverted_index_tfidf.dill", "rb") as f:
                inverted_index = InvertedIndex()
                ii = dill.load(f)
                inverted_index.index = ii.index
                inverted_index.doc_lengths = ii.doc_lengths
                inverted_index.N = ii.N

    # Transform the preprocessed query into a TF-IDF vector
    query_vec = vectorizer.transform([query])
    
    tokenized_query = TextPreprocessor.getInstance().preprocess_text(query)
    
    if(with_index):
        candidate_indices = inverted_index.get_documents_sharing_terms_with_query(
            tokenized_query
        )   
        candidate_tfidf_matrix = docs_tfidf_matrix[candidate_indices]
        cosine_sim = cosine_similarity(query_vec, candidate_tfidf_matrix).flatten()
    else:
        cosine_sim = cosine_similarity(query_vec, docs_tfidf_matrix).flatten()

    ranked_indices = np.argsort(cosine_sim)[::-1]

    # Prepare structured results
    results = []
    # Limit results to a reasonable number for display/API response, e.g., top 10 or 20
    for i in ranked_indices[:top_k]:
        if(with_index):
            original_doc_idx = candidate_indices[i]
        else:
            original_doc_idx = i

        doc = docs[original_doc_idx]
        results.append((
            docs[original_doc_idx].doc_id,
            float(cosine_sim[i]),
            doc.text[:40] + "..." if len(doc.text) > 40 else doc.text # Provide a snippet
        ))
    return results
