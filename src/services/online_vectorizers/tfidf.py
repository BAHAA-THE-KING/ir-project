import joblib
import numpy as np
from services.processing.preprocessing import TextPreprocessor
from sklearn.metrics.pairwise import cosine_similarity

def tfidf_search(dataset_name, query, top_k):

    # Load the model and the documents
    vectorizer = joblib.load(f"data/{dataset_name}/tfidf_vectorizer.joblib")
    docs = joblib.load(f"data/{dataset_name}/docs_list.joblib")
    docs_tfidf_matrix = joblib.load(f"data/{dataset_name}/tfidf_matrix.joblib")
    
    preprocessor = TextPreprocessor() # Initialize preprocessor for query

    # Preprocess the query using the same preprocessor
    preprocessed_query = preprocessor.preprocess_text(query)
    # Transform the preprocessed query into a TF-IDF vector
    query_vec = vectorizer.transform([preprocessed_query])

    # Calculate cosine similarity between the query vector and the pre-computed document matrix
    cosine_sim = cosine_similarity(query_vec, docs_tfidf_matrix).flatten()

    # Get indices of documents ranked by similarity (highest first)
    ranked_indices = np.argsort(cosine_sim)[::-1]

    # Prepare structured results
    results = []
    # Limit results to a reasonable number for display/API response, e.g., top 10 or 20
    for i in ranked_indices[:top_k]: # Return top 20 results for example
        results.append((
            docs[i].doc_id,
            float(cosine_sim[i]),docs[i].text[:40] + "..." if len(docs[i].text) > 40 else docs[i].text # Provide a snippet
        ))
    return results
