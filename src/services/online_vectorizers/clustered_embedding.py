# src/services/online_vectorizers/clustered_embedding.py
import os
import joblib
import numpy as np
import torch # Keep torch import for cosine_similarity if needed for sparse operations or conversion
from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer, util # REMOVED

# Ensure correct path for imports
import sys
from pathlib import Path

# Assuming this script is in src/services/online_vectorizers/
# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.services.online_vectorizers.Retriever import Retriever
from src.services.processing.text_preprocessor import TextPreprocessor
from src.loader import load_dataset # Import load_dataset here for snippet retrieval

class ClusteredEmbedding_online(Retriever):
    # Static/class variables for model/data caching
    _loaded = False
    _tfidf_vectorizer = None
    _svd_model = None
    _svd_doc_matrix = None
    _kmeans_model = None
    _kmeans_labels = None
    _cluster_to_doc_indices = None
    _doc_id_to_idx_map = None
    _docs_cache = None

    @staticmethod
    def _load_models(dataset_name: str):
        if ClusteredEmbedding_online._loaded:
            return
        data_base_dir = Path(project_root) / 'data' / dataset_name
        # Load TF-IDF vectorizer
        tfidf_vectorizer_path = data_base_dir / 'tfidf_vectorizer.joblib'
        ClusteredEmbedding_online._tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        # Load SVD model
        svd_model_path = data_base_dir / 'svd_model_1000_components.joblib'
        ClusteredEmbedding_online._svd_model = joblib.load(svd_model_path)
        # Load SVD-reduced doc matrix
        svd_doc_matrix_path = data_base_dir / 'tfidf_svd_1000_components_data.joblib'
        ClusteredEmbedding_online._svd_doc_matrix = joblib.load(svd_doc_matrix_path)
        # Load KMeans model
        kmeans_model_path = data_base_dir / 'kmeans_model_1000.joblib'
        ClusteredEmbedding_online._kmeans_model = joblib.load(kmeans_model_path)
        # Load KMeans labels
        kmeans_labels_path = data_base_dir / 'kmeans_labels_1000.joblib'
        ClusteredEmbedding_online._kmeans_labels = joblib.load(kmeans_labels_path)
        # Build cluster-to-docs index
        labels = ClusteredEmbedding_online._kmeans_labels
        cluster_to_doc_indices = {}
        for idx, label in enumerate(labels):
            cluster_to_doc_indices.setdefault(label, []).append(idx)
        ClusteredEmbedding_online._cluster_to_doc_indices = cluster_to_doc_indices
        # Print cluster distribution
        print("Cluster distribution (cluster_id: num_docs):")
        for cluster_id in sorted(cluster_to_doc_indices.keys()):
            print(f"  Cluster {cluster_id}: {len(cluster_to_doc_indices[cluster_id])} docs")
        # Load docs and build doc_id to index map
        docs = load_dataset(dataset_name)
        ClusteredEmbedding_online._docs_cache = docs
        ClusteredEmbedding_online._doc_id_to_idx_map = {doc.doc_id: i for i, doc in enumerate(docs)}
        ClusteredEmbedding_online._loaded = True

    def search(self, dataset_name: str, query: str, top_k: int = 10, with_index: bool = False):
        ClusteredEmbedding_online._load_models(dataset_name)
        tfidf_vectorizer = ClusteredEmbedding_online._tfidf_vectorizer
        svd_model = ClusteredEmbedding_online._svd_model
        svd_doc_matrix = ClusteredEmbedding_online._svd_doc_matrix
        kmeans_model = ClusteredEmbedding_online._kmeans_model
        cluster_to_doc_indices = ClusteredEmbedding_online._cluster_to_doc_indices
        docs = ClusteredEmbedding_online._docs_cache
        # Preprocess and vectorize query
        processed_query = TextPreprocessor.getInstance().preprocess_text(query)
        query_tfidf = tfidf_vectorizer.transform([processed_query])
        query_svd = svd_model.transform(query_tfidf)
        # Assign query to nearest cluster
        cluster_id = kmeans_model.predict(query_svd)[0]
        doc_indices = cluster_to_doc_indices.get(cluster_id, [])
        if not doc_indices:
            print(f"No documents in assigned cluster {cluster_id}. Returning no results.")
            return []
        # Compute cosine similarity in SVD space
        doc_vectors = svd_doc_matrix[doc_indices]
        scores = cosine_similarity(query_svd, doc_vectors)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            doc_idx = doc_indices[idx]
            doc_obj = docs[doc_idx]
            results.append((doc_obj.doc_id, scores[idx], doc_obj.text[:100] + "..." if len(doc_obj.text) > 100 else doc_obj.text))
        return results