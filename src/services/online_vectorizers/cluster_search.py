# import os
# import joblib
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from services.processing.text_preprocessor import TextPreprocessor
# from services.online_vectorizers.Retriever import Retriever
# from sklearn.decomposition import TruncatedSVD


# class ClusterSearch(Retriever):
#     def __init__(self, model_path, dataset_name="antique", kmeans_model_k_value: int = 8): # Added kmeans_model_k_value as a parameter with default
#         self.model_path = model_path
#         self.dataset_name = dataset_name 
#         self.kmeans_model_k_value = kmeans_model_k_value # Initialize the attribute here FIRST

#         self.tfidf_vectorizer = None
#         self.svd_model = None
#         self.kmeans_model = None
#         self.docs_list = None
#         self.original_docs_by_id = {}
#         self.cluster_labels = None
#         self.docs_tfidf_matrix_reduced = None 
#         self.models_loaded = False

#         # Define full paths to the model files based on the provided model_path
#         tfidf_vectorizer_path = os.path.join(self.model_path, "tfidf_vectorizer.joblib")
#         svd_model_path = os.path.join(self.model_path, "svd_model.joblib")
#         # Now self.kmeans_model_k_value is defined before use
#         kmeans_model_path = os.path.join(self.model_path, f"kmeans_model_k{self.kmeans_model_k_value}.joblib") 
#         docs_list_path = os.path.join(self.model_path, "docs_list.joblib")
#         cluster_labels_path = os.path.join(self.model_path, f"document_cluster_labels_k{self.kmeans_model_k_value}.joblib") 
        
#         tfidf_matrix_full_path = os.path.join(self.model_path, "tfidf_matrix.joblib")


#         print(f"Attempting to load models from: {self.model_path}")

#         try:
#             self.tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
#             print("✅ Loaded tfidf_vectorizer.joblib")

#             self.svd_model = joblib.load(svd_model_path)
#             print("✅ Loaded svd_model.joblib")

#             full_tfidf_matrix = joblib.load(tfidf_matrix_full_path)
#             print("✅ Loaded tfidf_matrix.joblib (full matrix)")
            
#             # Apply SVD transformation to the full TF-IDF matrix
#             self.docs_tfidf_matrix_reduced = self.svd_model.transform(full_tfidf_matrix)
#             print(f"✅ Applied SVD to full TF-IDF matrix. Reduced shape: {self.docs_tfidf_matrix_reduced.shape}")

#             self.docs_list = joblib.load(docs_list_path)
#             self.original_docs_by_id = {doc.doc_id: doc for doc in self.docs_list}
#             print("✅ Loaded docs_list.joblib")

#             self.kmeans_model = joblib.load(kmeans_model_path)
#             print(f"✅ Loaded kmeans_model_k{self.kmeans_model_k_value}.joblib")
            
#             self.cluster_labels = joblib.load(cluster_labels_path)
#             print(f"✅ Loaded document_cluster_labels_k{self.kmeans_model_k_value}.joblib")


#             self.models_loaded = True 
#             print("✅ All models loaded successfully.")

#         except FileNotFoundError as e:
#             print(f"❌ Error loading model file: {e}. Please ensure all required joblib files are in the specified path.")
#             self.models_loaded = False 
#         except Exception as e:
#             print(f"❌ An unexpected error occurred during model loading: {e}")
#             self.models_loaded = False 

#         if not self.models_loaded:
#             print("⚠️ Models not loaded. Search functionality will be unavailable.")


#     def preprocess_query_text(self, query: str) -> str:
#         """
#         Preprocesses the input query string using TextPreprocessor.
#         Returns a single string of processed tokens.
#         """
#         try:
#             preprocessor_instance = TextPreprocessor.getInstance()
#             processed_tokens_list = preprocessor_instance.preprocess_text(query)
#             return " ".join(processed_tokens_list)
#         except NameError:
#             print("❌ Error: TextPreprocessor class not found. Please ensure it is defined and accessible.")
#             return ""


#     def vectorize_and_reduce_query(self, preprocessed_query_string: str) -> np.ndarray | None:
#         """
#         Vectorizes the preprocessed query string using the loaded TF-IDF vectorizer
#         and applies the loaded SVD model for dimensionality reduction.
#         """
#         if not self.models_loaded or self.tfidf_vectorizer is None or self.svd_model is None:
#             print("⚠️ TF-IDF vectorizer or SVD model not loaded. Cannot vectorize and reduce query.")
#             return None

#         tfidf_query_vector = self.tfidf_vectorizer.transform([preprocessed_query_string])
#         reduced_query_vector = self.svd_model.transform(tfidf_query_vector)

#         return reduced_query_vector

#     def predict_query_cluster(self, reduced_query_vector: np.ndarray) -> int | None:
#         """
#         Predicts the cluster the query belongs to using the loaded KMeans model.
#         """
#         if not self.models_loaded or self.kmeans_model is None:
#             print("⚠️ KMeans model not loaded. Cannot predict query cluster.")
#             return None

#         predicted_cluster = self.kmeans_model.predict(reduced_query_vector)
#         return predicted_cluster[0]

#     def retrieve_documents_from_clusters(self, cluster_ids: int | list) -> list:
#         """
#         Retrieves indices of documents that belong to the specified cluster ID(s).
#         Returns a list of integer indices into the original docs_list/tfidf_matrix.
#         """
#         if self.docs_list is None or self.cluster_labels is None:
#             print("⚠️ Document list or cluster labels not loaded. Cannot retrieve documents.")
#             return []

#         if isinstance(cluster_ids, int):
#             cluster_ids_list = [cluster_ids]
#         elif isinstance(cluster_ids, (list, tuple)):
#             cluster_ids_list = cluster_ids
#         else:
#             print("⚠️ Invalid input for cluster_ids. Please provide an integer or a list/tuple of integers.")
#             return []

#         cluster_labels_np = np.array(self.cluster_labels)
#         document_indices_mask = np.isin(cluster_labels_np, cluster_ids_list)
#         document_indices = np.where(document_indices_mask)[0]

#         return document_indices

#     def rank_documents(self, retrieved_doc_indices: list, preprocessed_query_string: str, top_k: int) -> list:
#         """
#         Ranks a list of retrieved documents based on their relevance to the preprocessed query.
#         Uses the pre-loaded reduced TF-IDF matrix for efficiency.
#         Returns a list of (score, doc) tuples.
#         """
#         if self.tfidf_vectorizer is None or self.docs_tfidf_matrix_reduced is None:
#             print("⚠️ TF-IDF vectorizer or reduced document matrix not loaded. Cannot rank documents.")
#             return []

#         if not retrieved_doc_indices:
#             return []
        
#         retrieved_vectors = self.docs_tfidf_matrix_reduced[retrieved_doc_indices]
        
#         query_tfidf_vector = self.tfidf_vectorizer.transform([preprocessed_query_string])
#         reduced_query_vector = self.svd_model.transform(query_tfidf_vector)

#         similarity_scores = cosine_similarity(reduced_query_vector, retrieved_vectors).flatten()

#         scored_documents = [(score, self.docs_list[idx]) for score, idx in zip(similarity_scores, retrieved_doc_indices)]

#         scored_documents.sort(key=lambda item: item[0], reverse=True)

#         return scored_documents[:top_k]

#     def search(self, dataset_name: str, query: str, top_k: int = 10, with_index: bool = False) -> list[tuple[str, float, str]]:
#         """
#         Orchestrates the cluster-based document search process for a given raw query.
#         This method conforms to the Retriever.search signature.
#         """
#         if not self.models_loaded:
#             print("❌ Models are not loaded. Cannot perform search.")
#             return []

#         preprocessed_query_string = self.preprocess_query_text(query)
        
#         if not preprocessed_query_string:
#             print("❌ Query preprocessing failed or resulted in empty tokens.")
#             return []

#         reduced_query_vector = self.vectorize_and_reduce_query(preprocessed_query_string)
#         if reduced_query_vector is None:
#             print("❌ Query vectorization and reduction failed.")
#             return []

#         predicted_cluster_id = self.predict_query_cluster(reduced_query_vector)
#         if predicted_cluster_id is None:
#             print("❌ Query cluster prediction failed.")
#             return []

#         retrieved_doc_indices = self.retrieve_documents_from_clusters(predicted_cluster_id)
#         if not retrieved_doc_indices:
#             return []

#         ranked_scored_documents = self.rank_documents(retrieved_doc_indices, preprocessed_query_string, top_k)
        
#         if not ranked_scored_documents:
#             return []

#         final_results = []
#         for score, doc_obj in ranked_scored_documents:
#             final_results.append((
#                 doc_obj.doc_id,
#                 float(score),
#                 doc_obj.text[:100] + "..." if len(doc_obj.text) > 100 else doc_obj.text
#             ))

#         return final_results

# import os
# import joblib
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from services.processing.text_preprocessor import TextPreprocessor
# from services.online_vectorizers.Retriever import Retriever
# from sklearn.decomposition import TruncatedSVD


# class ClusterSearch(Retriever):
#     """
#     A class to perform cluster-based document search using TF-IDF, SVD, and KMeans.

#     It loads pre-trained models (TF-IDF vectorizer, SVD model, KMeans model)
#     and document information (document list, cluster labels) from a specified path.
#     It provides methods for query preprocessing, vectorization, cluster prediction,
#     document retrieval from predicted clusters, and ranking within those clusters.
#     """
#     def __init__(self, model_path, dataset_name="antique", kmeans_model_k_value: int = 8):
#         """
#         Initializes the ClusterSearch with the path to the saved models and loads them.

#         Args:
#             model_path (str): The path to the directory containing the joblib model files.
#             dataset_name (str): The name of the dataset (e.g., "antique").
#             kmeans_model_k_value (int): The 'K' value used for the KMeans model (e.g., 3, 8).
#         """
#         self.model_path = model_path
#         self.dataset_name = dataset_name 
#         self.kmeans_model_k_value = kmeans_model_k_value # Initialize the attribute first

#         self.tfidf_vectorizer = None
#         self.svd_model = None
#         self.kmeans_model = None
#         self.docs_list = None
#         self.original_docs_by_id = {} # For faster lookup of original doc objects by ID
#         self.cluster_labels = None
#         self.docs_tfidf_matrix_reduced = None # To store the reduced matrix of all documents
#         self.models_loaded = False

#         # Define full paths to the model files using os.path.join for robustness
#         tfidf_vectorizer_path = os.path.join(self.model_path, "tfidf_vectorizer.joblib")
#         svd_model_path = os.path.join(self.model_path, "svd_model.joblib")
#         kmeans_model_path = os.path.join(self.model_path, f"kmeans_model_k{self.kmeans_model_k_value}.joblib") 
#         docs_list_path = os.path.join(self.model_path, "docs_list.joblib")
#         cluster_labels_path = os.path.join(self.model_path, f"document_cluster_labels_k{self.kmeans_model_k_value}.joblib") 
#         tfidf_matrix_full_path = os.path.join(self.model_path, "tfidf_matrix.joblib") # Path to the full TF-IDF matrix

#         print(f"Attempting to load models from: {self.model_path}")

#         try:
#             # Load TF-IDF Vectorizer
#             self.tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
#             print("✅ Loaded tfidf_vectorizer.joblib")

#             # Load SVD Model
#             self.svd_model = joblib.load(svd_model_path)
#             print("✅ Loaded svd_model.joblib")

#             # Load the full TF-IDF matrix
#             full_tfidf_matrix = joblib.load(tfidf_matrix_full_path)
#             print("✅ Loaded tfidf_matrix.joblib (full matrix)")
            
#             # Apply SVD transformation to the full TF-IDF matrix ONCE during initialization
#             self.docs_tfidf_matrix_reduced = self.svd_model.transform(full_tfidf_matrix)
#             print(f"✅ Applied SVD to full TF-IDF matrix. Reduced shape: {self.docs_tfidf_matrix_reduced.shape}")

#             # Load original Document List and create a dict for quick lookup
#             self.docs_list = joblib.load(docs_list_path)
#             self.original_docs_by_id = {doc.doc_id: doc for doc in self.docs_list}
#             print("✅ Loaded docs_list.joblib")

#             # Load KMeans Model (for the specified K value)
#             self.kmeans_model = joblib.load(kmeans_model_path)
#             print(f"✅ Loaded kmeans_model_k{self.kmeans_model_k_value}.joblib")
            
#             # Load Document Cluster Labels (for the specified K value)
#             self.cluster_labels = joblib.load(cluster_labels_path)
#             print(f"✅ Loaded document_cluster_labels_k{self.kmeans_model_k_value}.joblib")

#             self.models_loaded = True 
#             print("✅ All models loaded successfully.")

#         except FileNotFoundError as e:
#             print(f"❌ Error loading model file: {e}. Please ensure all required joblib files are in the specified path.")
#             self.models_loaded = False 
#         except Exception as e:
#             print(f"❌ An unexpected error occurred during model loading: {e}")
#             self.models_loaded = False 

#         if not self.models_loaded:
#             print("⚠️ Models not loaded. Search functionality will be unavailable.")


#     def preprocess_query_text(self, query: str) -> str:
#         """
#         Preprocesses the input query string using the TextPreprocessor class.
#         Returns a single string of processed tokens, suitable for vectorizers.
#         """
#         try:
#             preprocessor_instance = TextPreprocessor.getInstance()
#             processed_tokens_list = preprocessor_instance.preprocess_text(query)
#             return " ".join(processed_tokens_list)
#         except NameError:
#             print("❌ Error: TextPreprocessor class not found. Please ensure it is defined and accessible.")
#             return ""


#     def vectorize_and_reduce_query(self, preprocessed_query_string: str) -> np.ndarray | None:
#         """
#         Vectorizes the preprocessed query string using the loaded TF-IDF vectorizer
#         and applies the loaded SVD model for dimensionality reduction.
#         """
#         if not self.models_loaded or self.tfidf_vectorizer is None or self.svd_model is None:
#             print("⚠️ TF-IDF vectorizer or SVD model not loaded. Cannot vectorize and reduce query.")
#             return None

#         tfidf_query_vector = self.tfidf_vectorizer.transform([preprocessed_query_string])
#         reduced_query_vector = self.svd_model.transform(tfidf_query_vector)

#         return reduced_query_vector

#     def predict_query_cluster(self, reduced_query_vector: np.ndarray) -> int | None:
#         """
#         Predicts the cluster the query belongs to using the loaded KMeans model.
#         Returns the predicted cluster ID (as a standard Python int).
#         """
#         if not self.models_loaded or self.kmeans_model is None:
#             print("⚠️ KMeans model not loaded. Cannot predict query cluster.")
#             return None

#         predicted_cluster = self.kmeans_model.predict(reduced_query_vector)
#         return int(predicted_cluster[0]) # Explicitly cast to Python int to avoid ValueError later


#     def retrieve_documents_from_clusters(self, cluster_ids: int | list) -> np.ndarray:
#         """
#         Retrieves indices of documents that belong to the specified cluster ID(s).
#         Returns a NumPy array of integer indices into the original docs_list/tfidf_matrix.
#         """
#         if self.docs_list is None or self.cluster_labels is None:
#             print("⚠️ Document list or cluster labels not loaded. Cannot retrieve documents.")
#             return np.array([]) # Return empty NumPy array if not loaded

#         # Ensure cluster_ids is a list for consistent handling
#         if isinstance(cluster_ids, int):
#             cluster_ids_list = [cluster_ids]
#         elif isinstance(cluster_ids, (list, tuple)):
#             cluster_ids_list = cluster_ids
#         else:
#             print("⚠️ Invalid input for cluster_ids. Please provide an integer or a list/tuple of integers.")
#             return np.array([]) # Return empty NumPy array for invalid input

#         cluster_labels_np = np.array(self.cluster_labels)
#         document_indices_mask = np.isin(cluster_labels_np, cluster_ids_list)
#         document_indices = np.where(document_indices_mask)[0] # np.where returns a tuple, [0] gets the array

#         return document_indices # Returns a NumPy array of indices

#     def rank_documents(self, retrieved_doc_indices: np.ndarray, preprocessed_query_string: str, top_k: int) -> list[tuple[float, object]]:
#         """
#         Ranks a list of retrieved documents based on their relevance to the preprocessed query.
#         Uses the pre-loaded reduced TF-IDF matrix for efficiency.
#         Returns a list of (score, doc_object) tuples, sorted by score descending.
#         """
#         if self.tfidf_vectorizer is None or self.docs_tfidf_matrix_reduced is None:
#             print("⚠️ TF-IDF vectorizer or reduced document matrix not loaded. Cannot rank documents.")
#             return []

#         # FIX: Check if the NumPy array is empty using its length
#         if len(retrieved_doc_indices) == 0: 
#             return [] # Just return an empty list, the calling search method handles the message
        
#         # Get the reduced vectors for ONLY the retrieved documents from the full reduced matrix
#         retrieved_vectors = self.docs_tfidf_matrix_reduced[retrieved_doc_indices]
        
#         # Transform the preprocessed query string into TF-IDF and then reduce it
#         query_tfidf_vector = self.tfidf_vectorizer.transform([preprocessed_query_string])
#         reduced_query_vector = self.svd_model.transform(query_tfidf_vector)

#         # Calculate cosine similarity between the single reduced query vector and each retrieved document's reduced vector
#         similarity_scores = cosine_similarity(reduced_query_vector, retrieved_vectors).flatten()

#         # Create a list of tuples, pairing each similarity score with its corresponding document object
#         # Use original_docs_by_id for faster lookup by doc_id if retrieved_doc_indices store doc_ids
#         # Assuming retrieved_doc_indices are 0-based indices into self.docs_list
#         scored_documents = [(score, self.docs_list[idx]) for score, idx in zip(similarity_scores, retrieved_doc_indices)]

#         # Sort the list of (score, document) tuples in descending order based on the similarity score
#         scored_documents.sort(key=lambda item: item[0], reverse=True)

#         return scored_documents[:top_k] # Return top_k (score, doc_object) tuples

#     def search(self, dataset_name: str, query: str, top_k: int = 10, with_index: bool = False) -> list[tuple[str, float, str]]:
#         """
#         Orchestrates the cluster-based document search process for a given raw query.
#         This method conforms to the Retriever.search signature.
#         """
#         if not self.models_loaded:
#             print("❌ Models are not loaded. Cannot perform search.")
#             return []

#         # Step 1: Preprocess the raw query string
#         preprocessed_query_string = self.preprocess_query_text(query)
        
#         if not preprocessed_query_string:
#             print("❌ Query preprocessing failed or resulted in empty tokens.")
#             return []

#         # Step 2: Vectorize and reduce the dimensionality of the preprocessed query
#         reduced_query_vector = self.vectorize_and_reduce_query(preprocessed_query_string)
#         if reduced_query_vector is None:
#             print("❌ Query vectorization and reduction failed.")
#             return []

#         # Step 3: Predict the cluster ID for the reduced query vector
#         predicted_cluster_id = self.predict_query_cluster(reduced_query_vector)
#         if predicted_cluster_id is None:
#             print("❌ Query cluster prediction failed.")
#             return []

#         # Step 4: Retrieve documents belonging to the predicted cluster(s) - get indices
#         retrieved_doc_indices = self.retrieve_documents_from_clusters(predicted_cluster_id)
#         # FIX: Check if the NumPy array of indices is empty
#         if len(retrieved_doc_indices) == 0:
#             print(f"❌ No documents found in predicted cluster {predicted_cluster_id}.") # Print message here
#             return []

#         # Step 5: Rank the retrieved documents based on relevance to the query
#         # rank_documents returns (score, doc_object) tuples
#         ranked_scored_documents = self.rank_documents(retrieved_doc_indices, preprocessed_query_string, top_k)
        
#         if not ranked_scored_documents: # This check is for the list of tuples from rank_documents
#             print(f"❌ Document ranking failed for cluster {predicted_cluster_id}.")
#             return []

#         # Step 6: Return the final list of ranked documents in the required format (doc_id, score, text_snippet)
#         final_results = []
#         for score, doc_obj in ranked_scored_documents:
#             final_results.append((
#                 doc_obj.doc_id,
#                 float(score),
#                 doc_obj.text[:100] + "..." if len(doc_obj.text) > 100 else doc_obj.text # Snippet length
#             ))

#         return final_results

import os
import joblib
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from services.processing.text_preprocessor import TextPreprocessor
from services.online_vectorizers.Retriever import Retriever
from sklearn.decomposition import TruncatedSVD


class ClusterSearch(Retriever):
    """
    Optimized ClusterSearch with added diagnostics to check cluster sizes during search.
    """
    def __init__(self, model_path, dataset_name="antique", kmeans_model_k_value: int = 12):
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.kmeans_model_k_value = kmeans_model_k_value
        self.cluster_to_docs_index = defaultdict(list)
        
        self.tfidf_vectorizer = None
        self.svd_model = None
        self.kmeans_model = None
        self.docs_list = None
        self.cluster_labels = None
        self.docs_tfidf_matrix_reduced = None
        self.models_loaded = False

        # Define paths
        tfidf_vectorizer_path = os.path.join(self.model_path, "tfidf_vectorizer.joblib")
        svd_model_path = os.path.join(self.model_path, "svd_model.joblib")
        kmeans_model_path = os.path.join(self.model_path, f"kmeans_model_svd_k{self.kmeans_model_k_value}.joblib")
        docs_list_path = os.path.join(self.model_path, "docs_list.joblib")
        cluster_labels_path = os.path.join(self.model_path, f"document_cluster_labels_kmeans_svd_k{self.kmeans_model_k_value}.joblib")
        tfidf_matrix_reduced_path = os.path.join(self.model_path, "tfidf_matrix_reduced_svd.joblib")

        print(f"Attempting to load models from: {self.model_path}")

        try:
            self.tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
            self.svd_model = joblib.load(svd_model_path)
            self.docs_list = joblib.load(docs_list_path)
            self.kmeans_model = joblib.load(kmeans_model_path)
            self.cluster_labels = joblib.load(cluster_labels_path)
            self.docs_tfidf_matrix_reduced = joblib.load(tfidf_matrix_reduced_path)
            print("✅ All model and data files loaded successfully.")

            print("⏳ Building cluster-to-document index for fast lookups...")
            for doc_index, cluster_id in enumerate(self.cluster_labels):
                self.cluster_to_docs_index[cluster_id].append(doc_index)
            
            for cluster_id in self.cluster_to_docs_index:
                self.cluster_to_docs_index[cluster_id] = np.array(self.cluster_to_docs_index[cluster_id], dtype=np.int32)
            print("✅ Cluster index built successfully.")

            self.models_loaded = True
            print("✅ All models loaded and optimized for search.")

        except (FileNotFoundError, EOFError) as e:
            print(f"❌ Error loading a required file: {e}. Please check the model_path and file integrity.")
            self.models_loaded = False
        except Exception as e:
            print(f"❌ An unexpected error occurred during model loading: {e}")
            self.models_loaded = False
    
    # --- MODIFICATION: Added a diagnostic print statement ---
    def search(self, dataset_name: str, query: str, top_k: int = 10, with_index: bool = False) -> list[tuple[str, float, str]]:
        if not self.models_loaded:
            return []

        preprocessed_query = self.preprocess_query_text(query)
        if not preprocessed_query: return []

        reduced_query_vector = self.vectorize_and_reduce_query(preprocessed_query)
        if reduced_query_vector is None: return []

        predicted_cluster_id = self.predict_query_cluster(reduced_query_vector)
        if predicted_cluster_id is None: return []

        retrieved_doc_indices = self.retrieve_documents_from_clusters(predicted_cluster_id)
        
        # --- DIAGNOSTIC LINE ---
        # This will tell us the size of the cluster we are about to rank.
        print(f"    [Diagnostic] Query landed in cluster {predicted_cluster_id}, which has {len(retrieved_doc_indices)} documents to rank.")
        
        if len(retrieved_doc_indices) == 0: return []

        ranked_docs = self.rank_documents(retrieved_doc_indices, reduced_query_vector, top_k)
        
        final_results = [
            (doc.doc_id, float(score), doc.text[:100] + "...")
            for score, doc in ranked_docs
        ]
        return final_results

    def preprocess_query_text(self, query: str) -> str:
        preprocessor_instance = TextPreprocessor.getInstance()
        return " ".join(preprocessor_instance.preprocess_text(query))

    def vectorize_and_reduce_query(self, preprocessed_query_string: str) -> np.ndarray | None:
        if not self.models_loaded: return None
        tfidf_query_vector = self.tfidf_vectorizer.transform([preprocessed_query_string])
        return self.svd_model.transform(tfidf_query_vector)

    def predict_query_cluster(self, reduced_query_vector: np.ndarray) -> int | None:
        if not self.models_loaded: return None
        return int(self.kmeans_model.predict(reduced_query_vector)[0])

    def retrieve_documents_from_clusters(self, cluster_id: int) -> np.ndarray:
        return self.cluster_to_docs_index.get(cluster_id, np.array([], dtype=np.int32))

    def rank_documents(self, retrieved_doc_indices: np.ndarray, reduced_query_vector: np.ndarray, top_k: int) -> list[tuple[float, object]]:
        if len(retrieved_doc_indices) == 0: return []
        retrieved_vectors = self.docs_tfidf_matrix_reduced[retrieved_doc_indices]
        similarity_scores = cosine_similarity(reduced_query_vector, retrieved_vectors).flatten()
        scored_documents = [(score, self.docs_list[idx]) for score, idx in zip(similarity_scores, retrieved_doc_indices)]
        scored_documents.sort(key=lambda item: item[0], reverse=True)
        return scored_documents[:top_k]