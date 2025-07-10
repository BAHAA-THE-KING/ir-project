# import joblib
# import os
# import re
# import nltk
# from sklearn.metrics.pairwise import cosine_similarity
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet
# from nltk import pos_tag
# from typing import List, Tuple

# # --- Start of Preprocessing Functions (Copied from src/services/processing/preprocessing.py) ---
# __lemmatizer__ = None # Will be initialized in _ensure_nltk_data
# __stop_words__ = None # Will be initialized in _ensure_nltk_data
# # src/test_cluster_standalone.py (modified _ensure_nltk_data function)

# # ... (keep all your existing imports at the top of the file) ...
# import sys # Ensure sys is imported for sys.exit

# # ... (keep __lemmatizer__ and __stop_words__ global declarations) ...

# def _ensure_nltk_data():
#     """Ensures necessary NLTK data is downloaded and initialized."""
#     required_nltk_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']

#     for data_item in required_nltk_data:
#         try:
#             # Try to find the data. nltk.data.find will raise LookupError if not found.
#             nltk.data.find(data_item) 
#             print(f"NLTK data '{data_item}' found.")
#         except LookupError:
#             print(f"NLTK data '{data_item}' not found. Attempting to download...")
#             try:
#                 # Attempt to download the missing data
#                 nltk.download(data_item)
#                 print(f"Successfully downloaded '{data_item}'.")
#             except Exception as download_e:
#                 # Catch any other error during download and provide guidance
#                 print(f"Failed to download '{data_item}'. Error: {download_e}")
#                 print(f"Please try running 'python -c \"import nltk; nltk.download(\'{data_item}\")\"' manually in your terminal.")
#                 sys.exit(1) # Exit the script if essential data cannot be downloaded
#         except Exception as e:
#             # Catch any unexpected errors during the check
#             print(f"An unexpected error occurred while checking NLTK data '{data_item}': {e}")
#             sys.exit(1)

#     global __lemmatizer__, __stop_words__
#     if __lemmatizer__ is None:
#         __lemmatizer__ = WordNetLemmatizer()
#     if __stop_words__ is None:
#         __stop_words__ = set(stopwords.words('english'))
#     print("All required NLTK data checked and initialized.")

# # ... (rest of your preprocessing functions and ClusterSearchApp class and main execution block) ...

# def __clean_text__(text):
#     """
#     Clean text by removing special characters and converting to lowercase
#     """
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z\s]', '', text)
#     return text

# def __remove_stopwords__(text):
#     """
#     Remove common stopwords from text
#     """
#     words = word_tokenize(text)
#     filtered_words = [word for word in words if word not in __stop_words__]
#     return ' '.join(filtered_words)

# def __get_wordnet_pos__(tag_parameter):
#     tag = tag_parameter[0].upper()
#     tag_dict = {
#         "J": wordnet.ADJ,
#         "N": wordnet.NOUN,
#         "V": wordnet.VERB,
#         "R": wordnet.ADV
#         }
#     return tag_dict.get(tag, wordnet.NOUN)

# def __lemmatize_text__(text):
#     """
#     Lemmatize words to their root form
#     """
#     words = word_tokenize(text)
#     pos_tags = pos_tag(words)
#     lemmatized_words = [__lemmatizer__.lemmatize(word, pos=__get_wordnet_pos__(tag)) for word, tag in pos_tags]
#     return ' '.join(lemmatized_words)

# def preprocess_text(text: str) -> List[str]:
#     """
#     Apply full preprocessing pipeline and return list of tokens.
#     """
#     if not isinstance(text, str):
#         return []
    
#     text = __clean_text__(text)
#     text = __remove_stopwords__(text)
#     text = __lemmatize_text__(text)

#     # Return as a list of tokens, not just split string
#     return text.strip().split()

# # --- End of Preprocessing Functions ---


# class ClusterSearchApp:
#     def __init__(self, project_path):
#         self.project_path = project_path
#         self.tfidf_vectorizer = None
#         self.tfidf_matrix = None
#         self.docs_list = None
#         self.document_cluster_labels = None
#         self.kmeans_model = None

#     def load_models(self):
#         """Loads the necessary models and data from joblib files."""
#         try:
#         # Find this line:
#             data_dir = os.path.join(self.project_path, 'src', 'data', 'quora')
#             # If your data is in src/data/antique
#             # data_dir = os.path.join(self.project_path, 'src', 'data', 'antique') 
            
#             self.tfidf_vectorizer = joblib.load(os.path.join(data_dir, 'tfidf_vectorizer.joblib'))
#             self.tfidf_matrix = joblib.load(os.path.join(data_dir, 'tfidf_matrix.joblib'))
#             self.docs_list = joblib.load(os.path.join(data_dir, 'docs_list.joblib'))
#             self.document_cluster_labels = joblib.load(os.path.join(data_dir, 'document_cluster_labels.joblib'))
#             self.kmeans_model = joblib.load(os.path.join(data_dir, 'kmeans_model.joblib'))
#             print("Successfully loaded models and data.")
#             return True
#         except FileNotFoundError as e:
#             print(f"Error loading files: {e}")
#             print("Please ensure your data (joblib files) are in 'data/antique/' relative to your project root.")
#             self.tfidf_vectorizer = self.tfidf_matrix = self.docs_list = self.document_cluster_labels = self.kmeans_model = None
#             return False
#         except Exception as e:
#             print(f"An unexpected error occurred during model loading: {e}")
#             self.tfidf_vectorizer = self.tfidf_matrix = self.docs_list = self.document_cluster_labels = self.kmeans_model = None
#             return False


#     def cluster_search(self, raw_query: str, top_n: int = 5) -> List[Tuple[str, float, str]]:
#         """
#         Performs cluster-based search using the loaded models.

#         Args:
#             raw_query (str): The raw search query string.
#             top_n (int): The number of top results to return from the relevant cluster.

#         Returns:
#             list: A list of top_n documents from the most relevant cluster, 
#                   formatted as (doc_id, similarity_score, text_snippet).
#         """
  

#         if not isinstance(raw_query, str) or not raw_query.strip():
#             print("Query must be a non-empty string.")
#             return []

#         # Preprocess the raw query using the included function
#         processed_query_tokens = preprocess_text(raw_query)
#         processed_query_str = " ".join(processed_query_tokens) # Join tokens back for TF-IDF vectorizer

#         if not processed_query_str:
#             print("Processed query is empty. No search will be performed.")
#             return []

#         try:
#             query_vector = self.tfidf_vectorizer.transform([processed_query_str])
#         except Exception as e:
#             print(f"Error transforming query with TF-IDF vectorizer: {e}")
#             return []

#         try:
#             query_cluster = self.kmeans_model.predict(query_vector)[0]
#         except Exception as e:
#             print(f"Error predicting cluster for query: {e}")
#             return []

#         print(f"Query '{raw_query}' (processed: '{processed_query_str}') belongs to cluster: {query_cluster}")

#         relevant_document_indices = [i for i, label in enumerate(self.document_cluster_labels) if label == query_cluster]

#         if not relevant_document_indices:
#             print(f"No documents found in cluster {query_cluster}.")
#             return []

#         try:
#             relevant_tfidf_matrix = self.tfidf_matrix[relevant_document_indices]
#         except Exception as e:
#             print(f"Error accessing relevant TF-IDF vectors: {e}")
#             return []

#         try:
#             similarities = cosine_similarity(query_vector, relevant_tfidf_matrix).flatten()
#         except Exception as e:
#             print(f"Error calculating cosine similarity: {e}")
#             return []

#         # Sort by similarity in descending order
#         sorted_indices_in_cluster = similarities.argsort()[::-1]
        
#         final_results = []
#         for i in range(min(top_n, len(sorted_indices_in_cluster))):
#             relative_idx = sorted_indices_in_cluster[i]
#             original_doc_idx = relevant_document_indices[relative_idx]
            
#             doc = self.docs_list[original_doc_idx]
#             score = similarities[relative_idx]
            
#             # Ensure doc has .doc_id and .text attributes, with fallbacks
#             doc_id = getattr(doc, 'doc_id', str(original_doc_idx)) 
#             text_content = getattr(doc, 'text', '')
#             text_snippet = text_content[:200] + "..." if len(text_content) > 200 else text_content
            
#             final_results.append((doc_id, float(score), text_snippet))
        
#         return final_results


# if __name__ == "__main__":
#     _ensure_nltk_data() # Ensure NLTK data is available first

#     current_script_path = os.path.abspath(__file__)
#     project_root = os.path.dirname(current_script_path) # This script is in project root

#     print(f"Running standalone cluster test from: {project_root}")

#     app = ClusterSearchApp(project_root)
#     if not app.load_models():
#         print("\nFailed to load models. Please ensure 'data/antique/' with .joblib files exists at your project root, and run 'src/run_clustering.py' first.")
#         input("Press Enter to exit...")
#         sys.exit(1)

#     print("\nEnter queries to test (type 'quit' to exit):")
#     while True:
#         query = input("\nQuery: ")
#         if query.lower() == 'quit':
#             break
        
#         results = app.cluster_search(query, top_n=5)
        
#         if results:
#             print("\n--- Top Results ---")
#             for i, (doc_id, score, snippet) in enumerate(results):
#                 print(f"{i+1}. Doc ID: {doc_id}, Score: {score:.4f}")
#                 print(f"   Snippet: {snippet}")
#                 print("-" * 20)
#         else:
#             print("No results found for your query.")


import joblib
import os
import re
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from typing import List, Tuple
import sys # Ensure sys is imported for sys.exit

# --- Start of Preprocessing Functions (Copied from src/services/processing/preprocessing.py) ---
# These global variables will be initialized by _ensure_nltk_data()
__lemmatizer__ = None
__stop_words__ = None

def _ensure_nltk_data():
    """Ensures necessary NLTK data is downloaded and initialized."""
    required_nltk_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']

    for data_item in required_nltk_data:
        try:
            nltk.data.find(data_item)
            # print(f"NLTK data '{data_item}' found.") # Uncomment for verbose output
        except LookupError:
            print(f"NLTK data '{data_item}' not found. Attempting to download...")
            try:
                nltk.download(data_item)
                print(f"Successfully downloaded '{data_item}'.")
            except Exception as download_e:
                print(f"Failed to download '{data_item}'. Error: {download_e}")
                print(f"Please try running 'python -c \"import nltk; nltk.download(\'{data_item}\")\"' manually in your terminal.")
                sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while checking NLTK data '{data_item}': {e}")
            sys.exit(1)

    global __lemmatizer__, __stop_words__
    if __lemmatizer__ is None:
        __lemmatizer__ = WordNetLemmatizer()
    if __stop_words__ is None:
        __stop_words__ = set(stopwords.words('english'))
    print("All required NLTK data checked and initialized.")

def __clean_text__(text):
    """Clean text by removing special characters and converting to lowercase"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def __remove_stopwords__(text):
    """Remove common stopwords from text"""
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in __stop_words__]
    return ' '.join(filtered_words)

def __get_wordnet_pos__(tag_parameter):
    tag = tag_parameter[0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
        }
    return tag_dict.get(tag, wordnet.NOUN)

def __lemmatize_text__(text):
    """Lemmatize words to their root form"""
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    lemmatized_words = [__lemmatizer__.lemmatize(word, pos=__get_wordnet_pos__(tag)) for word, tag in pos_tags]
    return ' '.join(lemmatized_words)

def preprocess_text(text: str) -> List[str]:
    """Apply full preprocessing pipeline and return list of tokens."""
    if not isinstance(text, str):
        return []
    
    text = __clean_text__(text)
    text = __remove_stopwords__(text)
    text = __lemmatize_text__(text)

    return text.strip().split()
# --- End of Preprocessing Functions ---


class ClusterSearchApp:
    def __init__(self, project_path):
        self.project_path = project_path
        self.tfidf_vectorizer = None
        self.svd_model = None # Add SVD model
        self.docs_list = None
        self.document_cluster_labels = None
        self.kmeans_model = None
        self.high_dim_tfidf_matrix = None # Store the high-dimensional TF-IDF matrix

    def load_models(self):
        """Loads the necessary models and data from joblib files."""
        try:
            # Assuming data is in 'data/quora' as per image
            data_dir = os.path.join(self.project_path, 'data', 'quora')
            
            self.tfidf_vectorizer = joblib.load(os.path.join(data_dir, 'tfidf_vectorizer.joblib'))
            self.high_dim_tfidf_matrix = joblib.load(os.path.join(data_dir, 'tfidf_matrix.joblib')) # This is the original high-dim matrix
            self.docs_list = joblib.load(os.path.join(data_dir, 'docs_list.joblib'))
            self.document_cluster_labels = joblib.load(os.path.join(data_dir, 'document_cluster_labels.joblib'))
            self.kmeans_model = joblib.load(os.path.join(data_dir, 'kmeans_model.joblib'))
            self.svd_model = joblib.load(os.path.join(data_dir, 'svd_model.joblib')) # Load the SVD model
            
            print("Successfully loaded models and data.")
            return True
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            print("Please ensure your data (joblib files) are in 'data/quora/' relative to your project root, and run 'run_clustering.py' first.")
            self.tfidf_vectorizer = self.high_dim_tfidf_matrix = self.docs_list = self.document_cluster_labels = self.kmeans_model = self.svd_model = None
            return False
        except Exception as e:
            print(f"An unexpected error occurred during model loading: {e}")
            self.tfidf_vectorizer = self.high_dim_tfidf_matrix = self.docs_list = self.document_cluster_labels = self.kmeans_model = self.svd_model = None
            return False

    def cluster_search(self, raw_query: str, top_n: int = 5) -> List[Tuple[str, float, str]]:
        """
        Performs cluster-based search using the loaded models.
        Args:
            raw_query (str): The raw search query string.
            top_n (int): The number of top results to return from the relevant cluster.
        Returns:
            list: A list of top_n documents from the most relevant cluster, 
                  formatted as (doc_id, similarity_score, text_snippet).
        """
        if not isinstance(raw_query, str) or not raw_query.strip():
            print("Query must be a non-empty string.")
            return []

        # 1. Preprocess the query
        processed_query_tokens = preprocess_text(raw_query)
        processed_query_str = " ".join(processed_query_tokens)

        if not processed_query_str:
            print("Processed query is empty. No search will be performed.")
            return []

        try:
            # 2. Transform query into TF-IDF vector (high-dimensional)
            query_vector_high_dim = self.tfidf_vectorizer.transform([processed_query_str])
            
            # 3. Reduce query vector dimensionality using the trained SVD model
            # This is crucial because KMeans was trained on reduced dimensions
            query_vector_reduced = self.svd_model.transform(query_vector_high_dim)

            # 4. Predict the cluster for the query using the K-Means model
            # The cluster is picked by finding the centroid closest to the query's reduced vector
            query_cluster = self.kmeans_model.predict(query_vector_reduced)[0]

        except Exception as e:
            print(f"Error processing query or predicting cluster: {e}")
            return []

        print(f"Query '{raw_query}' (processed: '{processed_query_str}') belongs to cluster: {query_cluster}")

        # 5. Filter documents to only include those from the predicted cluster
        # This significantly narrows down the search space for efficiency
        relevant_document_indices_in_full_list = [
            i for i, label in enumerate(self.document_cluster_labels) if label == query_cluster
        ]

        if not relevant_document_indices_in_full_list:
            print(f"No documents found in cluster {query_cluster}.")
            return []

        try:
            # 6. Retrieve the original high-dimensional TF-IDF vectors for these relevant documents
            original_relevant_tfidf_vectors = self.high_dim_tfidf_matrix[relevant_document_indices_in_full_list]
            
            # 7. Reduce the dimensionality of the relevant document vectors using the same SVD model
            relevant_tfidf_matrix_reduced = self.svd_model.transform(original_relevant_tfidf_vectors)

            # 8. Calculate cosine similarity between the reduced query vector and the reduced relevant document vectors
            # Cosine similarity measures the angle between vectors, indicating topical similarity
            similarities = cosine_similarity(query_vector_reduced, relevant_tfidf_matrix_reduced).flatten()

        except Exception as e:
            print(f"Error calculating similarities: {e}")
            return []

        # 9. Sort results by similarity in descending order
        sorted_indices_in_cluster_subset = similarities.argsort()[::-1]
        
        final_results = []
        for i in range(min(top_n, len(sorted_indices_in_cluster_subset))):
            relative_idx = sorted_indices_in_cluster_subset[i]
            original_doc_idx = relevant_document_indices_in_full_list[relative_idx]
            
            doc = self.docs_list[original_doc_idx]
            score = similarities[relative_idx]
            
            # Ensure doc has .doc_id and .text attributes, with fallbacks
            doc_id = getattr(doc, 'doc_id', str(original_doc_idx))
            text_content = getattr(doc, 'text', '')
            text_snippet = text_content[:200] + "..." if len(text_content) > 200 else text_content
            
            final_results.append((doc_id, float(score), text_snippet))
        
        return final_results


if __name__ == "__main__":
    _ensure_nltk_data() # Ensure NLTK data is available first

    current_script_path = os.path.abspath(__file__)
    # Adjust project_root based on your actual directory structure if this file is not directly in project root
    # For example, if this file is in 'src', then project_root is os.path.dirname(current_script_path)
    # If this file is in 'src/services/some_module', then project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_path)))
    project_root = os.path.dirname(current_script_path) 
    print(f"Running standalone cluster search test from: {project_root}")

    app = ClusterSearchApp(project_root)
    if not app.load_models():
        print("\nFailed to load models. Please ensure 'data/quora/' with .joblib files exists relative to your project root, and run 'run_clustering.py' first.")
        input("Press Enter to exit...")
        sys.exit(1)

    print("\nEnter queries to test (type 'quit' to exit):")
    while True:
        query = input("\nQuery: ")
        if query.lower() == 'quit':
            break
        
        results = app.cluster_search(query, top_n=5)
        
        if results:
            print("\n--- Top Results ---")
            for i, (doc_id, score, snippet) in enumerate(results):
                print(f"{i+1}. Doc ID: {doc_id}, Score: {score:.4f}")
                print(f"   Snippet: {snippet}")
                print("-" * 20)
        else:
            print("No results found for your query.")