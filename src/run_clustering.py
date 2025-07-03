import os
import sys
import joblib
from pathlib import Path

# Import essential libraries (THESE LINES WERE MISSING IN YOUR LAST PROVIDED SCRIPT)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD # Import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np

# Set up Python path to enable correct imports for the project structure
# current_script_dir will be 'c:/Users/FSOS/Documents/Projects/ir-project/src'
current_script_dir = Path(__file__).parent
# We need to add 'c:/Users/FSOS/Documents/Projects/ir-project' to the path
project_root_dir = current_script_dir.parent
if str(project_root_dir) not in sys.path:
    sys.path.insert(0, str(project_root_dir))

# Now, imports like 'from src.loader' will work
from src.loader import load_dataset_with_queries
from src.services.offline_vectorizers.tfidf import tfidf_train
from src.services.processing.text_preprocessor import TextPreprocessor

# --- 1. NLTK Data Download Check (Crucial Step) ---
print("--- Checking NLTK Data ---")
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    print("✅ Required NLTK data found.")
except LookupError:
    print("⚠️ NLTK data not found. Downloading...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    print("✅ NLTK data download complete.")
except Exception as e:
    print(f"❌ Error checking/downloading NLTK data: {e}")
print("-" * 50)


# --- 2. Custom Tokenizer Setup for TF-IDF (for training if needed) ---
class CustomVectorizerTokenizer:
    def __call__(self, text):
        return TextPreprocessor.getInstance().preprocess_text(text)


# --- 3. Load Trained TF-IDF Matrix (or train if not found) ---
dataset_name = "antique"  # <--- Change this to your dataset name
data_dir = Path(f"data/{dataset_name}")
data_dir.mkdir(parents=True, exist_ok=True) # Ensure data directory exists

tfidf_matrix_path = data_dir / "tfidf_matrix.joblib"
docs_list_path = data_dir / "docs_list.joblib"
tfidf_vectorizer_path = data_dir / "tfidf_vectorizer.joblib"

docs = None
tfidf_matrix = None
tfidf_vectorizer = None

print(f"--- Attempting to load TF-IDF from: {data_dir} ---")
if tfidf_matrix_path.exists() and docs_list_path.exists() and tfidf_vectorizer_path.exists():
    try:
        tfidf_matrix = joblib.load(tfidf_matrix_path)
        docs = joblib.load(docs_list_path)
        tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
        print("✅ TF-IDF matrix, document list, and vectorizer loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading TF-IDF files: {e}. Re-training will be attempted.")
        tfidf_matrix = None # Reset to trigger re-training
else:
    print("⚠️ Trained TF-IDF files not found. Loading dataset and training TF-IDF...")

# If TF-IDF was not loaded successfully, proceed to load dataset and train
if tfidf_matrix is None:
    docs, _, _ = load_dataset_with_queries(dataset_name)
    print(f"--- Training TF-IDF Model for '{dataset_name}' ---")
    tfidf_vectorizer, tfidf_matrix, docs = tfidf_train(docs, dataset_name) 
    print("✅ TF-IDF Model training complete.")

print("-" * 50)

if tfidf_matrix is None:
    print("❌ Cannot proceed: No TF-IDF matrix available after loading or training.")
    sys.exit(1) # Exit if matrix is not available


# # --- 4. Determine Optimal K using the Elbow Method (Commented Out as per your request) ---
# print("\n--- Determining Optimal K using Elbow Method ---")
# max_k_range = min(16, tfidf_matrix.shape[0] - 1) 
# if max_k_range < 2:
#     print("⚠️ Document count is too low for effective Elbow Method. At least 2 documents needed.")
#     optimal_k_from_elbow = 1 
# elif max_k_range == 2:
#     optimal_k_from_elbow = 2
# else:
#     k_values = range(2, max_k_range + 1)
#     inertia_values = []
    
#     for k in k_values:
#         print(f"  Testing K = {k}...")
#         kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) 
#         kmeans.fit(tfidf_matrix) 
#         inertia_values.append(kmeans.inertia_)

#     plt.figure(figsize=(10, 6))
#     plt.plot(k_values, inertia_values, marker='o')
#     plt.title('Elbow Method for Optimal K')
#     plt.xlabel('Number of Clusters (K)')
#     plt.ylabel('Inertia (Sum of squared distances)')
#     plt.xticks(k_values)
#     plt.grid(True)
#     plt.show()

#     optimal_k_from_elbow = 5 # Example value, you would determine this from the plot


# --- Explicitly set optimal_k here as per your choice ---
# Since you don't want to run the Elbow Method, you set K directly.
optimal_k = 6 # <--- SET YOUR CHOSEN K (e.g., 5 or 6) HERE
print(f"\n✅ Optimal K is manually set to: {optimal_k}") # Confirm manual setting
print("-" * 50)


# --- 5. Apply K-Means Clustering with Selected K ---
if optimal_k < 2:
    print("❌ Cannot apply K-Means: Optimal K is less than 2. Please set K to 2 or higher.")
else:
    print(f"\n--- Applying K-Means Clustering for K={optimal_k} ---")
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_model.fit(tfidf_matrix)

    cluster_labels = kmeans_model.labels_

    joblib.dump(kmeans_model, data_dir / "kmeans_model.joblib")
    joblib.dump(cluster_labels, data_dir / "document_cluster_labels.joblib")
    print(f"✅ Clustering complete for {optimal_k} clusters.")
    print(f"   K-Means model and cluster labels saved to: {data_dir}")
    print("-" * 50)


    # --- 6. Visualize Clusters (Cluster Distribution Chart) ---
    print("\n--- Creating Cluster Distribution Plot ---")
    
    num_docs_to_plot = min(tfidf_matrix.shape[0], 403666) 
    
    if tfidf_matrix.shape[0] > num_docs_to_plot:
        print(f"⚠️ Document count is high ({tfidf_matrix.shape[0]}). Plotting a sample of {num_docs_to_plot} documents to reduce memory usage.")
        np.random.seed(42) 
        sample_indices = np.random.choice(tfidf_matrix.shape[0], num_docs_to_plot, replace=False)
        
        # Use TruncatedSVD for sparse matrices
        svd = TruncatedSVD(n_components=2, random_state=42)
        reduced_features = svd.fit_transform(tfidf_matrix[sample_indices]) 
        sample_cluster_labels = cluster_labels[sample_indices]
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=sample_cluster_labels, cmap='viridis', s=50, alpha=0.8)
        plt.title(f'Document Clusters for {dataset_name} (TruncatedSVD Reduced, Sampled {num_docs_to_plot} Docs)')
    else:
        # If document count is small, plot all using TruncatedSVD
        svd = TruncatedSVD(n_components=2, random_state=42)
        reduced_features = svd.fit_transform(tfidf_matrix) 
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
        plt.title(f'Document Clusters for {dataset_name} (TruncatedSVD Reduced)')

    plt.xlabel('Component 1 (from TruncatedSVD)') 
    plt.ylabel('Component 2 (from TruncatedSVD)')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True)
    plt.show()

    # Display document count per cluster
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    cluster_distribution = dict(zip(unique_labels, counts))
    print("\nDocument Distribution per Cluster:")
    for cluster_id, count in cluster_distribution.items():
        print(f"  Cluster {cluster_id}: {count} documents")
    print("-" * 50)

print("\n--- Clustering process and visualization complete ---")
print("You can now integrate clustering results (kmeans_model and cluster_labels) into your search function to accelerate search.")