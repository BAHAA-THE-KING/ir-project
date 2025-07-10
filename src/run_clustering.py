# import os
# import sys
# import joblib
# from pathlib import Path

# from sklearn.cluster import KMeans
# from sklearn.decomposition import TruncatedSVD # Import TruncatedSVD for dimensionality reduction
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer # Explicitly import TfidfVectorizer

# # Set up Python path to enable correct imports for the project structure
# current_script_dir = Path(__file__).parent
# project_root_dir = current_script_dir.parent
# if str(project_root_dir) not in sys.path:
#     sys.path.insert(0, str(project_root_dir))

# # Assuming load_dataset_with_queries and preprocess_text are correctly imported
# # You might need to ensure src.loader and src.services.processing.preprocessing are accessible
# from src.loader import load_dataset_with_queries
# from src.services.processing.preprocessing import preprocess_text

# # --- 1. Custom Tokenizer Setup for TF-IDF ---
# class CustomVectorizerTokenizer:
#     def __call__(self, text):
#         # Ensure preprocess_text returns a list of tokens, not a single string
#         return preprocess_text(text)

# # --- 2. Load Trained TF-IDF Matrix (or train if not found) ---
# dataset_name = "quora"
# data_dir = Path(f"data/{dataset_name}")
# data_dir.mkdir(parents=True, exist_ok=True)

# tfidf_matrix_path = data_dir / "tfidf_matrix.joblib"
# docs_list_path = data_dir / "docs_list.joblib"
# tfidf_vectorizer_path = data_dir / "tfidf_vectorizer.joblib"

# docs = None
# tfidf_matrix = None
# tfidf_vectorizer = None

# print(f"--- Attempting to load TF-IDF from: {data_dir} ---")
# if tfidf_matrix_path.exists() and docs_list_path.exists() and tfidf_vectorizer_path.exists():
#     try:
#         tfidf_matrix = joblib.load(tfidf_matrix_path)
#         docs = joblib.load(docs_list_path)
#         tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
#         print("✅ TF-IDF matrix, document list, and vectorizer loaded successfully.")
#     except Exception as e:
#         print(f"❌ Error loading TF-IDF files: {e}. Re-training will be attempted.")
#         tfidf_matrix = None # Reset to trigger re-training
# else:
#     print("⚠️ Trained TF-IDF files not found. Loading dataset and training TF-IDF...")

# # If TF-IDF was not loaded successfully, proceed to load dataset and train
# if tfidf_matrix is None:
#     # This part will execute if the TF-IDF files are not found or loading failed
#     print(f"--- Loading dataset for '{dataset_name}' ---")
#     docs, _, _ = load_dataset_with_queries(dataset_name)

#     if not docs:
#         print(f"❌ No documents loaded for '{dataset_name}'. Cannot proceed with TF-IDF training.")
#         sys.exit(1)

#     print(f"--- Training TF-IDF Model for '{dataset_name}' ---")

#     # Initialize TfidfVectorizer with your desired parameters
#     # Keep experimenting with min_df and max_df to find the best balance
#     tfidf_vectorizer = TfidfVectorizer(
#         tokenizer=CustomVectorizerTokenizer(),
#         min_df=5,       # Ignore terms that appear in less than 5 documents
#         max_df=0.8,     # Ignore terms that appear in more than 80% of documents
#         max_features=50000, # Limit to top 50,000 features
#         sublinear_tf=True # Apply sublinear TF scaling
#     )
    
#     # Prepare documents for TF-IDF (join preprocessed tokens back into strings)
#     # Ensure 'docs' objects have a 'text' attribute with the document content
#     documents_for_tfidf = [" ".join(preprocess_text(doc.text)) for doc in docs if hasattr(doc, 'text')]
    
#     if not documents_for_tfidf:
#         print("❌ No valid text content found in documents for TF-IDF training.")
#         sys.exit(1)

#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents_for_tfidf)
    
#     # Save the newly trained TF-IDF models
#     joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)
#     joblib.dump(tfidf_matrix, tfidf_matrix_path)
#     joblib.dump(docs, docs_list_path)
#     print("✅ TF-IDF Model training complete and saved.")

# print("-" * 50)

# if tfidf_matrix is None or tfidf_matrix.shape[0] == 0:
#     print("❌ Cannot proceed: No TF-IDF matrix available or matrix is empty after loading or training.")
#     sys.exit(1)


# # --- 3. Apply Dimensionality Reduction (TruncatedSVD) ---
# # This is applied to the full TF-IDF matrix before clustering
# n_components_for_clustering = 300 # Experiment with this value (e.g., 100, 200, 500)

# print(f"\n--- Applying TruncatedSVD with {n_components_for_clustering} components for clustering ---")
# # Ensure that tfidf_matrix has at least n_components_for_clustering features
# if tfidf_matrix.shape[1] >= n_components_for_clustering:
#     svd_model = TruncatedSVD(n_components=n_components_for_clustering, random_state=42)
#     tfidf_matrix_reduced = svd_model.fit_transform(tfidf_matrix)
#     joblib.dump(svd_model, data_dir / "svd_model.joblib") # Save SVD model for consistency
#     print(f"✅ TF-IDF matrix reduced from {tfidf_matrix.shape} to {tfidf_matrix_reduced.shape}")
# else:
#     print(f"⚠️ TF-IDF matrix has only {tfidf_matrix.shape[1]} features, which is less than n_components_for_clustering ({n_components_for_clustering}). Skipping SVD reduction for clustering and using original matrix.")
#     tfidf_matrix_reduced = tfidf_matrix # Use original matrix if too few features
# print("-" * 50)


# # --- 4. Determine Optimal K using the Elbow Method (Uncommented for testing) ---
# print("\n--- Determining Optimal K using Elbow Method ---")
# # Ensure there are enough samples for clustering
# if tfidf_matrix_reduced.shape[0] < 2:
#     print("⚠️ Document count is too low for effective Elbow Method. At least 2 documents needed.")
#     optimal_k_from_elbow = 1
# elif tfidf_matrix_reduced.shape[0] == 2:
#     optimal_k_from_elbow = 2
#     print("⚠️ Only 2 documents available. Setting K to 2.")
# else:
#     # Use min(16, number of samples - 1) to prevent errors with small datasets
#     max_k_range = min(16, tfidf_matrix_reduced.shape[0] - 1)
#     k_values = range(2, max_k_range + 1)
#     inertia_values = []
    
#     for k in k_values:
#         print(f"    Testing K = {k}...")
#         # n_init=10 is used here as a good practice
#         kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#         kmeans.fit(tfidf_matrix_reduced) # Fit on reduced matrix
#         inertia_values.append(kmeans.inertia_)

#     plt.figure(figsize=(10, 6))
#     plt.plot(k_values, inertia_values, marker='o')
#     plt.title('Elbow Method for Optimal K')
#     plt.xlabel('Number of Clusters (K)')
#     plt.ylabel('Inertia (Sum of squared distances)')
#     plt.xticks(k_values)
#     plt.grid(True)
#     plt.savefig(data_dir / 'elbow_method_plot.png') # Save the plot for review
#     plt.show()

#     # This part is for you to manually observe the plot and set optimal_k
#     # For now, it defaults to 10 as in your previous code.
#     optimal_k_from_elbow = 10 # Default for now, update after observing plot

# print("-" * 50)

# # --- Explicitly set optimal_k here as per your choice (or based on Elbow Method) ---
# # After reviewing the elbow plot, you should adjust this `optimal_k` value.
# optimal_k = 10 # Manually set K based on your analysis of the elbow plot.
# print(f"\n✅ Optimal K is manually set to: {optimal_k}")
# print("-" * 50)


# # --- 5. Apply K-Means Clustering with Selected K ---
# if optimal_k < 2:
#     print("❌ Cannot apply K-Means: Optimal K is less than 2. Please set K to 2 or higher.")
# elif tfidf_matrix_reduced.shape[0] < optimal_k:
#     print(f"❌ Cannot apply K-Means: Number of samples ({tfidf_matrix_reduced.shape[0]}) is less than optimal K ({optimal_k}).")
# else:
#     print(f"\n--- Applying K-Means Clustering for K={optimal_k} ---")
#     # n_init=10 ensures multiple runs for better centroid initialization
#     kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
#     kmeans_model.fit(tfidf_matrix_reduced) # Fit K-Means on the REDUCED matrix

#     cluster_labels = kmeans_model.labels_

#     joblib.dump(kmeans_model, data_dir / "kmeans_model.joblib")
#     joblib.dump(cluster_labels, data_dir / "document_cluster_labels.joblib")
#     print(f"✅ Clustering complete for {optimal_k} clusters.")
#     print(f"   K-Means model and cluster labels saved to: {data_dir}")
#     print("-" * 50)

#     # --- 6. Visualize Clusters (Cluster Distribution Chart) ---
#     print("\n--- Creating Cluster Distribution Plot ---")
    
#     # Perform 2D SVD for visualization on the already reduced matrix
#     svd_plot = TruncatedSVD(n_components=2, random_state=42)
    
#     # Ensure there are enough features for 2D plotting
#     if tfidf_matrix_reduced.shape[1] >= 2:
#         reduced_features_for_plot = svd_plot.fit_transform(tfidf_matrix_reduced)
#     else:
#         # If the matrix was already 1D, make it 2D with zeros for visualization
#         if tfidf_matrix_reduced.shape[1] == 1:
#             reduced_features_for_plot = np.hstack((tfidf_matrix_reduced, np.zeros((tfidf_matrix_reduced.shape[0], 1))))
#         else: # No features or already 0D
#             print("⚠️ Not enough features for 2D visualization. Skipping cluster plot.")
#             reduced_features_for_plot = None # Signal to skip plotting

#     if reduced_features_for_plot is not None:
#         plt.figure(figsize=(12, 10))
#         scatter = plt.scatter(reduced_features_for_plot[:, 0], reduced_features_for_plot[:, 1], 
#                               c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
#         plt.title(f'Document Clusters for {dataset_name} (TruncatedSVD Reduced to 2D for Visualization)')
#         plt.xlabel('Component 1 (from TruncatedSVD)')
#         plt.ylabel('Component 2 (from TruncatedSVD)')
#         plt.colorbar(scatter, label='Cluster ID')
#         plt.grid(True)
#         plt.savefig(data_dir / 'document_clusters_plot.png') # Save the cluster plot
#         plt.show()

#     # Display document count per cluster
#     unique_labels, counts = np.unique(cluster_labels, return_counts=True)
#     cluster_distribution = dict(zip(unique_labels, counts))
#     print("\nDocument Distribution per Cluster:")
#     for cluster_id, count in cluster_distribution.items():
#         print(f"   Cluster {cluster_id}: {count} documents")
#     print("-" * 50)

# print("\n--- Clustering process and visualization complete ---")
# print("You can now integrate clustering results (kmeans_model, svd_model, and cluster_labels) into your search function to accelerate search.")


# import os
# import sys
# import joblib
# from pathlib import Path

# from sklearn.cluster import KMeans
# from sklearn.decomposition import TruncatedSVD
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer

# # Set up Python path to enable correct imports for the project structure
# current_script_dir = Path(__file__).parent
# project_root_dir = current_script_dir.parent
# if str(project_root_dir) not in sys.path:
#     sys.path.insert(0, str(project_root_dir))

# from src.loader import load_dataset_with_queries
# from src.services.processing.preprocessing import preprocess_text

# # --- 1. Custom Tokenizer Setup for TF-IDF ---
# class CustomVectorizerTokenizer:
#     def __call__(self, text):
#         return preprocess_text(text)

# # --- 2. Load Trained TF-IDF Matrix (or train if not found) ---
# dataset_name = "quora"
# data_dir = Path(f"data/{dataset_name}")
# data_dir.mkdir(parents=True, exist_ok=True)

# tfidf_matrix_path = data_dir / "tfidf_matrix.joblib"
# docs_list_path = data_dir / "docs_list.joblib"
# tfidf_vectorizer_path = data_dir / "tfidf_vectorizer.joblib"

# docs = None
# tfidf_matrix = None
# tfidf_vectorizer = None

# print(f"--- Attempting to load TF-IDF from: {data_dir} ---")
# if tfidf_matrix_path.exists() and docs_list_path.exists() and tfidf_vectorizer_path.exists():
#     try:
#         tfidf_matrix = joblib.load(tfidf_matrix_path)
#         docs = joblib.load(docs_list_path)
#         tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
#         print("✅ TF-IDF matrix, document list, and vectorizer loaded successfully.")
#     except Exception as e:
#         print(f"❌ Error loading TF-IDF files: {e}. Re-training will be attempted.")
#         tfidf_matrix = None
# else:
#     print("⚠️ Trained TF-IDF files not found. Loading dataset and training TF-IDF...")

# if tfidf_matrix is None:
#     print(f"--- Loading dataset for '{dataset_name}' ---")
#     docs, _, _ = load_dataset_with_queries(dataset_name)

#     if not docs:
#         print(f"❌ No documents loaded for '{dataset_name}'. Cannot proceed with TF-IDF training.")
#         sys.exit(1)

#     print(f"--- Training TF-IDF Model for '{dataset_name}' ---")

#     tfidf_vectorizer = TfidfVectorizer(
#         tokenizer=CustomVectorizerTokenizer(),
#         min_df=5,
#         max_df=0.8,
#         max_features=50000,
#         sublinear_tf=True
#     )
    
#     documents_for_tfidf = [" ".join(preprocess_text(doc.text)) for doc in docs if hasattr(doc, 'text')]
    
#     if not documents_for_tfidf:
#         print("❌ No valid text content found in documents for TF-IDF training.")
#         sys.exit(1)

#     tfidf_matrix = tfidf_vectorizer.fit_transform(documents_for_tfidf)
    
#     joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)
#     joblib.dump(tfidf_matrix, tfidf_matrix_path)
#     joblib.dump(docs, docs_list_path)
#     print("✅ TF-IDF Model training complete and saved.")

# print("-" * 50)

# if tfidf_matrix is None or tfidf_matrix.shape[0] == 0:
#     print("❌ Cannot proceed: No TF-IDF matrix available or matrix is empty after loading or training.")
#     sys.exit(1)


# # --- 3. Apply Dimensionality Reduction (TruncatedSVD) ---
# n_components_for_clustering = 300 # Experiment with this value (e.g., 100, 200, 500)

# print(f"\n--- Applying TruncatedSVD with {n_components_for_clustering} components for clustering ---")
# if tfidf_matrix.shape[1] >= n_components_for_clustering:
#     svd_model = TruncatedSVD(n_components=n_components_for_clustering, random_state=42)
#     tfidf_matrix_reduced = svd_model.fit_transform(tfidf_matrix)
#     joblib.dump(svd_model, data_dir / "svd_model.joblib")
#     print(f"✅ TF-IDF matrix reduced from {tfidf_matrix.shape} to {tfidf_matrix_reduced.shape}")
# else:
#     print(f"⚠️ TF-IDF matrix has only {tfidf_matrix.shape[1]} features, which is less than n_components_for_clustering ({n_components_for_clustering}). Skipping SVD reduction for clustering and using original matrix.")
#     tfidf_matrix_reduced = tfidf_matrix
# print("-" * 50)


# # # --- 4. Determine Optimal K using the Elbow Method (Uncommented for testing) ---
# # print("\n--- Determining Optimal K using Elbow Method ---")
# # if tfidf_matrix_reduced.shape[0] < 2:
# #     print("⚠️ Document count is too low for effective Elbow Method. At least 2 documents needed.")
# #     optimal_k_from_elbow = 1
# # elif tfidf_matrix_reduced.shape[0] == 2:
# #     optimal_k_from_elbow = 2
# #     print("⚠️ Only 2 documents available. Setting K to 2.")
# # else:
# #     max_k_range = min(40, tfidf_matrix_reduced.shape[0] - 1)
# #     k_values = range(30, max_k_range + 1)
# #     inertia_values = []
    
# #     for k in k_values:
# #         print(f"\n--- Testing K = {k} ---") # Added for clarity
# #         kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
# #         kmeans.fit(tfidf_matrix_reduced)
# #         inertia_values.append(kmeans.inertia_)

# #         # Calculate and print cluster distribution for current k
# #         current_cluster_labels = kmeans.labels_
# #         unique_labels, counts = np.unique(current_cluster_labels, return_counts=True)
# #         cluster_distribution = dict(zip(unique_labels, counts))
# #         print("Document Distribution per Cluster for K =", k, ":")
# #         for cluster_id, count in cluster_distribution.items():
# #             print(f"   Cluster {cluster_id}: {count} documents")
# #         print("-" * 20) # Separator for each K

# #     plt.figure(figsize=(10, 6))
# #     plt.plot(k_values, inertia_values, marker='o')
# #     plt.title('Elbow Method for Optimal K')
# #     plt.xlabel('Number of Clusters (K)')
# #     plt.ylabel('Inertia (Sum of squared distances)')
# #     plt.xticks(k_values)
# #     plt.grid(True)
# #     plt.savefig(data_dir / 'elbow_method_plot.png')
# #     plt.show()

# #     optimal_k_from_elbow = 30 # Default for now, update after observing plot

# # print("-" * 50)

# optimal_k = 38   # Manually set K based on your analysis of the elbow plot.
# # print(f"\n✅ Optimal K is manually set to: {optimal_k}")
# # print("-" * 50)


# # --- 5. Apply K-Means Clustering with Selected K ---
# if optimal_k < 2:
#     print("❌ Cannot apply K-Means: Optimal K is less than 2. Please set K to 2 or higher.")
# elif tfidf_matrix_reduced.shape[0] < optimal_k:
#     print(f"❌ Cannot apply K-Means: Number of samples ({tfidf_matrix_reduced.shape[0]}) is less than optimal K ({optimal_k}).")
# else:
#     print(f"\n--- Applying K-Means Clustering for K={optimal_k} ---")
#     kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
#     kmeans_model.fit(tfidf_matrix_reduced)

#     cluster_labels = kmeans_model.labels_

#     joblib.dump(kmeans_model, data_dir / "kmeans_model.joblib")
#     joblib.dump(cluster_labels, data_dir / "document_cluster_labels.joblib")
#     print(f"✅ Clustering complete for {optimal_k} clusters.")
#     print(f"   K-Means model and cluster labels saved to: {data_dir}")
#     print("-" * 50)

#     # --- 6. Visualize Clusters (Cluster Distribution Chart) ---
#     print("\n--- Creating Cluster Distribution Plot ---")
    
#     svd_plot = TruncatedSVD(n_components=2, random_state=42)
    
#     if tfidf_matrix_reduced.shape[1] >= 2:
#         reduced_features_for_plot = svd_plot.fit_transform(tfidf_matrix_reduced)
#     else:
#         if tfidf_matrix_reduced.shape[1] == 1:
#             reduced_features_for_plot = np.hstack((tfidf_matrix_reduced, np.zeros((tfidf_matrix_reduced.shape[0], 1))))
#         else:
#             print("⚠️ Not enough features for 2D visualization. Skipping cluster plot.")
#             reduced_features_for_plot = None

#     if reduced_features_for_plot is not None:
#         plt.figure(figsize=(12, 10))
#         scatter = plt.scatter(reduced_features_for_plot[:, 0], reduced_features_for_plot[:, 1], 
#                               c=cluster_labels, cmap='viridis', s=50, alpha=0.8)
#         plt.title(f'Document Clusters for {dataset_name} (TruncatedSVD Reduced to 2D for Visualization)')
#         plt.xlabel('Component 1 (from TruncatedSVD)')
#         plt.ylabel('Component 2 (from TruncatedSVD)')
#         plt.colorbar(scatter, label='Cluster ID')
#         plt.grid(True)
#         plt.savefig(data_dir / 'document_clusters_plot.png')
#         plt.show()

#     unique_labels, counts = np.unique(cluster_labels, return_counts=True)
#     cluster_distribution = dict(zip(unique_labels, counts))
#     print("\nDocument Distribution per Cluster:")
#     for cluster_id, count in cluster_distribution.items():
#         print(f"   Cluster {cluster_id}: {count} documents")
#     print("-" * 50)

# print("\n--- Clustering process and visualization complete ---")
# print("You can now integrate clustering results (kmeans_model, svd_model, and cluster_labels) into your search function.")

import os
import sys
import time
import joblib
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
# Removed silhouette_score, davies_bouldin_score as per user request

# Set up Python path to enable correct imports for the project structure
current_script_dir = Path(__file__).parent
# Modified sys.path.insert to add the 'src' directory itself to the path
if str(current_script_dir) not in sys.path:
    sys.path.insert(0, str(current_script_dir))

# --- IMPORTANT FIX: Import preprocess_text directly ---
# Reverted import path as 'services' will now be directly accessible since 'src' is in sys.path
from services.processing.preprocessing import preprocess_text

# --- 2. Load Trained TF-IDF Matrix ---
dataset_name = "quora"
# CORRECTED: Make data_dir explicitly relative to current_script_dir to avoid CWD issues
data_dir = current_script_dir / "data" / dataset_name
data_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists for loading

tfidf_matrix_path = data_dir / "tfidf_matrix.joblib"
docs_list_path = data_dir / "docs_list.joblib"
tfidf_vectorizer_path = data_dir / "tfidf_vectorizer.joblib"

docs = None
tfidf_matrix = None
tfidf_vectorizer = None

print(f"--- Attempting to load TF-IDF models from: {data_dir.resolve()} ---") # Print resolved path for clarity

# Debug prints to confirm file existence
print(f"Checking existence of: {tfidf_vectorizer_path.resolve()} -> {tfidf_vectorizer_path.exists()}")
print(f"Checking existence of: {tfidf_matrix_path.resolve()} -> {tfidf_matrix_path.exists()}")
print(f"Checking existence of: {docs_list_path.resolve()} -> {docs_list_path.exists()}")


# Check if all necessary TF-IDF files exist before attempting to load
if not (tfidf_matrix_path.exists() and docs_list_path.exists() and tfidf_vectorizer_path.exists()):
    print(f"❌ Required TF-IDF files NOT FOUND in {data_dir.resolve()}. Please ensure they are trained and saved there.")
    sys.exit(1) # Exit if files are not not found

try:
    print("\nAttempting joblib.load for TF-IDF files...")
    tfidf_matrix = joblib.load(tfidf_matrix_path)
    docs = joblib.load(docs_list_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    print("✅ TF-IDF matrix, document list, and vectorizer loaded successfully.")
except Exception as e:
    print(f"❌ Critical Error loading TF-IDF files: {e}. This likely means the files are corrupted or incompatible with the current code definitions.")
    print("If you recently updated TextPreprocessor or how TfidfVectorizer was saved, you MUST delete these old .joblib files and regenerate them once by running your TF-IDF training script.")
    sys.exit(1) # Exit if loading fails

print("-" * 50)

if tfidf_matrix is None or tfidf_matrix.shape[0] == 0:
    print("❌ Cannot proceed: TF-IDF matrix is empty after loading. Please check your saved files.")
    sys.exit(1)

# --- 3. Apply Dimensionality Reduction (TruncatedSVD) ---
n_components_for_clustering = 200

print(f"\n--- Applying TruncatedSVD with {n_components_for_clustering} components for clustering ---")
svd_model_path = data_dir / "svd_model.joblib"
svd_model = None

# Check if SVD model already exists and load it
if svd_model_path.exists():
    try:
        print(f"Checking existence of: {svd_model_path.resolve()} -> {svd_model_path.exists()}")
        svd_model = joblib.load(svd_model_path)
        if svd_model.n_components != n_components_for_clustering:
            print(f"⚠️ Loaded SVD model has {svd_model.n_components} components, but requested {n_components_for_clustering}. Forcing SVD re-training.")
            svd_model = None
        else:
            print("✅ SVD model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading SVD model from {svd_model_path.resolve()}: {e}. Forcing SVD re-training.")
        svd_model = None

if svd_model is None: # Train SVD if not loaded or if components didn't match
    if tfidf_matrix.shape[1] >= n_components_for_clustering:
        svd_model = TruncatedSVD(n_components=n_components_for_clustering, random_state=42)
        tfidf_matrix_reduced = svd_model.fit_transform(tfidf_matrix)
        joblib.dump(svd_model, data_dir / "svd_model.joblib")
        print(f"✅ SVD Model trained and saved to {svd_model_path.resolve()}. TF-IDF matrix reduced from {tfidf_matrix.shape} to {tfidf_matrix_reduced.shape}")
    else:
        print(f"⚠️ TF-IDF matrix has only {tfidf_matrix.shape[1]} features, which is less than n_components_for_clustering ({n_components_for_clustering}). Skipping SVD reduction for clustering and using original matrix.")
        tfidf_matrix_reduced = tfidf_matrix
else: # If SVD model was loaded
    if tfidf_matrix.shape[1] >= n_components_for_clustering:
        tfidf_matrix_reduced = svd_model.transform(tfidf_matrix)
        print(f"✅ TF-IDF matrix transformed using loaded SVD model to {tfidf_matrix_reduced.shape}")
    else:
        print(f"⚠️ TF-IDF matrix has only {tfidf_matrix.shape[1]} features. Using original matrix as loaded SVD not applicable.")
        tfidf_matrix_reduced = tfidf_matrix

print("-" * 50)


# --- 4. Determine Optimal K using the Elbow Method ---
print("\n--- Determining Optimal K using Elbow Method (Inertia Only) ---")
if tfidf_matrix_reduced.shape[0] < 2:
    print("⚠️ Document count is too low for effective Elbow Method. At least 2 documents needed.")
    optimal_k_from_elbow = 1
elif tfidf_matrix_reduced.shape[0] == 2:
    optimal_k_from_elbow = 2
    print("⚠️ Only 2 documents available. Setting K to 2.")
else:
    max_k_range = min(50, tfidf_matrix_reduced.shape[0] - 1)
    k_values = range(2, max_k_range + 1)
    inertia_values = []

    print(f"Starting Elbow Method iteration for K from 8 to {max_k_range}...")
    for k in k_values:
        iteration_start_time = time.time()
        print(f"\n--- Testing K = {k} --- (Starting at {time.ctime()})")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=300)
        kmeans.fit(tfidf_matrix_reduced)
        inertia_values.append(kmeans.inertia_)
        print(f"  ✅ KMeans fitted for K = {k}. Inertia: {kmeans.inertia_:.2f}")

        # Removed silhouette_score and davies_bouldin_score calculations as per user request
        print(f"  (Skipping Silhouette and Davies-Bouldin Index calculations as requested.)")

        current_cluster_labels = kmeans.labels_
        unique_labels, counts = np.unique(current_cluster_labels, return_counts=True)
        cluster_distribution = dict(zip(unique_labels, counts))
        print("Document Distribution per Cluster for K =", k, ":")
        for cluster_id, count in sorted(cluster_distribution.items(), key=lambda item: item[1], reverse=True):
            print(f"    Cluster {cluster_id}: {count} documents")

        iteration_end_time = time.time()
        elapsed_k_time = iteration_end_time - iteration_start_time
        print(f"--- K = {k} processing complete in {elapsed_k_time:.2f} seconds ---")
        print("-" * 20)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia_values, marker='o')
    plt.title('Elbow Method for Optimal K (Based on Inertia)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Sum of squared distances)')
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig(data_dir / 'elbow_method_plot.png')
    plt.show()

    optimal_k_from_elbow = 20 # Manually setting for demonstration based on the new reduced range

print("-" * 50)

optimal_k = 38 # Manually set for demonstration based on the new reduced range
print(f"\n✅ Optimal K is manually set to: {optimal_k}")
print("-" * 50)


# --- 5. Apply K-Means Clustering with Selected K ---
if optimal_k < 2:
    print("❌ Cannot apply K-Means: Optimal K is less than 2. Please set K to 2 or higher.")
elif tfidf_matrix_reduced.shape[0] < optimal_k:
    print(f"❌ Cannot apply K-Means: Number of samples ({tfidf_matrix_reduced.shape[0]}) is less than optimal K ({optimal_k}).")
else:
    print(f"\n--- Applying K-Means Clustering for K={optimal_k} ---")
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=50, max_iter=300)
    kmeans_model.fit(tfidf_matrix_reduced)

    cluster_labels = kmeans_model.labels_

    joblib.dump(kmeans_model, data_dir / "kmeans_model.joblib")
    joblib.dump(cluster_labels, data_dir / "document_cluster_labels.joblib")
    print(f"✅ Clustering complete for {optimal_k} clusters.")
    print(f"    K-Means model and cluster labels saved to: {data_dir}")
    print("-" * 50)

    # --- 6. Visualize Clusters (Cluster Distribution Chart) ---
    print("\n--- Creating Cluster Distribution Plot ---")

    svd_plot = TruncatedSVD(n_components=2, random_state=42)

    if tfidf_matrix_reduced.shape[1] >= 2:
        reduced_features_for_plot = svd_plot.fit_transform(tfidf_matrix_reduced)
    else:
        if tfidf_matrix_reduced.shape[1] == 1:
            reduced_features_for_plot = np.hstack((tfidf_matrix_reduced, np.zeros((tfidf_matrix_reduced.shape[0], 1))))
        else:
            print("⚠️ Not enough features for 2D visualization. Skipping cluster plot.")
            reduced_features_for_plot = None

    if reduced_features_for_plot is not None:
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(reduced_features_for_plot[:, 0], reduced_features_for_plot[:, 1],
                                c=cluster_labels, cmap='viridis', s=50, alpha=0.8)

        centroids_for_plot = svd_plot.transform(kmeans_model.cluster_centers_)
        plt.scatter(centroids_for_plot[:, 0], centroids_for_plot[:, 1],
                            marker='X', s=200, c='red', edgecolor='black', label='Centroids')

        plt.title(f'Document Clusters for {dataset_name} (TruncatedSVD Reduced to 2D for Visualization)')
        plt.xlabel('Component 1 (from TruncatedSVD)')
        plt.ylabel('Component 2 (from TruncatedSVD)')
        plt.colorbar(scatter, label='Cluster ID')
        plt.legend()
        plt.grid(True)
        plt.savefig(data_dir / 'document_clusters_plot.png')
        plt.show()

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    cluster_distribution = dict(zip(unique_labels, counts))
    print("\nDocument Distribution per Cluster:")
    for cluster_id, count in sorted(cluster_distribution.items(), key=lambda item: item[1], reverse=True):
        print(f"    Cluster {cluster_id}: {count} documents")
    print("-" * 50)

print("\n--- Clustering process and visualization complete ---")
print("You can now integrate clustering results (kmeans_model, svd_model, and cluster_labels) into your search function.")










import os
import sys
import time
import joblib
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
# Re-added imports for Silhouette and Davies-Bouldin scores for automatic K selection
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.utils import resample # For sampling data for Silhouette score calculation

# Set up Python path to enable correct imports for the project structure
current_script_dir = Path(__file__).parent
project_root_dir = current_script_dir.parent # This is the parent of 'src'
if str(project_root_dir) not in sys.path:
    sys.path.insert(0, str(project_root_dir)) 

# --- IMPORTANT: Ensure services package is accessible ---
# This line is crucial. If 'services' is a top-level package inside 'src', 
# adding project_root_dir to sys.path makes 'src' implicitly accessible.
# The import below then works.
from services.processing.preprocessing import preprocess_text 
# If TextPreprocessor is a class, ensure TextPreprocessor.getInstance() is used in your custom tokenizer or preprocessing.


# --- 1. Custom Tokenizer Setup for TF-IDF (if needed for re-training) ---
# This part is commented out as the user requested to load pre-trained TF-IDF.
# If you ever need to re-train TF-IDF, uncomment this and ensure TextPreprocessor.getInstance() is accessible.
# class CustomVectorizerTokenizer:
#     def __init__(self):
#         self.preprocessor_instance = TextPreprocessor.getInstance()
#     def __call__(self, text):
#         processed_text = self.preprocessor_instance.preprocess_text(text)
#         return processed_text.split()


# --- 2. Load Trained TF-IDF Matrix ---
dataset_name = "antique" # Changed to "antique" based on your evaluation output
data_dir = current_script_dir / "data" / dataset_name
data_dir.mkdir(parents=True, exist_ok=True) 

tfidf_matrix_path = data_dir / "tfidf_matrix.joblib"
docs_list_path = data_dir / "docs_list.joblib"
tfidf_vectorizer_path = data_dir / "tfidf_vectorizer.joblib"

docs = None
tfidf_matrix = None
tfidf_vectorizer = None

print(f"--- Attempting to load TF-IDF models from: {data_dir.resolve()} ---") 

print(f"Checking existence of: {tfidf_vectorizer_path.resolve()} -> {tfidf_vectorizer_path.exists()}")
print(f"Checking existence of: {tfidf_matrix_path.resolve()} -> {tfidf_matrix_path.exists()}")
print(f"Checking existence of: {docs_list_path.resolve()} -> {docs_list_path.exists()}")

if not (tfidf_matrix_path.exists() and docs_list_path.exists() and tfidf_vectorizer_path.exists()):
    print(f"❌ Required TF-IDF files NOT FOUND in {data_dir.resolve()}. Please ensure they are trained and saved there.")
    print("Run your TF-IDF training script to generate these files.")
    sys.exit(1) 

try:
    print("\nAttempting joblib.load for TF-IDF files...")
    tfidf_matrix = joblib.load(tfidf_matrix_path)
    docs = joblib.load(docs_list_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    print("✅ TF-IDF matrix, document list, and vectorizer loaded successfully.")
except Exception as e:
    print(f"❌ Critical Error loading TF-IDF files: {e}. Please ensure the files are not corrupted or incompatible.")
    print("If you recently updated TextPreprocessor or TfidfVectorizer, you might need to delete old .joblib files and regenerate them.")
    sys.exit(1) 

print("-" * 50)

if tfidf_matrix is None or tfidf_matrix.shape[0] == 0:
    print("❌ Cannot proceed: TF-IDF matrix is empty after loading. Please check your saved files.")
    sys.exit(1)


# --- 3. Apply Dimensionality Reduction (TruncatedSVD) ---
# AUTOMATED SVD COMPONENT SELECTION
target_explained_variance = 0.80 # Target 80% explained variance (adjust based on RAM/performance)
max_svd_components_to_test = min(1500, tfidf_matrix.shape[1] - 1) # Cap at 1500 for 12GB RAM, or full features if less

print(f"\n--- Applying TruncatedSVD to achieve {target_explained_variance*100}% explained variance ---")
print(f"  Testing up to {max_svd_components_to_test} components.")

# Check if SVD model already exists and explains enough variance
svd_model_path = data_dir / "svd_model.joblib"
svd_model = None
n_components_for_clustering = None

if svd_model_path.exists():
    try:
        loaded_svd_model = joblib.load(svd_model_path)
        # Calculate explained variance for the loaded model if it has enough components
        if loaded_svd_model.n_components >= max_svd_components_to_test or \
           (hasattr(loaded_svd_model, 'explained_variance_ratio_') and np.sum(loaded_svd_model.explained_variance_ratio_) >= target_explained_variance):
            svd_model = loaded_svd_model
            n_components_for_clustering = svd_model.n_components
            print(f"✅ Existing SVD model loaded with {n_components_for_clustering} components.")
            print(f"  Explains {np.sum(svd_model.explained_variance_ratio_):.4f} variance.")
        else:
            print("⚠️ Existing SVD model insufficient variance/components. Retraining SVD.")
    except Exception as e:
        print(f"❌ Error loading SVD model: {e}. Retraining SVD.")

# If no suitable SVD model loaded, train a new one
if svd_model is None:
    # Perform SVD to find optimal n_components for desired variance
    svd_temp = TruncatedSVD(n_components=max_svd_components_to_test, random_state=42)
    svd_temp.fit(tfidf_matrix)

    cumulative_variance = np.cumsum(svd_temp.explained_variance_ratio_)
    
    # Find the number of components needed for the target variance
    n_components_for_clustering = np.where(cumulative_variance >= target_explained_variance)[0]
    if len(n_components_for_clustering) > 0:
        n_components_for_clustering = n_components_for_clustering[0] + 1 # +1 because index is 0-based
        # Cap components at max_svd_components_to_test if calculated value is higher
        n_components_for_clustering = min(n_components_for_clustering, max_svd_components_to_test)
        print(f"  Found {n_components_for_clustering} components explain >= {target_explained_variance*100:.0f}% variance.")
    else:
        # If target variance not reached, use max components tested
        n_components_for_clustering = max_svd_components_to_test
        print(f"  Target variance not reached. Using max components tested: {n_components_for_clustering}.")
    
    # Retrain SVD with the selected number of components
    svd_model = TruncatedSVD(n_components=n_components_for_clustering, random_state=42)
    tfidf_matrix_reduced = svd_model.fit_transform(tfidf_matrix)
    joblib.dump(svd_model, svd_model_path)
    
    print(f"✅ SVD Model trained and saved to {svd_model_path.resolve()}.")
    print(f"  TF-IDF matrix reduced from {tfidf_matrix.shape} to {tfidf_matrix_reduced.shape}")
    print(f"  Total variance explained: {np.sum(svd_model.explained_variance_ratio_):.4f}")

else: # If SVD model was loaded
    if tfidf_matrix.shape[1] >= n_components_for_clustering:
        tfidf_matrix_reduced = svd_model.transform(tfidf_matrix)
        print(f"✅ TF-IDF matrix transformed using loaded SVD model to {tfidf_matrix_reduced.shape}")
    else:
        print(f"⚠️ TF-IDF matrix has only {tfidf_matrix.shape[1]} features. Using original matrix as loaded SVD not applicable.")
        tfidf_matrix_reduced = tfidf_matrix

print("-" * 50)


# --- 4. Determine Optimal K using the Elbow Method (with Silhouette/Davies-Bouldin for auto-selection) ---
# AUTOMATED K SELECTION BASED ON SILHOUETTE SCORE
max_k_range = min(50, tfidf_matrix_reduced.shape[0] - 1) # Max K to test
k_values = range(2, max_k_range + 1)
inertia_values = []
silhouette_scores = []
davies_bouldin_scores = []
evaluated_k_data = [] # To store (k, inertia, silhouette, davies_bouldin) for analysis

print(f"\n--- Determining Optimal K (Automated Selection) ---")
print(f"  Testing K from 2 to {max_k_range}...")

# Sample data for Silhouette Score calculation (speeds up computation for large datasets)
sample_size = min(10000, tfidf_matrix_reduced.shape[0]) # Sample up to 10k documents
if tfidf_matrix_reduced.shape[0] > sample_size:
    print(f"  Sampling {sample_size} documents for Silhouette Score calculation.")
    sampled_data_for_silhouette = resample(tfidf_matrix_reduced, n_samples=sample_size, random_state=42)
else:
    sampled_data_for_silhouette = tfidf_matrix_reduced


for k in k_values:
    iteration_start_time = time.time()
    print(f"\n--- Testing K = {k} --- (Starting at {time.ctime()})")
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=300)
    kmeans.fit(tfidf_matrix_reduced) # Fit KMeans on the full reduced matrix
    inertia_values.append(kmeans.inertia_)
    print(f"  ✅ KMeans fitted for K = {k}. Inertia: {kmeans.inertia_:.2f}")

    # Calculate Silhouette and Davies-Bouldin Scores on sampled data
    current_silhouette = -1.0
    current_davies_bouldin = float('inf')

    if k > 1 and len(sampled_data_for_silhouette) >= k:
        # Predict labels for sampled data using the fitted KMeans model
        sampled_labels = kmeans.predict(sampled_data_for_silhouette)
        
        current_silhouette = silhouette_score(sampled_data_for_silhouette, sampled_labels)
        current_davies_bouldin = davies_bouldin_score(sampled_data_for_silhouette, sampled_labels)
    
    silhouette_scores.append(current_silhouette)
    davies_bouldin_scores.append(current_davies_bouldin)
    
    print(f"  ✅ Silhouette Score (sampled): {current_silhouette:.4f}")
    print(f"  ✅ Davies-Bouldin Index (sampled): {current_davies_bouldin:.4f}")

    current_cluster_labels = kmeans.labels_
    unique_labels, counts = np.unique(current_cluster_labels, return_counts=True)
    cluster_distribution = dict(zip(unique_labels, counts))
    print("Document Distribution per Cluster for K =", k, ":")
    for cluster_id, count in sorted(cluster_distribution.items(), key=lambda item: item[1], reverse=True):
        print(f"    Cluster {cluster_id}: {count} documents")

    evaluated_k_data.append({
        'k': k,
        'inertia': kmeans.inertia_,
        'silhouette': current_silhouette,
        'davies_bouldin': current_davies_bouldin,
        'distribution': cluster_distribution,
        'model': kmeans # Store the model temporarily for best K selection
    })

    iteration_end_time = time.time()
    elapsed_k_time = iteration_end_time - iteration_start_time
    print(f"--- K = {k} processing complete in {elapsed_k_time:.2f} seconds ---")
    print("-" * 20)

# --- Automatic Optimal K Selection ---
best_k_silhouette = -1
max_silhouette_score = -1.0
best_model_for_k = None

for k_data in evaluated_k_data:
    if k_data['silhouette'] > max_silhouette_score:
        max_silhouette_score = k_data['silhouette']
        best_k_silhouette = k_data['k']
        best_model_for_k = k_data['model']

# If all silhouette scores are -1 (e.g., K=1 or issues), default to a small K
if best_k_silhouette == -1 and evaluated_k_data:
    best_k_silhouette = evaluated_k_data[0]['k'] # Default to the first K tested
    best_model_for_k = evaluated_k_data[0]['model']


print(f"\n--- Automatic K Selection Results ---")
print(f"Optimal K (based on max Silhouette Score): {best_k_silhouette} (Score: {max_silhouette_score:.4f})")
print("-" * 50)


# --- Plotting Results (Elbow and Silhouette/Davies-Bouldin) ---
plt.figure(figsize=(10, 6))
plt.plot(k_values, inertia_values, marker='o')
plt.title('Elbow Method for Optimal K (Based on Inertia)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia (Sum of squared distances)')
plt.xticks(k_values)
plt.grid(True)
plt.savefig(data_dir / 'elbow_method_plot.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Optimal K (Sampled Data)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Average Silhouette Score')
plt.xticks(k_values)
plt.grid(True)
plt.savefig(data_dir / 'silhouette_scores_plot.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(k_values, davies_bouldin_scores, marker='o')
plt.title('Davies-Bouldin Index for Optimal K (Sampled Data)')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Davies-Bouldin Index')
plt.xticks(k_values)
plt.grid(True)
plt.savefig(data_dir / 'davies_bouldin_plot.png')
plt.show()


# --- 5. Apply K-Means Clustering with Selected K (Train and Save Final Model) ---
optimal_k = best_k_silhouette # Use automatically selected K
print(f"\n✅ Final Optimal K selected by system: {optimal_k}")
print("-" * 50)

if optimal_k < 2:
    print("❌ Cannot apply K-Means: Optimal K is less than 2. Please review K selection logic.")
    sys.exit(1)
elif tfidf_matrix_reduced.shape[0] < optimal_k:
    print(f"❌ Cannot apply K-Means: Number of samples ({tfidf_matrix_reduced.shape[0]}) is less than optimal K ({optimal_k}).")
    sys.exit(1)
else:
    print(f"\n--- Training and Saving Final K-Means Clustering Model for K={optimal_k} ---")
    # Use the best model found during elbow method if available, otherwise train new
    if best_model_for_k and best_model_for_k.n_clusters == optimal_k:
        kmeans_model = best_model_for_k
        print("  Using pre-fitted model for optimal K.")
    else:
        # Fallback: re-train if best_model_for_k is not the one for optimal_k
        print("  Re-fitting K-Means model for optimal K.")
        kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=50, max_iter=300)
        kmeans_model.fit(tfidf_matrix_reduced)

    cluster_labels = kmeans_model.labels_

    # Save with specific K in filename for clarity
    joblib.dump(kmeans_model, data_dir / f"kmeans_model_k{optimal_k}.joblib")
    joblib.dump(cluster_labels, data_dir / f"document_cluster_labels_k{optimal_k}.joblib")
    print(f"✅ Clustering complete for {optimal_k} clusters.")
    print(f"    K-Means model and cluster labels saved to: {data_dir}/kmeans_model_k{optimal_k}.joblib and document_cluster_labels_k{optimal_k}.joblib")
    print("-" * 50)

    # --- 6. Visualize Clusters (Cluster Distribution Chart) ---
    print("\n--- Creating Cluster Distribution Plot ---")

    svd_plot = TruncatedSVD(n_components=2, random_state=42)

    if tfidf_matrix_reduced.shape[1] >= 2:
        reduced_features_for_plot = svd_plot.fit_transform(tfidf_matrix_reduced)
    else:
        if tfidf_matrix_reduced.shape[1] == 1:
            reduced_features_for_plot = np.hstack((tfidf_matrix_reduced, np.zeros((tfidf_matrix_reduced.shape[0], 1))))
        else:
            print("⚠️ Not enough features for 2D visualization. Skipping cluster plot.")
            reduced_features_for_plot = None

    if reduced_features_for_plot is not None:
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(reduced_features_for_plot[:, 0], reduced_features_for_plot[:, 1],
                                c=cluster_labels, cmap='viridis', s=50, alpha=0.8)

        centroids_for_plot = svd_plot.transform(kmeans_model.cluster_centers_)
        plt.scatter(centroids_for_plot[:, 0], centroids_for_plot[:, 1],
                                marker='X', s=200, c='red', edgecolor='black', label='Centroids')

        plt.title(f'Document Clusters for {dataset_name} (TruncatedSVD Reduced to 2D for Visualization)')
        plt.xlabel('Component 1 (from TruncatedSVD)')
        plt.ylabel('Component 2 (from TruncatedSVD)')
        plt.colorbar(scatter, label='Cluster ID')
        plt.legend()
        plt.grid(True)
        plt.savefig(data_dir / 'document_clusters_plot.png')
        plt.show()

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    cluster_distribution = dict(zip(unique_labels, counts))
    print("\nDocument Distribution per Cluster:")
    for cluster_id, count in sorted(cluster_distribution.items(), key=lambda item: item[1], reverse=True):
        print(f"    Cluster {cluster_id}: {count} documents")
    print("-" * 50)

print("\n--- Clustering process and visualization complete ---")
print("You can now integrate clustering results (kmeans_model, svd_model, and cluster_labels) into your search function.")


