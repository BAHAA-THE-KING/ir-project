# import time
# from config import DEFAULT_DATASET
# from loader import load_dataset_with_queries
# from services.online_vectorizers.bm25 import BM25_online
# from services.online_vectorizers.tfidf import TFIDF_online

# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    

# def main():
#     dataset_name = DEFAULT_DATASET
#     docs, queries, qrels = load_dataset_with_queries(dataset_name)
#     retriever = TFIDF_online()

#     retriever.__loadInstance__(dataset_name)
#     retriever.__loadInvertedIndex__(dataset_name)
#     print("search started")
    
#     retriever.evaluateNDCG(dataset_name, queries, qrels, docs, 10)
#     # retriever.evaluateMAP()

#     start_time = time.time()
#     results = retriever.search(dataset_name, "saddam", 10, True)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Function execution time: {elapsed_time:.4f} seconds")

#     start_time = time.time()
#     results = retriever.search(dataset_name, "politicians", 10, True)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Function execution time: {elapsed_time:.4f} seconds")

    # st = time.time()
    # print(BM25_online().evaluateMRR(dataset_name, queries, qrels))
    # print(f"{time.time() - st}s")

#     start_time = time.time()
#     results = retriever.search(dataset_name, "please don't", 10)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Function execution time: {elapsed_time:.4f} seconds")

    # st = time.time()
    # print(f"MAP= {Hybrid_online().evaluateMAP(dataset_name, queries, qrels, docs)}")
    # print(f"{time.time() - st}s")

#     print("\nSearch Results (Ranked by Relevance):")
#     for res in results:
#         print(f"  Doc ID: {res[0]}, Score: {res[1]:.4f}, Text: '{res[2]}'")
#     print("-" * 60)

    
#     # retriever = BM25_online()
#     # MRR = retriever.evaluateMRR(dataset_name, queries, qrels)
#     # MAP = retriever.evaluateMAP(dataset_name, queries, qrels)

#     # print(f"MAP={MAP}")
#     # print(f"MRR={MRR}")
    
# if __name__ == "__main__":
#     main()


# import time
# import os 
# from config import DEFAULT_DATASET
# from loader import load_dataset_with_queries

# from services.processing.text_preprocessor import TextPreprocessor
# from services.online_vectorizers.tfidf import TFIDF_online
# from services.online_vectorizers.cluster_search import ClusterSearch
# from pathlib import Path # ADDED: Import Path

# import sys

# # Ensure project root is in sys.path for correct module imports
# current_script_dir = Path(__file__).parent
# project_root_dir = current_script_dir.parent
# if str(project_root_dir) not in sys.path:
#     sys.path.insert(0, str(project_root_dir)) 


# def main():
#     dataset_name = DEFAULT_DATASET # "antique" as per your config
#     # model_path should point to src/data/antique
#     model_path = os.path.join(os.path.dirname(__file__), "data", dataset_name)
    
#     # Instantiate TextPreprocessor once (singleton pattern) - important for all processing
#     text_preprocessor_instance = TextPreprocessor.getInstance()

#     print(f"--- Loading dataset: {dataset_name} ---")
#     docs, queries, qrels = load_dataset_with_queries(dataset_name)
#     print(f"✅ Dataset loaded. Docs: {len(docs)}, Queries: {len(queries)}, Qrels: {len(qrels)}")

#     # --- Initialize Search Models ---
#     print("\n--- Initializing Standard TF-IDF Searcher ---")
#     tfidf_searcher = TFIDF_online()
#     tfidf_searcher.__loadInstance__(dataset_name) # Ensure TFIDF models are loaded
#     # __loadInvertedIndex__ is called inside __loadInstance__ now, no need to call separately
#     print("✅ Standard TF-IDF Searcher ready.")

#     print("\n--- Initializing Cluster-Based Searcher ---")
#     # Pass the model_path to ClusterSearch
#     # Assuming kmeans_model_k8.joblib and document_cluster_labels_k8.joblib exist.
#     # You can change kmeans_model_k_value to 3 if you prefer K=3.
#     cluster_searcher = ClusterSearch(model_path, dataset_name=dataset_name, kmeans_model_k_value=3) 
#     if not cluster_searcher.models_loaded:
#         print("❌ ClusterSearch models failed to load. Skipping evaluation for ClusterSearch.")
#         cluster_searcher = None # Set to None if not loaded

#     # --- Define Test Queries ---
#     # Using a subset of queries for performance evaluation (e.g., first 50 queries)
#     num_test_queries = min(50, len(queries)) # Test 50 queries or fewer if dataset is smaller
#     test_queries = queries[:num_test_queries]
#     if not test_queries:
#         print("❌ No test queries loaded. Cannot perform search evaluation.")
#         return

#     # --- Performance and Evaluation for Standard TF-IDF ---
#     print("\n--- Evaluating Standard TF-IDF Search Performance & Quality ---")
#     tfidf_search_times = []
    
#     for i, query_obj in enumerate(test_queries):
#         query_text = query_obj.text
#         start_time = time.time()
#         # Set with_index=False to ensure full matrix search (as TFIDF_online.search no longer uses inverted index)
#         results = tfidf_searcher.search(dataset_name, query_text, 10, with_index=False) 
#         end_time = time.time()
#         tfidf_search_times.append(end_time - start_time)
#         print(f"  TF-IDF Query {i+1}/{len(test_queries)} ('{query_text[:30]}...') took {tfidf_search_times[-1]:.4f} seconds.")

#     avg_tfidf_time = sum(tfidf_search_times) / len(tfidf_search_times)
#     print(f"\nAverage Standard TF-IDF Search Time (across {len(test_queries)} queries): {avg_tfidf_time:.4f} seconds.")

#     # Evaluate quality for Standard TF-IDF (using the full set of queries for official evaluation)
#     # The 'evaluate' methods are passed with_index=False to ensure they use full matrix search
#     print("\nCalculating TF-IDF Evaluation Metrics...")
#     tfidf_ndcg = tfidf_searcher.evaluateNDCG(dataset_name, queries, qrels, docs, K=10, print_more=False, with_index=False)
#     tfidf_mrr = tfidf_searcher.evaluateMRR(dataset_name, queries, qrels, docs, K=100, print_more=False, with_index=False)
#     tfidf_map = tfidf_searcher.evaluateMAP(dataset_name, queries, qrels, docs, K=10, print_more=False, with_index=False)
    
#     print(f"  Standard TF-IDF nDCG@10: {tfidf_ndcg:.4f}")
#     print(f"  Standard TF-IDF MRR@100: {tfidf_mrr:.4f}")
#     print(f"  Standard TF-IDF MAP@10: {tfidf_map:.4f}")
#     print("-" * 60)

#     # --- Performance and Evaluation for Cluster-Based Search ---
#     if cluster_searcher is not None:
#         print("\n--- Evaluating Cluster-Based Search Performance & Quality ---")
#         cluster_search_times = []
        
#         for i, query_obj in enumerate(test_queries):
#             query_text = query_obj.text
#             start_time = time.time()
#             # ClusterSearch doesn't use inverted index, so with_index=False is appropriate
#             results = cluster_searcher.search(dataset_name, query_text, 10, with_index=False) 
#             end_time = time.time()
#             cluster_search_times.append(end_time - start_time)
#             print(f"  Cluster Search Query {i+1}/{len(test_queries)} ('{query_text[:30]}...') took {cluster_search_times[-1]:.4f} seconds.")

#         avg_cluster_time = sum(cluster_search_times) / len(cluster_search_times)
#         print(f"\nAverage Cluster-Based Search Time (across {len(test_queries)} queries): {avg_cluster_time:.4f} seconds.")

#         # Evaluate quality for Cluster-Based Search (using the full set of queries)
#         print("\nCalculating Cluster-Based Search Evaluation Metrics...")
#         cluster_ndcg = cluster_searcher.evaluateNDCG(dataset_name, queries, qrels, docs, K=10, print_more=False, with_index=False)
#         cluster_mrr = cluster_searcher.evaluateMRR(dataset_name, queries, qrels, docs, K=100, print_more=False, with_index=False)
#         cluster_map = cluster_searcher.evaluateMAP(dataset_name, queries, qrels, docs, K=10, print_more=False, with_index=False)
#         print(f"  Cluster-Based Search nDCG@10: {cluster_ndcg:.4f}")
#         print(f"  Cluster-Based Search MRR@100: {cluster_mrr:.4f}")
#         print(f"  Cluster-Based Search MAP@10: {cluster_map:.4f}")
#         print("-" * 60)

#     print("\n--- All evaluations complete ---")

# if __name__ == "__main__":
#     main()


# import time
# from config import DEFAULT_DATASET
# from loader import load_dataset_with_queries
# from services.online_vectorizers.bm25 import BM25_online
# from services.online_vectorizers.tfidf import TFIDF_online
# from services.processing.text_preprocessor import TextPreprocessor

# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    

# def main():
#     dataset_name = DEFAULT_DATASET
#     docs, queries, qrels = load_dataset_with_queries(dataset_name)
#     retriever = TFIDF_online()

#     retriever.__loadInstance__(dataset_name)
#     # retriever.__loadInvertedIndex__(dataset_name)
#     print("search started")
    
#     retriever.evaluateNDCG(dataset_name, queries, qrels, docs, 10)
#     # retriever.evaluateMAP()

#     start_time = time.time()
#     results = retriever.search(dataset_name, "saddam", 10, True)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Function execution time: {elapsed_time:.4f} seconds")

#     start_time = time.time()
#     results = retriever.search(dataset_name, "politicians", 10, True)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Function execution time: {elapsed_time:.4f} seconds")


#     start_time = time.time()
#     results = retriever.search(dataset_name, "please don't", 10)
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print(f"Function execution time: {elapsed_time:.4f} seconds")


#     print("\nSearch Results (Ranked by Relevance):")
#     for res in results:
#         print(f"  Doc ID: {res[0]}, Score: {res[1]:.4f}, Text: '{res[2]}'")
#     print("-" * 60)

    
    # retriever = BM25_online()
    # MRR = retriever.evaluateMRR(dataset_name, queries, qrels)
    # MAP = retriever.evaluateMAP(dataset_name, queries, qrels)

    # print(f"MAP={MAP}")
    # print(f"MRR={MRR}")
    
# if __name__ == "__main__":
#     main()

    
    # retriever = BM25_online()
    # MRR = retriever.evaluateMRR(dataset_name, queries, qrels)
    # MAP = retriever.evaluateMAP(dataset_name, queries, qrels)

    # print(f"MAP={MAP}")
    # print(f"MRR={MRR}")

















# import time
# import os
# import sys
# from pathlib import Path

# # --- Setup Project Path ---
# current_script_dir = Path(__file__).parent
# project_root_dir = current_script_dir.parent
# if str(project_root_dir) not in sys.path:
#     sys.path.insert(0, str(project_root_dir))

# from config import DEFAULT_DATASET
# from loader import load_dataset_with_queries
# from services.processing.text_preprocessor import TextPreprocessor
# from services.online_vectorizers.cluster_search import ClusterSearch
# from services.online_vectorizers.tfidf import TFIDF_online # Assuming this is the correct path

# def main():
#     dataset_name = DEFAULT_DATASET
#     model_path = os.path.join(project_root_dir, "src", "data", dataset_name)
    
#     # Initialize TextPreprocessor
#     TextPreprocessor.getInstance()

#     # --- 1. Load Dataset ---
#     print(f"--- Loading dataset: {dataset_name} ---")
#     docs, queries, qrels = load_dataset_with_queries(dataset_name)
#     if not all([docs, queries, qrels]):
#         print("❌ Failed to load dataset. Exiting.")
#         return
#     print(f"✅ Dataset loaded. Docs: {len(docs)}, Queries: {len(queries)}, Qrels: {len(qrels)}")

#     # --- 2. Initialize Searchers ---
#     # Initialize Cluster-Based Searcher
#     print("\n--- Initializing Cluster-Based Searcher (k=12) ---")
#     cluster_searcher = ClusterSearch(model_path, dataset_name=dataset_name, kmeans_model_k_value=12)
#     if not cluster_searcher.models_loaded:
#         print("❌ ClusterSearch models failed to load. Skipping its evaluation.")
#         cluster_searcher = None

#     # Initialize Standard TF-IDF Searcher (without inverted index)
#     print("\n--- Initializing Standard TF-IDF Searcher ---")
#     tfidf_searcher = TFIDF_online()
#     try:
#         # The __loadInstance__ method will load the vectorizer, docs, and the full matrix
#         tfidf_searcher.__loadInstance__(dataset_name)
#         print("✅ Standard TF-IDF Searcher ready.")
#     except Exception as e:
#         print(f"❌ TF-IDF Searcher failed to load: {e}. Skipping its evaluation.")
#         tfidf_searcher = None

#     # --- 3. Performance Evaluation ---
#     num_test_queries = min(50, len(queries))
#     test_queries = queries[:num_test_queries]

#     # Performance for Cluster Search
#     if cluster_searcher:
#         print(f"\n--- Evaluating Cluster Search Performance (on {num_test_queries} queries) ---")
#         cluster_search_times = []
#         for i, query_obj in enumerate(test_queries):
#             start_time = time.time()
#             cluster_searcher.search(dataset_name, query_obj.text, top_k=10)
#             cluster_search_times.append(time.time() - start_time)
#         avg_cluster_time = sum(cluster_search_times) / len(cluster_search_times) if cluster_search_times else 0
#         print(f"✅ Average Cluster Search Time: {avg_cluster_time:.4f} seconds.")

#     # Performance for TF-IDF Search
#     if tfidf_searcher:
#         print(f"\n--- Evaluating Standard TF-IDF Performance (on {num_test_queries} queries) ---")
#         tfidf_search_times = []
#         for i, query_obj in enumerate(test_queries):
#             start_time = time.time()
#             # The `with_index=False` argument is not strictly necessary with your new code,
#             # but it's good practice to be explicit about wanting a full matrix scan.
#             tfidf_searcher.search(dataset_name, query_obj.text, top_k=10, with_index=False)
#             tfidf_search_times.append(time.time() - start_time)
#         avg_tfidf_time = sum(tfidf_search_times) / len(tfidf_search_times) if tfidf_search_times else 0
#         print(f"✅ Average Standard TF-IDF Search Time: {avg_tfidf_time:.4f} seconds.")

#     # --- 4. Quality Evaluation ---
#     print("\n" + "="*50)
#     print("--- Calculating Full Quality Metrics (on all queries) ---")
    
#     # Metrics for Cluster Search
#     if cluster_searcher:
#         print("\nCalculating Cluster Search Metrics...")
#         cluster_ndcg = cluster_searcher.evaluateNDCG(dataset_name, queries, qrels, docs, K=10)
#         cluster_mrr = cluster_searcher.evaluateMRR(dataset_name, queries, qrels, docs, K=100)
#         cluster_map = cluster_searcher.evaluateMAP(dataset_name, queries, qrels, docs, K=10)

#     # Metrics for TF-IDF Search
#     if tfidf_searcher:
#         print("\nCalculating Standard TF-IDF Metrics...")
#         tfidf_ndcg = tfidf_searcher.evaluateNDCG(dataset_name, queries, qrels, docs, K=10, with_index=False)
#         tfidf_mrr = tfidf_searcher.evaluateMRR(dataset_name, queries, qrels, docs, K=100, with_index=False)
#         tfidf_map = tfidf_searcher.evaluateMAP(dataset_name, queries, qrels, docs, K=10, with_index=False)
        
#     # --- 5. Final Comparison ---
#     print("\n" + "="*60)
#     print("--- FINAL EVALUATION RESULTS ---")
#     print("="*60)
#     print(f"{'Metric':<20} {'Cluster Search (k=12)':<25} {'Standard TF-IDF':<20}")
#     print("-"*60)
    
#     if cluster_searcher:
#         print(f"{'Avg Query Time (s)':<20} {avg_cluster_time:<25.4f}", end="")
#         if tfidf_searcher:
#             print(f"{avg_tfidf_time:<20.4f}")
#         else:
#             print()

#         print(f"{'nDCG@10':<20} {cluster_ndcg:<25.4f}", end="")
#         if tfidf_searcher:
#             print(f"{tfidf_ndcg:<20.4f}")
#         else:
#             print()

#         print(f"{'MRR@100':<20} {cluster_mrr:<25.4f}", end="")
#         if tfidf_searcher:
#             print(f"{cluster_mrr:<20.4f}")
#         else:
#             print()
            
#         print(f"{'MAP@10':<20} {cluster_map:<25.4f}", end="")
#         if tfidf_searcher:
#             print(f"{tfidf_map:<20.4f}")
#         else:
#             print()
            
#     elif tfidf_searcher:
#         # Case where only TF-IDF ran
#         print(f"{'Avg Query Time (s)':<20} {'N/A':<25} {avg_tfidf_time:<20.4f}")
#         print(f"{'nDCG@10':<20} {'N/A':<25} {tfidf_ndcg:<20.4f}")
#         print(f"{'MRR@100':<20} {'N/A':<25} {tfidf_mrr:<20.4f}")
#         print(f"{'MAP@10':<20} {'N/A':<25} {tfidf_map:<20.4f}")
        
#     print("="*60)


# if __name__ == "__main__":
#     main()



# # src/main.py
# import time
# import os
# import sys
# import numpy as np
# from statistics import mean

# # Adjust sys.path to ensure imports from src work correctly
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# from src.config import DEFAULT_DATASET
# from src.loader import load_dataset_with_queries
# from src.services.online_vectorizers.clustered_embedding import ClusteredEmbedding_online
# from src.services.processing.text_preprocessor import TextPreprocessor

# def main():
#     dataset_name = DEFAULT_DATASET
#     print(f"--- Loading Dataset: {dataset_name} ---")
#     docs, queries, qrels = load_dataset_with_queries(dataset_name)
#     print(f"Dataset loaded. Documents: {len(docs)}, Queries: {len(queries)}, Qrels: {len(qrels)}")
#     print("-" * 50)

#     # Initialize TextPreprocessor (for global use)
#     global preprocess_text
#     preprocess_text = TextPreprocessor.getInstance().preprocess_text

#     # Initialize ClusteredEmbedding_online (loads models and prints cluster distribution)
#     cluster_searcher = ClusteredEmbedding_online()
#     cluster_searcher._load_models(dataset_name)

#     # Evaluation config
#     TOP_K_EVAL = 10
#     SAMPLE_QUERY_COUNT = 50
#     if len(queries) > SAMPLE_QUERY_COUNT:
#         np.random.seed(42)
#         sample_indices = np.random.choice(len(queries), SAMPLE_QUERY_COUNT, replace=False)
#         eval_queries = [queries[i] for i in sample_indices]
#         eval_qrels = [qrel for qrel in qrels if qrel.query_id in [q.query_id for q in eval_queries]]
#         print(f"Evaluating on a sample of {SAMPLE_QUERY_COUNT} queries.")
#     else:
#         eval_queries = queries
#         eval_qrels = qrels
#         print(f"Evaluating on all {len(queries)} queries.")

#     # --- Clustered Search Evaluation ---
#     print("\n--- Evaluating SVD+KMeans Clustered Search ---")
#     single_query_test_runs = 5
#     time_clustered_single_runs = []
#     print(f"Measuring average search time over {single_query_test_runs} runs for a sample query...")
#     for _ in range(single_query_test_runs):
#         start_time = time.time()
#         _ = cluster_searcher.search(dataset_name, eval_queries[0].text, TOP_K_EVAL)
#         end_time = time.time()
#         time_clustered_single_runs.append(end_time - start_time)
#     time_clustered_single = mean(time_clustered_single_runs)
#     print(f"Time for single query (Clustered, avg over {single_query_test_runs} runs): {time_clustered_single:.4f} seconds")

#     # Debug: Show preprocessed query and assigned cluster for first sample query
#     preprocessed_query = preprocess_text(eval_queries[0].text)
#     print(f"Sample query: {eval_queries[0].text}")
#     print(f"Preprocessed: {preprocessed_query}")
#     # Get assigned cluster
#     _ = cluster_searcher._load_models(dataset_name)  # Ensure models loaded
#     tfidf_vectorizer = cluster_searcher._tfidf_vectorizer
#     svd_model = cluster_searcher._svd_model
#     kmeans_model = cluster_searcher._kmeans_model
#     assigned_cluster = None
#     if tfidf_vectorizer is not None and svd_model is not None and kmeans_model is not None:
#         query_tfidf = tfidf_vectorizer.transform([" ".join(preprocessed_query)])
#         query_svd = svd_model.transform(query_tfidf)
#         assigned_cluster = kmeans_model.predict(query_svd)[0]
#     print(f"Assigned cluster for sample query: {assigned_cluster}")

#     # Evaluate real IR metrics using Retriever's methods
#     print(f"\nCalculating IR metrics for Clustered search on {len(eval_queries)} queries...")
#     ndcg = cluster_searcher.evaluateNDCG(dataset_name, eval_queries, eval_qrels, docs, K=TOP_K_EVAL, print_more=False)
#     mrr = cluster_searcher.evaluateMRR(dataset_name, eval_queries, eval_qrels, K=TOP_K_EVAL, print_more=False)
#     map_score = cluster_searcher.evaluateMAP(dataset_name, eval_queries, eval_qrels, docs, K=TOP_K_EVAL, print_more=False)
#     print("\n--- IR Metrics ---")
#     print(f"nDCG@{TOP_K_EVAL}: {ndcg if ndcg is not None else 0:.2f}%")
#     print(f"MRR@{TOP_K_EVAL}: {mrr if mrr is not None else 0:.4f}")
#     print(f"MAP@{TOP_K_EVAL}: {map_score if map_score is not None else 0:.4f}")
#     print("-" * 50)

#     # Pick a sample query (e.g., the first one in eval_queries)
#     sample_query = eval_queries[0]
#     sample_query_id = sample_query.query_id
#     sample_query_text = sample_query.text

#     print("\n=== Sample Query Debug ===")
#     print(f"Query ID: {sample_query_id}")
#     print(f"Query Text: {sample_query_text}")

#     # Run search for this query
#     top_k = 10
#     results = cluster_searcher.search(dataset_name, sample_query_text, top_k)

#     print("\nTop 10 Results (DocID, Score):")
#     for doc_id, score, doc_text in results:
#         print(f"  DocID: {doc_id}, Score: {score:.4f}")

#     # Get relevant doc IDs from qrels for this query
#     relevant_doc_ids = [qrel.doc_id for qrel in eval_qrels if qrel.query_id == sample_query_id]
#     print(f"\nRelevant DocIDs from qrels for this query: {relevant_doc_ids}")

#     # Optionally, check if any of the top results are relevant
#     retrieved_doc_ids = [doc_id for doc_id, _, _ in results]
#     relevant_in_top = set(retrieved_doc_ids) & set(relevant_doc_ids)
#     print(f"\nRelevant DocIDs found in top {top_k}: {relevant_in_top}")
#     print("==========================\n")


# if __name__ == "__main__":
#     main()
