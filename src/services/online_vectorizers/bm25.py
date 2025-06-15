import joblib
from services.processing.preprocessing import TextPreprocessor

def bm25_search(dataset_name: str, query: str, top_k: int = 10):
    # Load the model and the documents
    bm25 = joblib.load(f"data/{dataset_name}/bm25_model.joblib")
    docs = joblib.load(f"data/{dataset_name}/docs_list.joblib")

    # Execute the query
    preprocessor = TextPreprocessor()
    query_tokens = preprocessor.preprocess_text(query).split()
    scores = bm25.get_scores(query_tokens)

    # Sort the results
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    # Display the results
    print("\n✅ Best results:")
    for rank, idx in enumerate(top_indices, 1):
        # snippet = docs[idx][:80].replace("\n", " ") + ("..." if len(docs[idx]) > 80 else "")
        print(f"{rank}. [Score: {scores[idx]:.4f}] {docs[idx]}")