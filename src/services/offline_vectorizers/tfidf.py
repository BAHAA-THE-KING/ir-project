import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from services.processing.text_preprocessor import TextPreprocessor

# --- TF-IDF Training Function ---
def tfidf_train(docs, dataset_name):

    corpus = [doc.text for doc in docs]

    vectorizer = TfidfVectorizer(
        analyzer=TextPreprocessor.getInstance().preprocess_text,
        lowercase=False,  # Handled by preprocessor
        stop_words=None   # Handled by preprocessor
    )

    # Train the vectorizer (fit) on the documents
    tfidf_matrix = vectorizer.fit_transform(corpus)

    output_dir = f"data/{dataset_name}"
    # Create the directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the trained TF-IDF vectorizer
    joblib.dump(vectorizer, f"data/{dataset_name}/tfidf_vectorizer.joblib")

    # For search efficiency, also save the pre-computed TF-IDF matrix for documents
    joblib.dump(tfidf_matrix, f"data/{dataset_name}/tfidf_matrix.joblib")
    
    print(f"✅ TF-IDF Vectorizer trained and saved for dataset: {dataset_name}")
    print(f"   TF-IDF Matrix shape (num_docs, vocab_size): {tfidf_matrix.shape}")
