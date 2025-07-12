from services.processing.text_preprocessor import TextPreprocessor
from services.online_vectorizers.tfidf import TFIDF_online
from services.online_vectorizers.embedding import Embedding_online
from services.online_vectorizers.bm25 import BM25_online
from services.online_vectorizers.hybrid import Hybrid_online

from typing import List, Tuple, Optional
from enum import Enum

class SearchModel(Enum):
    TFIDF = "TF-IDF"
    BM25 = "BM25"
    HYBRID = "Hybrid"
    EMBEDDING = "Embedding"

class ir_engine:
    def __init__(self):
        text = TextPreprocessor().getInstance()
        text.preprocess_text("init")
        
        tfidf = TFIDF_online()
        tfidf.__loadInstance__('antique')
        tfidf.__loadInstance__('quora')
        tfidf.__loadInvertedIndex__('antique')
        tfidf.__loadInvertedIndex__('quora')
        
        bm25 = BM25_online()
        bm25.__loadDocs__('antique')
        bm25.__loadDocs__('quora')
        bm25.__loadInstance__('antique')
        bm25.__loadInstance__('quora')
        bm25.__loadInvertedIndex__('antique')
        bm25.__loadInvertedIndex__('quora')

        embedding = Embedding_online()
        embedding.__loadInstance__('antique')
        embedding.__loadInstance__('quora')
        embedding.__loadModelInstance__()
        embedding.__get_collection__('antique')
        embedding.__get_collection__('quora')
        embedding.__loadDocs__('antique')
        embedding.__loadDocs__('quora')

    def search(self, model_name: str, dataset_name: str, query: str, top_k: int = 10,
            use_inverted_index: bool = False, use_vector_store: bool = False) -> List[Tuple[str, float, str]]:
        """
        Search the current dataset using the specified model and options.
        Returns a list of (doc_id, score, snippet) tuples.
        Args:
            model_name: The search model to use (e.g., "TF-IDF", "BM25", "Hybrid", "Embedding").
            query: The search query string.
            top_k: Number of top results to return.
            use_inverted_index: For TF-IDF/BM25, whether to use an inverted index.
            use_vector_store: For Embedding, whether to use a vector store.
            include_cluster_info: For future use (currently ignored).
        """
        print(f"Searching for query: '{query}' using model: {model_name} in dataset {dataset_name} (inverted_index={use_inverted_index}, vector_store={use_vector_store})")
        if use_inverted_index:
            TFIDF_online.with_index = True
            BM25_online.with_index = True
        else:
            TFIDF_online.with_index = False
            BM25_online.with_index = False
        
        if use_vector_store:
            Embedding_online.with_index = True
        else:
            Embedding_online.with_index = False
        
        if model_name == SearchModel.HYBRID.value:
            return Hybrid_online().search(dataset_name, query, top_k)
        elif model_name == SearchModel.BM25.value:
            return BM25_online().search(dataset_name, query, top_k)
        elif model_name == SearchModel.TFIDF.value:
            return TFIDF_online().search(dataset_name, query, top_k)
        elif model_name == SearchModel.EMBEDDING.value:
            return Embedding_online().search(dataset_name, query, top_k, with_index=use_vector_store)
        else:
            raise ValueError(f"Model {model_name} not supported for search.")
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """Get the content of a specific document by its doc_id."""
        print(f"Getting document: {doc_id}")
        doc_obj = self.docs.get(doc_id) # Efficient lookup using the dictionary
        if doc_obj:
            return doc_obj.text
        return None
    