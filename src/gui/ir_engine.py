import sys
import os
from typing import Dict, List, Tuple, Optional
from enum import Enum

# Add the parent directory to the Python path
# This assumes ir_engine.py is in src/gui/, so parent is src/, and grand-parent is project root.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import DATASETS, DEFAULT_DATASET
from src.loader import load_dataset_with_queries # Ensure this import is correct based on your loader.py
from src.services.online_vectorizers.bm25 import BM25_online # Import the class
from src.services.online_vectorizers.tfidf import TFIDF_online # Import the class
from src.services.online_vectorizers.embedding import Embedding_online # Import the class


class SearchModel(Enum):
    TFIDF = "TF-IDF"
    BM25 = "BM25"
    HYBRID = "Hybrid"
    EMBEDDING = "Embedding"

class IREngine:
    def __init__(self):
        print("Initializing IREngine...")
        self.docs: Dict = {}
        self.queries: Dict = {}
        self.qrels: Dict = {}
        self.current_dataset: str = DEFAULT_DATASET
        self.current_model: SearchModel = SearchModel.TFIDF
        self._load_dataset(DEFAULT_DATASET)
        print("IREngine initialized successfully")
        
    def search(self, model_name: str, query: str, top_k: int = 10,
            use_inverted_index: bool = False, use_vector_store: bool = False,
            include_cluster_info: bool = False) -> List[Tuple[str, float, str]]:
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
        print(f"Searching for query: '{query}' using model: {model_name} (inverted_index={use_inverted_index}, vector_store={use_vector_store}, cluster_info={include_cluster_info})")
        if model_name == SearchModel.HYBRID.value:
            # Hybrid search: rerank, then fetch snippet for each doc
            from src.services.online_vectorizers.hybrid import Hybrid_online
            hybrid_service = Hybrid_online()
            hybrid_results = hybrid_service.search(self.current_dataset, query, top_k, with_index=True)
            results = []
            for item in hybrid_results:
                if len(item) == 2:
                    doc_id, score = item
                    snippet = self.docs[doc_id].text[:80] + "..." if doc_id in self.docs else ""
                elif len(item) == 3:
                    doc_id, score, snippet = item
                else:
                    continue
                results.append((doc_id, score, snippet))
            return results
        elif model_name == SearchModel.BM25.value:
            return BM25_online().search(self.current_dataset, query, top_k, with_inverted_index=use_inverted_index)
        elif model_name == SearchModel.TFIDF.value:
            return TFIDF_online().search(self.current_dataset, query, top_k, with_index=use_inverted_index)
        elif model_name == SearchModel.EMBEDDING.value:
            embedding_results = Embedding_online().search(self.current_dataset, query, top_k, with_index=use_vector_store)
            if not embedding_results or not isinstance(embedding_results, list):
                return []
            results = []
            for item in embedding_results:
                if len(item) == 2:
                    doc_id, score = item
                    snippet = self.docs[doc_id].text[:80] + "..." if doc_id in self.docs else ""
                elif len(item) == 3:
                    doc_id, score, snippet = item
                else:
                    continue
                results.append((doc_id, score, snippet))
            return results
        else:
            raise ValueError(f"Model {model_name} not supported for search.")
    
    def _load_dataset(self, dataset_name: str) -> None:
        """Load a dataset by name."""
        print(f"Loading dataset: {dataset_name}")
        if dataset_name not in DATASETS:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        # Load docs, queries, and qrels
        docs_list, queries, qrels = load_dataset_with_queries(dataset_name)
        
        # Convert the list of Docs to a dictionary for efficient lookup by doc_id
        self.docs = {doc.doc_id: doc for doc in docs_list}
        # Convert queries and qrels to dicts for hybrid_search compatibility
        self.queries = {q.query_id: q for q in queries}
        self.qrels = {(q.query_id, q.doc_id): q for q in qrels}
        self.current_dataset = dataset_name
        print(f"Dataset loaded successfully. Documents: {len(self.docs)}")
        # Ensure TFIDF model is loaded for this dataset
        TFIDF_online.__loadInstance__(dataset_name)
        # Ensure BM25 model is loaded for this dataset
        from src.services.online_vectorizers.bm25 import BM25_online
        BM25_online.__loadInstance__(dataset_name)
        # Ensure Embedding model is loaded for this dataset
        from src.services.online_vectorizers.embedding import Embedding_online
        Embedding_online.__loadModelInstance__()
        Embedding_online.__loadInstance__(dataset_name)
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names."""
        return list(DATASETS.keys())
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get information about a specific dataset."""
        if dataset_name not in DATASETS:
            raise ValueError(f"Dataset {dataset_name} not found")
        return DATASETS[dataset_name]
    
    def change_dataset(self, dataset_name: str) -> None:
        """Change the current dataset."""
        self._load_dataset(dataset_name)
    
    def get_dataset_stats(self, dataset_name: Optional[str] = None) -> Dict:
        """Get statistics about the current or specified dataset."""
        if dataset_name is None:
            dataset_name = self.current_dataset
        return {
            "name": dataset_name,
            "description": DATASETS[dataset_name]["description"],
            "num_docs": len(self.docs),
            "num_queries": len(self.queries),
            "num_qrels": len(self.qrels)
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available search models."""
        return [model.value for model in SearchModel]
    
    def change_model(self, model_name: str) -> None:
        """Change the current search model."""
        try:
            self.current_model = SearchModel(model_name)
            print(f"Changed search model to: {model_name}")
        except ValueError:
            raise ValueError(f"Model {model_name} not found")
    
    def get_current_model(self) -> str:
        """Get the current search model name."""
        return self.current_model.value
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """Get the content of a specific document by its doc_id."""
        print(f"Getting document: {doc_id}")
        doc_obj = self.docs.get(doc_id) # Efficient lookup using the dictionary
        if doc_obj:
            return doc_obj.text
        return None