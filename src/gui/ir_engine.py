from typing import Dict, List, Tuple, Optional
from enum import Enum

from src.config import DATASETS, DEFAULT_DATASET
from src.loader import load_dataset_with_queries
from src.services.online_vectorizers.hybrid import hybrid_search
from src.services.online_vectorizers.bm25 import bm25_search
from src.services.online_vectorizers.tfidf import tfidf_search

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
    
    def _load_dataset(self, dataset_name: str) -> None:
        """Load a dataset by name."""
        print(f"Loading dataset: {dataset_name}")
        if dataset_name not in DATASETS:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        self.docs, self.queries, self.qrels = load_dataset_with_queries(dataset_name)
        self.current_dataset = dataset_name
        print(f"Dataset loaded successfully. Documents: {len(self.docs)}")
    
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
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the current dataset."""
        return {
            "name": self.current_dataset,
            "description": DATASETS[self.current_dataset]["description"],
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
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float, str]]:
        """
        Search the current dataset using the selected model.
        Returns a list of (doc_id, score) tuples.
        """
        print(f"Searching for query: {query} using model: {self.current_model.value}")
        if (self.current_model.value=='Hybrid'):
            return hybrid_search(query, self.docs, self.queries, self.qrels, top_k)
        elif (self.current_model.value=='BM25'):
            return bm25_search(self.current_dataset, query, top_k)
        elif (self.current_model.value=='TF-IDF'):
            return tfidf_search(self.current_dataset, query, top_k)
        else:
            raise ValueError(f"Model {self.current_model.value} not found")
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """Get the content of a specific document."""
        print(f"Getting document: {doc_id}")
        for doc in self.docs:
            if doc.doc_id == doc_id:
                return doc.text
        return None
