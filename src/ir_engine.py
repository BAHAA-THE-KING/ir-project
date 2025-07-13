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
    HYBRID = "HYBRID"
    EMBEDDING = "EMBEDDING"

class ir_engine:
    def __init__(self, db_connector, loaded_docs):
        self.db_connector = db_connector
        self.loaded_docs = loaded_docs
        text = TextPreprocessor().getInstance()
        text.preprocess_text("init")
        self.docs = {}
        tfidf = TFIDF_online(db_connector=db_connector, docs=loaded_docs)
        tfidf.__loadInstance__('antique')
        tfidf.__loadInstance__('quora')
        tfidf.__loadInvertedIndex__('antique')
        tfidf.__loadInvertedIndex__('quora')
        bm25 = BM25_online(db_connector=db_connector, docs=loaded_docs)
        bm25.__loadDocs__('antique')
        bm25.__loadDocs__('quora')
        bm25.__loadInstance__('antique')
        bm25.__loadInstance__('quora')
        bm25.__loadInvertedIndex__('antique')
        bm25.__loadInvertedIndex__('quora')
        embedding = Embedding_online(db_connector=db_connector, docs=loaded_docs)
        embedding.__loadInstance__('antique')
        embedding.__loadInstance__('quora')
        embedding.__loadModelInstance__()
        embedding.__get_collection__('antique')
        embedding.__get_collection__('quora')
        embedding.__loadDocs__('antique')
        embedding.__loadDocs__('quora')
        hybrid = Hybrid_online(db_connector=db_connector, docs=loaded_docs)

    def search(self, model_name: str, dataset_name: str, query: str, top_k: int = 10,
            use_inverted_index: bool = False, use_vector_store: bool = False) -> list:
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
            return Hybrid_online(db_connector=self.db_connector, docs=self.loaded_docs).search(dataset_name, query, top_k)
        elif model_name == SearchModel.BM25.value:
            return BM25_online(db_connector=self.db_connector, docs=self.loaded_docs).search(dataset_name, query, top_k)
        elif model_name == SearchModel.TFIDF.value:
            return TFIDF_online(db_connector=self.db_connector, docs=self.loaded_docs).search(dataset_name, query, top_k)
        elif model_name == SearchModel.EMBEDDING.value:
            return Embedding_online(db_connector=self.db_connector, docs=self.loaded_docs).search(dataset_name, query, top_k)
        else:
            raise ValueError(f"Model {model_name} not supported for search.")
    
    def get_document(self, doc_id: str) -> Optional[str]:
        """Get the content of a specific document by its doc_id."""
        print(f"Getting document: {doc_id}")
        doc_obj = self.docs.get(doc_id) # Efficient lookup using the dictionary
        if doc_obj:
            return doc_obj.text
        return None
    