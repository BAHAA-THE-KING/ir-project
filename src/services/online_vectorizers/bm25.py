import math
import dill
from rank_bm25 import BM25Okapi
from src.loader import load_dataset, Doc
from src.services.online_vectorizers.inverted_index import InvertedIndex
from src.services.processing.text_preprocessor import TextPreprocessor
from src.services.online_vectorizers.Retriever import Retriever


class BM25_online(Retriever):
    __bm25instance__ : dict[str, BM25Okapi] = {}
    __invertedIndex__ : dict[str, InvertedIndex] = {}
    __docs__ : dict[str, list[Doc]] = {}
    @staticmethod
    def __loadInstance__(dataset_name : str):
        if dataset_name not in BM25_online.__bm25instance__.keys():
            try:
                with open(f"data/{dataset_name}/bm25_model.dill", "rb") as f:
                    BM25_online.__bm25instance__[dataset_name] = dill.load(f)
            except FileNotFoundError:
                print(f"[WARNING] BM25 model file not found for dataset '{dataset_name}'. Returning None.")
                return None
        return BM25_online.__bm25instance__.get(dataset_name, None)
    @staticmethod
    def __loadInvertedIndex__(dataset_name : str):
        if dataset_name not in BM25_online.__invertedIndex__.keys():
            with open(f"data/{dataset_name}/inverted_index.dill", "rb") as f:
                inverted_index = InvertedIndex()
                ii = dill.load(f)
                inverted_index.index = ii.index
                inverted_index.doc_lengths = ii.doc_lengths
                inverted_index.N = ii.N
                BM25_online.__invertedIndex__[dataset_name] = inverted_index
        return BM25_online.__invertedIndex__[dataset_name]
    @staticmethod
    def __loadDocs__(dataset_name : str):
        if dataset_name not in BM25_online.__docs__.keys():
            BM25_online.__docs__[dataset_name] = load_dataset(dataset_name)
        return BM25_online.__docs__[dataset_name]

    def search(self, dataset_name: str, query: str, top_k: int = 10) -> list[tuple[str, float, str]]:
        # Load the model and the documents
        bm25 = BM25_online.__loadInstance__(dataset_name)
        if bm25 is None:
            raise ValueError(f"BM25 model file not found for dataset '{dataset_name}'. Please train or provide the model file.")
        docs = BM25_online.__loadDocs__(dataset_name)
        if BM25_online.with_index:
            inverted_index = BM25_online.__loadInvertedIndex__(dataset_name)

        # Execute the query
        query_tokens = TextPreprocessor.getInstance().preprocess_text(query)

        if BM25_online.with_index:
            documents_sharing_terms_with_query = inverted_index.get_documents_sharing_terms_with_query(query_tokens)
            scores = bm25.get_batch_scores(query_tokens, documents_sharing_terms_with_query)
        else:
            scores = bm25.get_scores(query_tokens)

        # Sort the results
        if BM25_online.with_index:
            top_indices = sorted(list(enumerate(documents_sharing_terms_with_query)), key=lambda  elm: scores[elm[0]], reverse=True)[:top_k]
        else:
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []

        # Display the results
        if BM25_online.with_index:
            # Create a mapping from doc_id to doc for fast lookup
            doc_id_to_doc = {doc.doc_id: doc for doc in docs}
            for score_idx, doc_id_idx in top_indices:
                doc_id = documents_sharing_terms_with_query[doc_id_idx]
                doc = doc_id_to_doc.get(doc_id)
                if doc is not None:
                    text = doc.text
                    results.append((doc.doc_id, scores[score_idx], text))
        else:
            for idx in top_indices:
                doc = docs[idx]
                text = doc.text
                results.append((doc.doc_id, scores[idx], text))
        
        return results
