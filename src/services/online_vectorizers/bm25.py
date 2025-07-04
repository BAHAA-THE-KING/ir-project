import math
import dill
from rank_bm25 import BM25Okapi
from loader import load_dataset
from services.online_vectorizers.inverted_index import InvertedIndex
from services.processing.text_preprocessor import TextPreprocessor
from services.online_vectorizers.Retriever import Retriever


class BM25_online(Retriever):
    __bm25instance__ : dict[str, BM25Okapi] = {}
    __invertedIndex__ : dict[str, InvertedIndex] = {}
    @staticmethod
    def __loadInstance__(dataset_name : str):
        if dataset_name not in BM25_online.__bm25instance__.keys():
            with open(f"data/{dataset_name}/bm25_model.dill", "rb") as f:
                BM25_online.__bm25instance__[dataset_name] = dill.load(f) 
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

    def search(self, dataset_name: str, query: str, top_k: int = 10, with_inverted_index: bool = True) -> list[tuple[str, float, str]]:
        # Load the model and the documents
        BM25_online.__loadInstance__(dataset_name)
        bm25 = BM25_online.__bm25instance__[dataset_name]
        docs = load_dataset(dataset_name)
        if with_inverted_index:
            BM25_online.__loadInvertedIndex__(dataset_name)
            inverted_index = BM25_online.__invertedIndex__[dataset_name]

        # Execute the query
        query_tokens = TextPreprocessor.getInstance().preprocess_text(query)

        if with_inverted_index:
            documents_sharing_terms_with_query = inverted_index.get_documents_sharing_terms_with_query(query_tokens)
            scores = bm25.get_batch_scores(query_tokens, documents_sharing_terms_with_query)
        else:
            scores = bm25.get_scores(query_tokens)

        # Sort the results
        if with_inverted_index:
            top_indices = sorted(list(enumerate(documents_sharing_terms_with_query)), key=lambda  elm: scores[elm[0]], reverse=True)[:top_k]
        else:
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []

        # Display the results
        if with_inverted_index:
            for elm in top_indices:
                text = docs[elm[1]].text
                results.append((docs[elm[1]].doc_id, scores[elm[0]], text))
        else:
            for idx in top_indices:
                text = docs[idx].text
                results.append((docs[idx].doc_id, scores[idx], text))
        
        return results
