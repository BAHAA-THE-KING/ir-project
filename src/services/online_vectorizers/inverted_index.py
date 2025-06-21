import math
from collections import defaultdict
import joblib
import os

class InvertedIndex:
    @staticmethod
    def load(dataset_name) -> 'InvertedIndex':
        path = f"data/{dataset_name}/inverted_index.joblib"
        return joblib.load(path)

    def __init__(self, dataset_name):
        self.index = defaultdict(dict)       # term -> {doc_id: freq}
        self.doc_lengths = defaultdict(int)  # doc_id -> total terms
        self.N = 0                           # total documents
        self.dataset_name = dataset_name     # dataset name

    def add_document(self, doc_id, tokens):
        self.N += 1
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.index[token][doc_id] = self.index[token].get(doc_id, 0) + 1

    def get_idf(self, term):
        df = len(self.index.get(term, {}))  # document frequency
        if df == 0:
            return 0
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))  # BM25 idf

    def avg_doc_length(self):
        return sum(self.doc_lengths.values()) / self.N if self.N > 0 else 0

    def save(self):
        path = f"data/{self.dataset_name}/inverted_index.joblib"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'a').close()
        joblib.dump(self, path)

    def get_documents_sharing_terms_with_query(self, query_tokens):
        """
        Returns a set of doc_ids that share at least one word with the query.
        """
        related_docs = set()

        for token in query_tokens:
            related_docs.update(self.index.get(token, {}).keys())

        return related_docs

    def __repr__(self):
        return f"InvertedIndex(terms={len(self.index)}, docs={self.N})"
