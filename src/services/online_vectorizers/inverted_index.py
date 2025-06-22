from collections import defaultdict

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)        # term -> {doc_id}
        self.doc_lengths = defaultdict(int)  # doc_id -> total terms
        self.N = 0                           # total documents

    def add_document(self, doc_id, tokens):
        self.N += 1
        self.doc_lengths[doc_id] = len(tokens)
        for token in tokens:
            self.index[token].add(doc_id)

    def get_documents_sharing_terms_with_query(self, query_tokens):
        """
        Returns a set of doc_ids that share at least one word with the query.
        """
        related_docs = set()

        for token in query_tokens:
            related_docs.update(self.index.get(token, set()))

        return list(related_docs)
