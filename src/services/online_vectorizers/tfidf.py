import time
import dill
import joblib
import numpy as np
from loader import load_dataset
from src.services.processing.text_preprocessor import TextPreprocessor
import __main__
setattr(__main__, 'TextPreprocessor', TextPreprocessor)
from sklearn.metrics.pairwise import cosine_similarity
from services.online_vectorizers.Retriever import Retriever
from services.online_vectorizers.inverted_index import InvertedIndex

class TFIDF_online(Retriever):
    __tfidfInstance__ : dict[str, list] = {}
    __invertedIndex__ : dict[str, InvertedIndex] = {}

    @staticmethod
    def __loadInstance__(dataset_name : str):
        if dataset_name not in TFIDF_online.__tfidfInstance__.keys():

            # Load the model and the documents
            docs = load_dataset(dataset_name)
            vectorizer = joblib.load(f"data/{dataset_name}/tfidf_vectorizer.joblib")
            docs_tfidf_matrix = joblib.load(f"data/{dataset_name}/tfidf_matrix.joblib")

            TFIDF_online.__tfidfInstance__[dataset_name] = [docs,vectorizer,docs_tfidf_matrix]

    @staticmethod
    def __loadInvertedIndex__(dataset_name : str):
        if dataset_name not in TFIDF_online.__invertedIndex__.keys():
            with open(f"data/{dataset_name}/inverted_index.dill", "rb") as f:
                inverted_index = InvertedIndex()
                ii = dill.load(f)
                inverted_index.index = ii.index
                inverted_index.doc_lengths = ii.doc_lengths
                inverted_index.N = ii.N
                TFIDF_online.__invertedIndex__[dataset_name] = inverted_index

    def search(self, dataset_name, query, top_k, with_index = True):

        # Load the model and the index
        self.__loadInstance__(dataset_name)
        docs = self.__tfidfInstance__[dataset_name][0]
        vectorizer = self.__tfidfInstance__[dataset_name][1]
        docs_tfidf_matrix = self.__tfidfInstance__[dataset_name][2]
        
        self.__loadInvertedIndex__(dataset_name)
        inverted_index = self.__invertedIndex__[dataset_name]

        # Start the process
        query_vec = vectorizer.transform([query])
        
        if(with_index):
            tokenized_query = TextPreprocessor.getInstance().preprocess_text(query)
            candidate_indices = inverted_index.get_documents_sharing_terms_with_query(tokenized_query)   
            docs_tfidf_matrix = docs_tfidf_matrix[candidate_indices]

        cosine_sim = cosine_similarity(query_vec, docs_tfidf_matrix).flatten()

        ranked_indices = np.argsort(cosine_sim)[::-1]

        # Prepare structured results
        results = []
        # Limit results to a reasonable number for display/API response, e.g., top 10 or 20
        for i in ranked_indices[:top_k]:
            if(with_index):
                original_doc_idx = candidate_indices[i]
            else:
                original_doc_idx = i

            doc = docs[original_doc_idx]
            results.append((
                docs[original_doc_idx].doc_id,
                float(cosine_sim[i]),
                doc.text[:40] + "..." if len(doc.text) > 40 else doc.text # Provide a snippet
            ))
        return results
