from services.online_vectorizers.Retriever import Retriever
from services.online_vectorizers.tfidf import TFIDF_online
from services.online_vectorizers.bm25 import BM25_online
from services.online_vectorizers.embedding import Embedding_online
from src.loader import load_dataset

class Hybrid_online(Retriever):
    def __init__(self, db_connector, docs):
        self.db = db_connector
        self.docs = docs
    def __loadDocs__(self, dataset_name):
        return self.docs[dataset_name]
    def __normalize_scores__(self, ranked_list: list) -> list:
        """
        Normalizes scores in a ranked list to a [0, 1] scale.
        """
        scores = [score for doc_id, score, text in ranked_list]
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [(doc_id, 1.0) for doc_id, score, text in ranked_list]
        
        normalized_list = []
        for doc_id, score, text in ranked_list:
            normalized_score = (score - min_score) / (max_score - min_score)
            normalized_list.append((doc_id, normalized_score))
        return normalized_list

    def search(self, dataset_name: str, query: str, top_k: int = 10) -> list[tuple[str, float, str]]:
        """
        Performs a complex hybrid search.
        """
        print("\nPerforming complex hybrid search (Stage 1: Fusion)...")

        # --- Get the required service modules from the registry ---
        tfidf_service = TFIDF_online(self.db, self.docs)
        bm25_service = BM25_online(self.db, self.docs)
        embedding_service = Embedding_online(self.db, self.docs)

        # ==========================================================================
        #  STAGE 1: Parallel Fusion of TF-IDF and BM25
        # ==========================================================================
        
        tfidf_results = tfidf_service.search(dataset_name, query, top_k*2)

        bm25_results = bm25_service.search(dataset_name, query, top_k*2)

        # --- Normalize and Fuse the lexical results ---
        norm_tfidf = self.__normalize_scores__(tfidf_results)
        norm_bm25 = self.__normalize_scores__(bm25_results)

        fused_scores = {}
        tfidf_weight = 0.3
        bm25_weight = 0.7

        for doc_id, score in norm_tfidf:
            fused_scores[str(doc_id)] = score * tfidf_weight

        for doc_id, score in norm_bm25:
            doc_id_str = str(doc_id)
            if doc_id_str in fused_scores:
                fused_scores[doc_id_str] += score * bm25_weight
            else:
                fused_scores[doc_id_str] = score * bm25_weight
                
        # --- Create the final candidate list from Stage 1 ---
        candidate_list = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Extract just the document IDs for the next stage
        candidate_doc_ids = [doc_id for doc_id, score in candidate_list]
        
        # ==========================================================================
        #  STAGE 2: Serial Re-ranking with Embedding Model
        # ==========================================================================

        # Call the new, efficient rerank function from the embedding service
        reranked_results = embedding_service.embedding_rerank(dataset_name, query, candidate_doc_ids)
        
        # Convert reranked results to the expected format with text snippets
        docs_list = self.docs[dataset_name]
        doc_id_to_doc = {doc.doc_id: doc for doc in docs_list}
        
        final_list = []
        for doc_id, score, _ in reranked_results[:top_k]:
            doc_obj = doc_id_to_doc.get(doc_id)
            if doc_obj:
                doc_id = doc_obj.doc_id
                doc_text = doc_obj.text[:100] + "..."
            else:
                doc_text = ""
            final_list.append((doc_id, score, doc_text))
        return final_list