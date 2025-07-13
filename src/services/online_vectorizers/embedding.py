import math
import chromadb
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch 
from src.loader import load_dataset
from src.services.processing.text_preprocessor import TextPreprocessor
from src.services.online_vectorizers.Retriever import Retriever
import __main__
setattr(__main__, 'TextPreprocessor', TextPreprocessor)

class Embedding_online(Retriever):
    __embeddingInstance__ : dict[str, object] = {}
    __collection_instance__: dict = {}
    __modelInstance__ = None
    __modelInstance__ = None
    __docs__: dict = {}

    @staticmethod
    def __loadModelInstance__():
        if Embedding_online.__modelInstance__ == None:
          
            Embedding_online.__modelInstance__ = SentenceTransformer("all-MiniLM-L6-v2") 
        return Embedding_online.__modelInstance__

    @staticmethod
    def __loadInstance__(dataset_name : str):
        if dataset_name not in Embedding_online.__embeddingInstance__.keys():
            with open(f"data/{dataset_name}/bert_embeddings.npy", "rb") as f:
                Embedding_online.__embeddingInstance__[dataset_name] = np.load(f)
        return Embedding_online.__embeddingInstance__[dataset_name]

    @staticmethod
    def __get_collection__(dataset_name: str):
        if dataset_name not in Embedding_online.__collection_instance__:
            print(f"Connecting to ChromaDB and getting collection: {dataset_name}_embeddings...")
            client = chromadb.PersistentClient(path=f"data/{dataset_name}/chroma_db")
            Embedding_online.__collection_instance__[dataset_name] = client.get_collection(name=f"{dataset_name}_embeddings")
        return Embedding_online.__collection_instance__[dataset_name]

    @staticmethod
    def __loadDocs__(dataset_name: str):
        if dataset_name not in Embedding_online.__docs__:
            Embedding_online.__docs__[dataset_name] = load_dataset(dataset_name)
        return Embedding_online.__docs__[dataset_name]

    def search(self, dataset_name: str, query: str, top_k: int = 10):
        if Embedding_online.with_index:
            return self.embedding_vectors_search(dataset_name, query, top_k)
        else:
            return self.embedding_search(dataset_name, query, top_k)

    def embedding_search(self, dataset_name: str, query: str, top_k: int):
        print("DEBUG: embedding_search called for dataset:", dataset_name)
        print("DEBUG: About to load bert_embeddings.npy")
        model = Embedding_online.__loadModelInstance__()
        document_embeddings = Embedding_online.__loadInstance__(dataset_name)
        print("DEBUG: bert_embeddings.npy loaded")
        print("DEBUG: About to load docs with load_dataset")
        docs = Embedding_online.__loadDocs__(dataset_name)
        processedQuery = TextPreprocessor.getInstance().preprocess_text(query)
        query_embedding = model.encode(" ".join(processedQuery), convert_to_tensor=True)

         # Convert document_embeddings to tensor for cosine similarity calculation
        doc_embeddings_tensor = torch.tensor(document_embeddings).to(query_embedding.device)
        cos_scores = util.cos_sim(query_embedding, doc_embeddings_tensor)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            doc_id = docs[idx].doc_id
            doc_text = docs[idx].text[:100] + "..."
            results.append((doc_id, score.item(), doc_text))
        return results

    def embedding_vectors_search(self, dataset_name: str, query: str, top_k: int):
        print("DEBUG: embedding_vectors_search called for dataset:", dataset_name)
        model = Embedding_online.__loadModelInstance__()
        collection = Embedding_online.__get_collection__(dataset_name)
        #process query
        processedQuery = TextPreprocessor.getInstance().preprocess_text(query)
        query_embedding = model.encode(" ".join(processedQuery))
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        results = []
        ids = search_results['ids'][0]
        distances = search_results['distances'][0]
        metadatas = search_results['metadatas'][0]
        for doc_id, score, meta in zip(ids, distances, metadatas):
            similarity_score = 1 - score
            text = meta.get('text', '')[:100] + "..." 
            results.append((doc_id, similarity_score, text))
        return results

    def embedding_rerank(self, dataset_name: str, query: str, doc_ids: list) -> list[tuple[str, float]]:
        """
        Efficiently re-ranks a list of documents using the loaded embeddings.
        """

        docs_list =  Embedding_online.__loadDocs__(dataset_name)
        document_embeddings =  Embedding_online.__loadInstance__(dataset_name)
        model = Embedding_online.__loadModelInstance__()

        # 1. Create a quick lookup map for doc_id to its index
        doc_id_to_index = enumerate(docs_list)

        # 2. Get the indices and embeddings for the documents we need to rerank
        candidate_indices = [i for i, doc in doc_id_to_index if doc.doc_id in doc_ids]

        # Filter out any docs that might not be in our list
        valid_indices = [idx for idx in candidate_indices if idx is not None]
        
        if not valid_indices:
            return []
            
        candidate_embeddings = document_embeddings[valid_indices]
        
        # 3. Encode the query
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # 4. Calculate similarity scores
        cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
        
        # 5. Pair the original doc_ids (that were valid) with their new scores
        valid_doc_ids = [docs_list[i] for i in valid_indices]
        reranked_results = []
        for doc, score in zip(valid_doc_ids, cosine_scores):
            reranked_results.append((doc.doc_id, score.item(), doc.text))

        return sorted(reranked_results, key=lambda item: item[1], reverse=True)