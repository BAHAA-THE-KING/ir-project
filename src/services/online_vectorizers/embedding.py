import math
import chromadb
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch 
from loader import load_dataset
from sentence_transformers import SentenceTransformer
from services.processing.preprocessing import preprocess_text
from services.online_vectorizers.Retriever import Retriever

class Embedding_online(Retriever):
    __embeddingInstance__ : dict[str, any] = {}
    __collection_instance__: dict = {}
    __modelInstance__  = None

    @staticmethod
    def __loadModelInstance__():
        if Embedding_online.__modelInstance__ == None:
            Embedding_online.__modelInstance__ = SentenceTransformer("data/models/all-MiniLM-L6-v2") 
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
            client = chromadb.PersistentClient(path="chroma_db")
            Embedding_online.__collection_instance__[dataset_name] = client.get_collection(name=f"{dataset_name}_embeddings")
        return Embedding_online.__collection_instance__[dataset_name]

    def search(self, dataset_name: str, query: str, top_k: int = 10, with_index: bool = True):
        if with_index:
            return self.embedding_vectors_search(dataset_name, query, top_k)
        else:
            return self.embedding_search(dataset_name, query, top_k)

    def embedding_search(self, dataset_name: str, query: str, top_k: int):
        # Load model and documents
        model = Embedding_online.__loadModelInstance__()
        document_embeddings =  Embedding_online.__loadInstance__(dataset_name)
        docs = load_dataset(dataset_name)
        processedQuery = preprocess_text(query)
        query_embedding = model.encode(processedQuery)

        cos_scores = util.cos_sim(torch.tensor(query_embedding), torch.tensor(document_embeddings))[0]
        top_results = torch.topk(cos_scores, k=top_k)
        results = []
        # print(f"\nTop {top_k} results for query: '{query}'")
        for score, idx in zip(top_results[0], top_results[1]):
            doc_id = docs[idx].doc_id
            doc_text = docs[idx].text[:100] + "..." 
            results.append((doc_id, score.item(), doc_text))
            # print(f"Doc ID: {doc_id}, Score: {score.item():.4f}, Text: {doc_text}"
        return results

    def embedding_vectors_search(dataset_name: str, query: str, top_k: int):
        #Load model and collection
        model = Embedding_online.__loadModelInstance__()
        collection = Embedding_online.__get_collection__(dataset_name)
        #process query
        processedQuery = preprocess_text(query)
        query_embedding = model.encode(processedQuery)
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
