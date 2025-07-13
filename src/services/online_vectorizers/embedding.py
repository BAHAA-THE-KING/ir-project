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

class DocObj:
    def __init__(self, doc_id, text):
        self.doc_id = doc_id
        self.text = text

class Embedding_online(Retriever):
    __embeddingInstance__ : dict[str, object] = {}
    __collection_instance__: dict = {}
    __modelInstance__ = None
    __modelInstance__ = None
    __docs__: dict = {}

    def __init__(self, db_connector, docs):
        self.db = db_connector
        self.docs = docs

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

    def __loadDocs__(self, dataset_name: str):
        if dataset_name not in Embedding_online.__docs__:
            Embedding_online.__docs__[dataset_name] = load_dataset(dataset_name)
        return Embedding_online.__docs__[dataset_name]

    def search(self, dataset_name: str, query: str, top_k: int = 10):
        if Embedding_online.with_index:
            return self.embedding_vectors_search(dataset_name, query, top_k)
        else:
            return self.embedding_search(dataset_name, query, top_k)

    def embedding_search(self, dataset_name: str, query: str, top_k: int):
        model = Embedding_online.__loadModelInstance__()
        document_embeddings = Embedding_online.__loadInstance__(dataset_name)
        processedQuery = TextPreprocessor.getInstance().preprocess_text(query)
        query_embedding = model.encode(" ".join(processedQuery), convert_to_tensor=True)
        # Ensure document embeddings are on the same device as query_embedding
        doc_embeddings_tensor = torch.tensor(document_embeddings).to(query_embedding.device)
        cos_scores = util.cos_sim(query_embedding, doc_embeddings_tensor)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            idx_int = int(idx)
            doc_id = self.db.get_doc_id_by_id(dataset_name, idx_int + 1, cleaned=False)
            doc_text = self.docs[dataset_name][idx_int].text
            if doc_text:
                doc_text = doc_text[:100] + "..."
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

    def embedding_rerank(self, dataset_name: str, query: str, doc_ids: list) -> list[tuple[str, float, str]]:
        docs_list = self.__loadDocs__(dataset_name)
        # Ensure docs_list is a list of DocObj
        fixed_docs_list = []
        for item in docs_list:
            if hasattr(item, "doc_id") and hasattr(item, "text"):
                fixed_docs_list.append(item)
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                fixed_docs_list.append(DocObj(item[0], item[1]))
        docs_list = fixed_docs_list

        document_embeddings = Embedding_online.__loadInstance__(dataset_name)
        model = Embedding_online.__loadModelInstance__()
        doc_id_to_index = {doc.doc_id: i for i, doc in enumerate(docs_list)}
        candidate_indices = [doc_id_to_index.get(doc_id) for doc_id in doc_ids]
        valid_indices = [idx for idx in candidate_indices if idx is not None and 0 <= idx < len(docs_list)]
        if not valid_indices:
            return []
        candidate_embeddings = np.array(document_embeddings)[valid_indices]
        query_embedding = model.encode(query, convert_to_tensor=True)
        candidate_embeddings_tensor = torch.tensor(candidate_embeddings).to(query_embedding.device)
        cosine_scores = util.cos_sim(query_embedding, candidate_embeddings_tensor)[0]
        valid_doc_objs = []
        for i in valid_indices:
            try:
                valid_doc_objs.append(docs_list[i])
            except Exception:
                continue
        reranked_results = []
        for doc, score in zip(valid_doc_objs, cosine_scores):
            reranked_results.append((doc.doc_id, score.item(), doc.text))
        return sorted(reranked_results, key=lambda item: item[1], reverse=True)