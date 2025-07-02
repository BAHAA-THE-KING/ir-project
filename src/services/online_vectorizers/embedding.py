import math
import chromadb
from sentence_transformers import SentenceTransformer, util
import numpy as np
import joblib
import torch 
from loader import load_dataset
from sentence_transformers import SentenceTransformer
from services.processing.preprocessing import preprocess_text

# def embedding_search(dataset_name: str, query: str, top_k: int = 10):
   
#     print("Loading pre-trained models and data...")
#     model_path = 'data/models/all-MiniLM-L6-v2' 
#     print(f"Loading Sentence-BERT model from local path: {model_path}...")
#     model = SentenceTransformer(model_path)
    
#     embedding_path = f"data/{dataset_name}/bert_embeddings.npy"
#     document_embeddings = np.load(embedding_path)


#     docs = load_dataset(dataset_name)

#     print("Encoding query...")
#     query_embedding = model.encode(query)

#     cos_scores = util.cos_sim(torch.tensor(query_embedding), torch.tensor(document_embeddings))[0]
#     top_results = torch.topk(cos_scores, k=top_k)

#     results = []
#     print(f"\nTop {top_k} results for query: '{query}'")
#     for score, idx in zip(top_results[0], top_results[1]):
#         doc_id = docs[idx].doc_id
#         doc_text = docs[idx].text[:100] + "..." 
#         results.append((doc_id, score.item(), doc_text))
#         print(f"Doc ID: {doc_id}, Score: {score.item():.4f}, Text: {doc_text}")
    
#     return results


class Embedding_online:
    __embeddingInstance__ : dict[str, any] = {}
    __collection_instance__: dict = {}
    __modelInstance__  = None

    @staticmethod
    def loadModelInstance():
        if Embedding_online.__modelInstance__ == None:
            Embedding_online.__modelInstance__ = SentenceTransformer("data/models/all-MiniLM-L6-v2") 
        return Embedding_online.__modelInstance__

    @staticmethod
    def loadInstance(dataset_name : str):
        if dataset_name not in Embedding_online.__embeddingInstance__.keys():
            with open(f"data/{dataset_name}/bert_embeddings.npy", "rb") as f:
                Embedding_online.__embeddingInstance__[dataset_name] = np.load(f)
        return Embedding_online.__embeddingInstance__[dataset_name]

    @staticmethod
    def _get_collection(dataset_name: str):
        if dataset_name not in Embedding_online.__collection_instance__:
            print(f"Connecting to ChromaDB and getting collection: {dataset_name}_embeddings...")
            client = chromadb.PersistentClient(path="chroma_db")
            Embedding_online.__collection_instance__[dataset_name] = client.get_collection(name=f"{dataset_name}_embeddings")
        return Embedding_online.__collection_instance__[dataset_name]

    @staticmethod
    def embedding_search(dataset_name: str, query: str, top_k: int = 10):
        # Load model and documents
        model = Embedding_online.loadModelInstance()
        document_embeddings =  Embedding_online.loadInstance(dataset_name)
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

    @staticmethod
    def embedding_vectors_search(dataset_name: str, query: str, top_k: int = 10):
        #Load model and collection
        model = Embedding_online.loadModelInstance()
        collection = Embedding_online._get_collection(dataset_name)
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

    @staticmethod
    def calc_dcg(rel, rank):
        return ((2 ** rel) - 1) / math.log10(rank + 1)

    @staticmethod
    def evaluate_embedding(dataset_name, queries, qrels, K = 10):
        nDCG = []

        for i in range(1000):
            query = queries[i]
            # print(f"Query: {query.text}")
            # print(f"Query: {bm25_preprocess_text(query.text)}")
            
            # Search using BM25
            # results = Embedding_online.embedding_search(dataset_name, query.text, K)
            results = Embedding_online.embedding_vectors_search(dataset_name, query.text, K)
            # for i, res in enumerate(results):
                # print(f"Result #{i} {res[1]}: {res[2]}")
                # print(f"Result #{i} {res[1]}: {bm25_preprocess_text(res[2])}")

            # Find relevant documents for this query
            relevant_qrels = [qrel for qrel in qrels if qrel.query_id == query.query_id]
            relevant_qrels = sorted(relevant_qrels, key=lambda x: x.relevance, reverse=True)

            DCG = [
                Embedding_online.calc_dcg(
                    list(
                        filter(
                            lambda qrel: qrel.doc_id == doc[0], relevant_qrels
                            )
                        )[0].relevance if list(
                        filter(
                            lambda qrel: qrel.doc_id == doc[0], relevant_qrels
                            )
                        ) else 0
                    , i+1
                ) for i, doc in enumerate(results)]
            
            iDCG = [Embedding_online.calc_dcg(qrel.relevance, i+1) for i, qrel in enumerate(relevant_qrels[:K])]
            
            res = sum(DCG) 
            ires = sum(iDCG) 
            
            print("")
            print(f"query: {i}")
            print(f"nDCG: {res}")
            print(f"iDCG: {ires}")
            print(f"nDCG: {res/ires*100}%")
            nDCG.append(res/ires)
        
        print(f"Average nDCG: {sum(nDCG)/len(nDCG)*100}%")
