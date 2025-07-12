# import math
# import chromadb
# from sentence_transformers import SentenceTransformer, util
# import numpy as np
# import torch 
# from loader import load_dataset 
# from services.processing.text_preprocessor import TextPreprocessor
# from services.online_vectorizers.Retriever import Retriever 

# class Embedding_online(Retriever):
#     __embeddingInstance__ : dict[str, any] = {}
#     __collection_instance__: dict = {}
#     __modelInstance__ = None

#     @staticmethod
#     def __loadModelInstance__():
#         if Embedding_online.__modelInstance__ == None:
         
#             Embedding_online.__modelInstance__ = SentenceTransformer("all-MiniLM-L6-v2") 
            
         
#             if torch.cuda.is_available():
#                 Embedding_online.__modelInstance__.to('cuda')
#                 print("Embedding model moved to GPU.")
#             else:
#                 print("GPU not available, embedding model running on CPU.")
                
#         return Embedding_online.__modelInstance__

#     @staticmethod
#     def __loadInstance__(dataset_name : str):
    
#         if dataset_name not in Embedding_online.__embeddingInstance__.keys():
#             with open(f"./data/{dataset_name}/bert_embeddings.npy", "rb") as f:
#                 Embedding_online.__embeddingInstance__[dataset_name] = np.load(f)
#         return Embedding_online.__embeddingInstance__[dataset_name]

#     @staticmethod
#     def __get_collection__(dataset_name: str):
     
#         if dataset_name not in Embedding_online.__collection_instance__:
#             print(f"Connecting to ChromaDB and getting collection: {dataset_name}_embeddings...")
         
#             client = chromadb.PersistentClient(path="./chroma_db") 
#             Embedding_online.__collection_instance__[dataset_name] = client.get_collection(name=f"{dataset_name}_embeddings")
#         return Embedding_online.__collection_instance__[dataset_name]

#     def search(self, dataset_name: str, query: str, top_k: int = 10, with_index: bool = True) -> list[tuple[str, float, str]]:
#         if with_index:
#             return self.embedding_vectors_search(dataset_name, query, top_k)
#         else:
#             return self.embedding_search(dataset_name, query, top_k)

#     def embedding_search(self, dataset_name: str, query: str, top_k: int):
#         model = Embedding_online.__loadModelInstance__() 
#         document_embeddings = Embedding_online.__loadInstance__(dataset_name)
#         docs = load_dataset(dataset_name)
        
#         processedQueryTokens = TextPreprocessor.getInstance().preprocess_text(query, remove_stopwords_flag=False)
#         processedQuery = " ".join(processedQueryTokens) 
        
#         query_embedding = model.encode(processedQuery, convert_to_tensor=True) 
        
#         cos_scores = util.cos_sim(query_embedding, torch.tensor(document_embeddings).to(query_embedding.device))[0]
#         top_results = torch.topk(cos_scores, k=top_k)
#         results = []
#         for score, idx in zip(top_results[0], top_results[1]):
#             doc_id = docs[idx].doc_id
#             doc_text = docs[idx].text[:100] + "..." 
#             results.append((doc_id, score.item(), doc_text))
#         return results

#     def embedding_vectors_search(self, dataset_name: str, query: str, top_k: int):
#         model = Embedding_online.__loadModelInstance__() 
#         collection = Embedding_online.__get_collection__(dataset_name)
        
#         processedQueryTokens = TextPreprocessor.getInstance().preprocess_text(query, remove_stopwords_flag=False)
#         processedQuery = " ".join(processedQueryTokens) 
        
#         query_embedding = model.encode(processedQuery, convert_to_tensor=False) 
#         search_results = collection.query(
#             query_embeddings=[query_embedding],
#             n_results=top_k
#         )
#         results = []
#         ids = search_results['ids'][0]
#         distances = search_results['distances'][0]
#         metadatas = search_results['metadatas'][0]
#         for doc_id, score, meta in zip(ids, distances, metadatas):
#             similarity_score = 1 - score
#             text = meta.get('text', '')[:100] + "..." 
#             results.append((doc_id, similarity_score, text))
#         return results

#     def embedding_rerank(self, dataset_name: str, query: str, doc_ids: list) -> list[tuple[str, float]]:
#         docs_list = load_dataset(dataset_name)
#         model = Embedding_online.__loadModelInstance__() 

#         doc_texts_to_rerank = []
#         doc_id_to_original_doc_obj = {doc.doc_id: doc for doc in docs_list} 
        
#         for doc_id in doc_ids:
#             if doc_id in doc_id_to_original_doc_obj:
#                 doc_texts_to_rerank.append(doc_id_to_original_doc_obj[doc_id].text)
            
#         if not doc_texts_to_rerank:
#             return []

#         candidate_embeddings = model.encode(doc_texts_to_rerank, convert_to_tensor=True) 
        
#         query_embedding = model.encode(query, convert_to_tensor=True)
        
#         cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
        
#         reranked_results = []
#         for i, score in enumerate(cosine_scores):
#             reranked_results.append((doc_ids[i], score.item()))

#         return sorted(reranked_results, key=lambda item: item[1], reverse=True)

# src/services/online_vectorizers/embedding.py

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
            client = chromadb.PersistentClient(path="./chroma_db")
            Embedding_online.__collection_instance__[dataset_name] = client.get_collection(name=f"{dataset_name}_embeddings")
        return Embedding_online.__collection_instance__[dataset_name]

    def search(self, dataset_name: str, query: str, top_k: int = 10, with_index: bool = True):
        if with_index:
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
        docs = load_dataset(dataset_name)
        print("DEBUG: docs loaded, number of docs:", len(docs))

        processedQueryTokens = TextPreprocessor.getInstance().preprocess_text(query, remove_stopwords_flag=False)
        processedQuery = " ".join(processedQueryTokens)

        # Ensure both embeddings are on the same device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        query_embedding = model.encode(processedQuery, convert_to_tensor=True, device=device)
        doc_embeddings_tensor = torch.tensor(document_embeddings, device=device)

        cos_scores = util.cos_sim(query_embedding, doc_embeddings_tensor)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            doc_id = docs[idx].doc_id
            doc_text = docs[idx].text[:100] + "..."
            results.append((doc_id, score.item(), doc_text))
        return results

    def embedding_vectors_search(self, dataset_name: str, query: str, top_k: int):
        print("DEBUG: embedding_vectors_search called for dataset:", dataset_name)
        model = Embedding_online.__loadModelInstance__()
        collection = Embedding_online.__get_collection__(dataset_name)
        

        processedQueryTokens = TextPreprocessor.getInstance().preprocess_text(query, remove_stopwords_flag=False)
        processedQuery = " ".join(processedQueryTokens) 
        
        query_embedding = model.encode(processedQuery, convert_to_tensor=False) # ChromaDB قد يفضل numpy
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
        docs_list = load_dataset(dataset_name)
        document_embeddings = Embedding_online.__loadInstance__(dataset_name)
        model = Embedding_online.__loadModelInstance__()

        # 1. Create a quick lookup map for doc_id to its index
        # تصحيح: هذا الجزء لا ينشئ خريطة doc_id إلى الفهرس بشكل صحيح
        # يجب أن تكون خريطة حقيقية {doc.doc_id: index for index, doc in enumerate(docs_list)}
        # ولكن لغرض الـ rerank، سنقوم بتحويل doc_ids إلى النصوص أولاً
        
        # تحويل قائمة doc_ids إلى قائمة النصوص المقابلة لها
        doc_texts_to_rerank = []
        doc_id_to_original_doc_obj = {doc.doc_id: doc for doc in docs_list} # قاموس للبحث السريع
        
        for doc_id in doc_ids:
            if doc_id in doc_id_to_original_doc_obj:
                doc_texts_to_rerank.append(doc_id_to_original_doc_obj[doc_id].text)
            
        if not doc_texts_to_rerank:
            return []

        # 2. Encode the candidate documents
        candidate_embeddings = model.encode(doc_texts_to_rerank, convert_to_tensor=True)
        
        # 3. Encode the query
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # 4. Calculate similarity scores
        cosine_scores = util.cos_sim(query_embedding, candidate_embeddings)[0]
        
        # 5. Pair the original doc_ids with their new scores
        reranked_results = []
        for i, score in enumerate(cosine_scores):
            reranked_results.append((doc_ids[i], score.item())) # استخدام doc_ids الأصلية بالترتيب

        return sorted(reranked_results, key=lambda item: item[1], reverse=True)