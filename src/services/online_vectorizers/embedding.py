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
            try:
                client = chromadb.PersistentClient(path=f"data/{dataset_name}/chroma_db")
                
                # Check if collection exists
                try:
                    collection = client.get_collection(name=f"{dataset_name}_embeddings")
                    print(f"DEBUG: Successfully got collection: {type(collection)}")
                    Embedding_online.__collection_instance__[dataset_name] = collection
                except Exception as collection_error:
                    print(f"ERROR: Collection '{dataset_name}_embeddings' not found: {collection_error}")
                    print(f"Available collections: {client.list_collections()}")
                    raise ValueError(f"ChromaDB collection '{dataset_name}_embeddings' not found. Please build the vector index first.")
                    
            except Exception as e:
                print(f"ERROR getting collection: {e}")
                raise e
        else:
            print(f"DEBUG: Using cached collection for {dataset_name}")
        return Embedding_online.__collection_instance__[dataset_name]

    @staticmethod
    def __loadDocs__(dataset_name: str):
        if dataset_name not in Embedding_online.__docs__:
            Embedding_online.__docs__[dataset_name] = load_dataset(dataset_name)
        return Embedding_online.__docs__[dataset_name]

    def search(self, dataset_name: str, query: str, top_k: int = 10, with_index: bool = False):
        if with_index:
            try:
                return self.embedding_vectors_search(dataset_name, query, top_k)
            except Exception as e:
                print(f"WARNING: Vector store search failed ({e}), falling back to regular embedding search")
                return self.embedding_search(dataset_name, query, top_k)
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
        processedQuery = TextPreprocessor.getInstance().preprocess_text(query)
        query_embedding = model.encode(" ".join(processedQuery))

        # Convert document_embeddings to tensor for cosine similarity calculation
        doc_embeddings_tensor = torch.tensor(document_embeddings).to(query_embedding.device)
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
        
        try:
            collection = Embedding_online.__get_collection__(dataset_name)
            print(f"DEBUG: Collection type: {type(collection)}")
            print(f"DEBUG: Collection: {collection}")
            
            if not hasattr(collection, 'query'):
                raise ValueError(f"Collection object does not have 'query' method. Type: {type(collection)}")
            
            #process query
            processedQuery = TextPreprocessor.getInstance().preprocess_text(query)
            query_embedding = model.encode(" ".join(processedQuery))
            
            print(f"DEBUG: About to call collection.query with embedding shape: {query_embedding.shape}")
            search_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            print(f"DEBUG: Search results keys: {search_results.keys()}")
            
            results = []
            ids = search_results['ids'][0]
            distances = search_results['distances'][0]
            metadatas = search_results['metadatas'][0]
            for doc_id, score, meta in zip(ids, distances, metadatas):
                similarity_score = 1 - score
                text = meta.get('text', '')[:100] + "..." 
                results.append((doc_id, similarity_score, text))
            return results
            
        except Exception as e:
            print(f"ERROR in embedding_vectors_search: {e}")
            print(f"ERROR type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise e

    def embedding_rerank(self, dataset_name: str, query: str, doc_ids: list) -> list[tuple[str, float]]:
        docs_list = load_dataset(dataset_name)
        document_embeddings = Embedding_online.__loadInstance__(dataset_name)
        model = Embedding_online.__loadModelInstance__()

        # 1. Create a quick lookup map for doc_id to its index
        # Note: This part doesn't create a proper doc_id to index mapping
        # It should be a real map {doc.doc_id: index for index, doc in enumerate(docs_list)}
        # But for reranking purposes, we'll convert doc_ids to texts first
        
        # Convert list of doc_ids to list of corresponding texts
        doc_texts_to_rerank = []
        doc_id_to_original_doc_obj = {doc.doc_id: doc for doc in docs_list} # Dictionary for fast lookup
        
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
            reranked_results.append((doc_ids[i], score.item())) 
        return sorted(reranked_results, key=lambda item: item[1], reverse=True)