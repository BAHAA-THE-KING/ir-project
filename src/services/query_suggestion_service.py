# import os
# import joblib
# from collections import Counter
# import numpy as np 
# import chromadb 
# from uuid import uuid4 
# import gc


# from loader import load_dataset, load_queries_and_qrels 
# from services.processing.text_preprocessor import TextPreprocessor 
# from services.online_vectorizers.embedding import Embedding_online 

# INDEX_SAVE_PATH_BASE = "./data" 

# N_GRAM_CHROMADB_PATH = "./chroma_db_suggestions" 

# class QuerySuggestionService:
#     def __init__(self, dataset_name, preprocessor: TextPreprocessor):
#         self.dataset_name = dataset_name
#         self.preprocessor = preprocessor
#         self.embedding_model = Embedding_online.__loadModelInstance__() 
        
#         self.all_docs_by_id = {}
#         self.all_queries_by_id = {}

#         self.chroma_client = chromadb.PersistentClient(path=N_GRAM_CHROMADB_PATH)
#         self.suggestions_collection_name = f"{dataset_name.replace('/', '_')}_ngrams"
        
#         self._load_or_build_index()

#     def _get_index_file_path(self):
#         return os.path.join(INDEX_SAVE_PATH_BASE, self.dataset_name.replace('/', '_'), "query_suggestion_chromadb_indexed.flag")

#     def _load_index(self):
#         flag_path = self._get_index_file_path()
#         if os.path.exists(flag_path):
#             try:
#                 self.suggestions_collection = self.chroma_client.get_collection(name=self.suggestions_collection_name)
#                 print(f"Query suggestion ChromaDB collection '{self.suggestions_collection_name}' loaded successfully.")
#                 return True
#             except Exception as e:
#                 print(f"Failed to load ChromaDB collection: {e}. Rebuilding index.")
#                 return False
#         return False

#     def _build_index(self):
#         print(f"Building query suggestion index with embeddings for {self.dataset_name} in ChromaDB...")
        
#         try:
#             self.suggestions_collection = self.chroma_client.get_or_create_collection(name=self.suggestions_collection_name)
#         except Exception as e:
#             print(f"Error creating/getting ChromaDB collection: {e}. Aborting index build.")
#             return

#         existing_ids = self.suggestions_collection.get(limit=1)['ids'] 
#         if existing_ids: 
#             print(f"ChromaDB collection '{self.suggestions_collection_name}' has existing data. Clearing...")
#             self.suggestions_collection.delete(where={}) 
#             print(f"Cleared existing data in ChromaDB collection '{self.suggestions_collection_name}'.")
#         else:
#             print(f"ChromaDB collection '{self.suggestions_collection_name}' is already empty. No data to clear.")


#         try:
#             docs_list = load_dataset(self.dataset_name) 
#             queries_list, _ = load_queries_and_qrels(self.dataset_name)
#         except KeyError:
#             print(f"Dataset '{self.dataset_name}' not found in DATASETS configuration.")
#             return

#         self.all_docs_by_id = {doc.doc_id: doc.text for doc in docs_list}
#         self.all_queries_by_id = {query.query_id: query.text for query in queries_list}

#         all_text_sources = []
        
#         if not docs_list:
#             print(f"Warning: No documents loaded from ir_datasets for {self.dataset_name}. Suggestions will be limited for document-based content.")
#         for doc in docs_list:
#             if isinstance(doc.text, str):
#                 all_text_sources.append((doc.text, doc.doc_id, 'doc'))

#         if not queries_list:
#              print(f"Warning: No queries loaded from ir_datasets for {self.dataset_name}. Suggestions will be limited for query-based content.")
#         for query_obj in queries_list:
#             if isinstance(query_obj.text, str):
#                 all_text_sources.append((query_obj.text, query_obj.query_id, 'query'))
        
#         if not all_text_sources:
#             print(f"No text data available to build query suggestion index for {self.dataset_name}.")
#             return

#         N_GRAM_PROCESSING_BATCH_SIZE = 5000 
#         MAX_SOURCES_PER_NGRAM = 10 
#         CHROMA_ADD_SUB_BATCH_SIZE = 200 

#         total_ngrams_added = 0

#         for i in range(0, len(all_text_sources), N_GRAM_PROCESSING_BATCH_SIZE):
#             batch_texts_with_source_info = all_text_sources[i:i + N_GRAM_PROCESSING_BATCH_SIZE]
            
#             current_batch_ngrams_meta = {} 

#             for text, source_id, source_type in batch_texts_with_source_info:
#                 processed_tokens = self.preprocessor.preprocess_text(text, remove_stopwords_flag=False) 
                
#                 for token in processed_tokens:
#                     gram_text = token
#                     meta = current_batch_ngrams_meta.setdefault(gram_text, {'sources': set(), 'count': 0})
#                     if len(meta['sources']) < MAX_SOURCES_PER_NGRAM:
#                         meta['sources'].add((source_id, source_type))
#                     meta['count'] += 1 

#                 if len(processed_tokens) > 1:
#                     for j in range(len(processed_tokens) - 1):
#                         gram_text = " ".join((processed_tokens[j], processed_tokens[j+1]))
#                         meta = current_batch_ngrams_meta.setdefault(gram_text, {'sources': set(), 'count': 0})
#                         if len(meta['sources']) < MAX_SOURCES_PER_NGRAM:
#                             meta['sources'].add((source_id, source_type))
#                         meta['count'] += 1

#                 if len(processed_tokens) > 2:
#                     for j in range(len(processed_tokens) - 2):
#                         gram_text = " ".join((processed_tokens[j], processed_tokens[j+1], processed_tokens[j+2]))
#                         meta = current_batch_ngrams_meta.setdefault(gram_text, {'sources': set(), 'count': 0})
#                         if len(meta['sources']) < MAX_SOURCES_PER_NGRAM:
#                             meta['sources'].add((source_id, source_type))
#                         meta['count'] += 1
            
#             batch_ngrams_texts = []
#             batch_ngrams_ids = []
#             batch_ngrams_metadatas = []

#             for gram_text, meta in current_batch_ngrams_meta.items():
#                 batch_ngrams_texts.append(gram_text)
#                 batch_ngrams_ids.append(str(uuid4()))
#                 source_refs_str = ",".join([f"{sid}:{stype}" for sid, stype in meta['sources']])
#                 batch_ngrams_metadatas.append({'text': gram_text, 'sources': source_refs_str, 'count': meta['count']}) 

#             if not batch_ngrams_texts:
#                 print(f"Skipping empty N-gram batch {i//N_GRAM_PROCESSING_BATCH_SIZE + 1}.")
#                 continue

#             print(f"Generating embeddings for {len(batch_ngrams_texts)} N-grams in batch {i//N_GRAM_PROCESSING_BATCH_SIZE + 1}...")
#             n_gram_embeddings = self.embedding_model.encode(batch_ngrams_texts, convert_to_tensor=False, device='cuda') 

#             for j in range(0, len(batch_ngrams_ids), CHROMA_ADD_SUB_BATCH_SIZE):
#                 sub_batch_ids = batch_ngrams_ids[j:j+CHROMA_ADD_SUB_BATCH_SIZE]
#                 sub_batch_embeddings = n_gram_embeddings[j:j+CHROMA_ADD_SUB_BATCH_SIZE]
#                 sub_batch_metadatas = batch_ngrams_metadatas[j:j+CHROMA_ADD_SUB_BATCH_SIZE]
                
#                 self.suggestions_collection.add(
#                     embeddings=sub_batch_embeddings.tolist(),
#                     metadatas=sub_batch_metadatas,
#                     ids=sub_batch_ids
#                 )
#                 total_ngrams_added += len(sub_batch_ids)
#             print(f"Processed source batch {i//N_GRAM_PROCESSING_BATCH_SIZE + 1}. Total N-grams added so far: {total_ngrams_added}")
            
#             del batch_texts_with_source_info
#             del current_batch_ngrams_meta
#             del batch_ngrams_texts
#             del batch_ngrams_ids
#             del batch_ngrams_metadatas
#             del n_gram_embeddings 
#             gc.collect()

#         print(f"Finished processing all text sources. Total N-grams added to ChromaDB: {total_ngrams_added}")
        
#         os.makedirs(os.path.dirname(self._get_index_file_path()), exist_ok=True)
#         with open(self._get_index_file_path(), 'w') as f:
#             f.write("indexed")
#         print("Query suggestion index built and saved successfully in ChromaDB.")


#     def _load_or_build_index(self):
#         if not self._load_index():
#             self._build_index()

#     def _extract_snippet(self, full_text: str, phrase: str, window_size: int = 50) -> str:
#         if not full_text or not phrase:
#             return ""
        
#         full_text_lower = full_text.lower()
#         phrase_lower = phrase.lower()

#         try:
#             start_index = full_text_lower.index(phrase_lower)
#             end_index = start_index + len(phrase_lower)

#             snippet_start = max(0, start_index - window_size)
#             snippet_end = min(len(full_text), end_index + window_size)

#             snippet = full_text[snippet_start:snippet_end]
            
#             if snippet_start > 0:
#                 snippet = "..." + snippet
#             if snippet_end < len(full_text):
#                 snippet = snippet + "..."
            
#             return snippet.replace('\n', ' ').strip() 
#         except ValueError:
#             return full_text[:window_size * 2].replace('\n', ' ').strip() + "..." if len(full_text) > window_size * 2 else full_text.replace('\n', ' ').strip()


#     def get_suggestions(self, query_prefix: str, top_k: int = 10) -> list[tuple[str, str]]:
#         if not hasattr(self, 'suggestions_collection') or self.suggestions_collection is None:
#             print("ChromaDB suggestions collection not initialized. Please rebuild index.")
#             return []

#         processed_prefix_tokens = self.preprocessor.preprocess_text(query_prefix, remove_stopwords_flag=False)
#         processed_prefix = " ".join(processed_prefix_tokens)

#         if not processed_prefix:
#             return []

#         query_embedding = self.embedding_model.encode(processed_prefix, convert_to_tensor=False, device='cuda') 

#         search_results = self.suggestions_collection.query(
#             query_embeddings=[query_embedding.tolist()], 
#             n_results=top_k * 5, 
#             include=['metadatas', 'distances']
#         )
        
#         candidate_suggestions_with_details = []

#         if search_results and search_results['ids'] and search_results['metadatas']:
#             for i in range(len(search_results['ids'][0])):
#                 gram_text = search_results['metadatas'][0][i]['text']
#                 source_refs_str = search_results['metadatas'][0][i]['sources'] 
#                 distance = search_results['distances'][0][i]
#                 similarity = 1 - (distance / (self.embedding_model.get_sentence_embedding_dimension()**0.5)) if self.embedding_model.get_sentence_embedding_dimension() else 1 - distance 

#                 source_references = set()
#                 for ref_str in source_refs_str.split(','):
#                     if ref_str:
#                         try:
#                             sid, stype = ref_str.split(':', 1)
#                             source_references.add((sid, stype))
#                         except ValueError:
#                             pass 

#                 candidate_suggestions_with_details.append((gram_text, similarity, source_references))

#         candidate_suggestions_with_details.sort(key=lambda x: x[1], reverse=True)
        
#         final_suggestions_with_snippets = []
#         seen_suggestions = set()
        
#         for suggestion_text, _, source_references in candidate_suggestions_with_details:
#             if suggestion_text not in seen_suggestions:
#                 snippet = ""
#                 if source_references:
#                     source_id, source_type = next(iter(source_references)) 
                    
#                     original_text = ""
#                     if source_type == 'doc' and source_id in self.all_docs_by_id:
#                         original_text = self.all_docs_by_id[source_id]
#                     elif source_type == 'query' and source_id in self.all_queries_by_id:
#                         original_text = self.all_queries_by_id[source_id]

#                     if original_text:
#                         snippet = self._extract_snippet(original_text, suggestion_text)
                
#                 final_suggestions_with_snippets.append((suggestion_text, snippet))
#                 seen_suggestions.add(suggestion_text)
#             if len(final_suggestions_with_snippets) >= top_k:
#                 break
        
#         return final_suggestions_with_snippets

# src/services/query_suggestion_service.py
import os
import joblib
from collections import Counter
from itertools import chain
import numpy as np # لإجراء عمليات على المتجهات

from loader import load_dataset, load_queries_and_qrels  # , load_dataset_with_queries
from services.processing.text_preprocessor import TextPreprocessor 
from services.online_vectorizers.embedding import Embedding_online  # استيراد كلاس Embedding_online

INDEX_SAVE_PATH_BASE = "C:\\Users\\FSOS\\Documents\\Projects\\ir-project\\data\\antique" 

class QuerySuggestionService:
    def __init__(self, dataset_name, preprocessor: TextPreprocessor):
        self.dataset_name = dataset_name
        self.preprocessor = preprocessor
        self.suggestion_index = None
        
        # تحميل موديل الـ Embedding مرة واحدة عند تهيئة الخدمة
        self.embedding_model = Embedding_online.__loadModelInstance__() 
        
        self._load_or_build_index()

    def _get_index_file_path(self):
        return os.path.join(INDEX_SAVE_PATH_BASE, self.dataset_name.replace('/', '_'), "query_suggestion_index_with_embeddings.joblib")

    def _load_index(self):
        index_path = self._get_index_file_path()
        if os.path.exists(index_path):
            print(f"Loading query suggestion index with embeddings for {self.dataset_name}...")
            self.suggestion_index = joblib.load(index_path)
            print("Query suggestion index loaded successfully.")
            return True
        return False

    def _build_index(self):
        print(f"Building query suggestion index with embeddings for {self.dataset_name}...")
        
        try:
            docs = load_dataset(self.dataset_name) 
            queries, _ = load_queries_and_qrels(self.dataset_name)
        except KeyError:
            print(f"Dataset '{self.dataset_name}' not found in DATASETS configuration.")
            self.suggestion_index = {}
            return

        texts_to_process = []
        if docs:
            texts_to_process.extend([doc.text for doc in docs])
        else:
            print(f"Warning: No documents found for dataset {self.dataset_name}. This will result in very limited suggestions.")

        if queries:
            texts_to_process.extend([query.text for query in queries])
        
        if not texts_to_process:
            print(f"No text data available to build query suggestion index for {self.dataset_name}.")
            self.suggestion_index = {}
            return

        processed_ngrams_counter = Counter() # استخدم اسم مختلف لتجنب الالتباس

        for text in texts_to_process:
            if isinstance(text, str):
                # لا تقم بإزالة الكلمات المتوقفة هنا
                processed_tokens = self.preprocessor.preprocess_text(text, remove_stopwords_flag=False) 
                
                # توليد الـ N-grams من الـ processed_tokens
                processed_ngrams_counter.update(processed_tokens)
                if len(processed_tokens) > 1:
                    bigrams = [(processed_tokens[i], processed_tokens[i+1]) for i in range(len(processed_tokens) - 1)]
                    processed_ngrams_counter.update(bigrams)
                if len(processed_tokens) > 2:
                    trigrams = [(processed_tokens[i], processed_tokens[i+1], processed_tokens[i+2]) for i in range(len(processed_tokens) - 2)]
                    processed_ngrams_counter.update(trigrams)
            else:
                print(f"Warning: Skipping non-string item in texts_to_process: {text}")

        top_n_grams_for_embedding = 500000 # يمكن تعديل هذا الحد
        n_grams_texts_to_embed = []
        n_grams_tuple_map = {} # لربط النص بالـ tuple الأصلي

        for gram, count in processed_ngrams_counter.most_common(top_n_grams_for_embedding):
            gram_text = " ".join(gram) if isinstance(gram, tuple) else gram
            n_grams_texts_to_embed.append(gram_text)
            n_grams_tuple_map[gram_text] = gram 

        print(f"Generating embeddings for {len(n_grams_texts_to_embed)} N-grams...")
        # استخدام الموديل المحمل من Embedding_online مباشرة
        n_gram_embeddings = self.embedding_model.encode(n_grams_texts_to_embed, convert_to_tensor=False) 

        suggestion_index_with_embeddings = {}

        for i, gram_text in enumerate(n_grams_texts_to_embed):
            gram_embedding = n_gram_embeddings[i]
            original_gram_tuple = n_grams_tuple_map[gram_text]
            
            if isinstance(original_gram_tuple, tuple):
                prefix = " ".join(original_gram_tuple[:-1])
            else:
                prefix = original_gram_tuple 

            if prefix not in suggestion_index_with_embeddings:
                suggestion_index_with_embeddings[prefix] = []
            
            suggestion_index_with_embeddings[prefix].append((gram_text, gram_embedding))

        self.suggestion_index = suggestion_index_with_embeddings

        os.makedirs(os.path.dirname(self._get_index_file_path()), exist_ok=True)
        joblib.dump(self.suggestion_index, self._get_index_file_path())
        print("Query suggestion index with embeddings built and saved successfully.")

    def _load_or_build_index(self):
        if not self._load_index():
            self._build_index()

    def get_suggestions(self, query_prefix: str, top_k: int = 10) -> list[str]:
        if not self.suggestion_index:
            return []

        processed_prefix_tokens = self.preprocessor.preprocess_text(query_prefix, remove_stopwords_flag=False)
        processed_prefix = " ".join(processed_prefix_tokens)

        if not processed_prefix:
            return []

        # توليد الـ Embedding لبادئة الكويري باستخدام الموديل المحمل
        query_embedding = self.embedding_model.encode(processed_prefix, convert_to_tensor=False) 

        candidate_suggestions = []

        # البحث عن طريق تطابق البادئة
        for key_prefix, full_grams_data in self.suggestion_index.items():
            if key_prefix.startswith(processed_prefix):
                for gram_text, gram_embedding in full_grams_data:
                    if gram_text.startswith(processed_prefix):
                        similarity = np.dot(query_embedding, gram_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(gram_embedding))
                        candidate_suggestions.append((gram_text, similarity))

        # هذا الجزء مكلف للغاية للبحث الدلالي الكامل إذا لم يتم استخدام فهرس متجهات متخصص
        # قم بتنشيطه فقط إذا كنت مستعداً لتحديات الأداء
        # إذا لم نجد اقتراحات بتطابق البادئة، يمكن البحث دلالياً بشكل أوسع (ولكن ببطء)
        if not candidate_suggestions: # أو إذا كانت النتائج قليلة جداً
             all_gram_texts = []
             all_gram_embeddings = []
             for key_prefix, full_grams_data in self.suggestion_index.items():
                 for gram_text, gram_embedding in full_grams_data:
                     all_gram_texts.append(gram_text)
                     all_gram_embeddings.append(gram_embedding)

             if all_gram_embeddings: # لتجنب القسمة على صفر إذا كانت فارغة
                all_gram_embeddings = np.array(all_gram_embeddings)
                similarities = np.dot(all_gram_embeddings, query_embedding) / (np.linalg.norm(all_gram_embeddings, axis=1) * np.linalg.norm(query_embedding))
                
                for i, sim in enumerate(similarities):
                    candidate_suggestions.append((all_gram_texts[i], sim))


        candidate_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        final_suggestions = []
        seen_suggestions = set()
        for suggestion_text, _ in candidate_suggestions:
            if suggestion_text not in seen_suggestions:
                final_suggestions.append(suggestion_text)
                seen_suggestions.add(suggestion_text)
            if len(final_suggestions) >= top_k:
                break
        
        return final_suggestions[:top_k]