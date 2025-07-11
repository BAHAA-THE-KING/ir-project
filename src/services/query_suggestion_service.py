import os
import joblib
from collections import Counter
import numpy as np 
import chromadb 
from uuid import uuid4 
import gc 
import time
from concurrent.futures import ThreadPoolExecutor

from nltk.tokenize import word_tokenize 
import re 
from spellchecker import SpellChecker 

from src.loader import load_dataset, load_queries_and_qrels
from src.services.processing.text_preprocessor import TextPreprocessor
from src.services.online_vectorizers.embedding import Embedding_online
from src.database.db_connector import DBConnector



INDEX_SAVE_PATH_BASE = "./data" 
N_GRAM_CHROMADB_PATH = "./data/chroma_db_suggestions" 

class QuerySuggestionService:
    def __init__(self, dataset_name, preprocessor: TextPreprocessor, db_connector: DBConnector):
        self.dataset_name = dataset_name
        self.preprocessor = preprocessor
        self.embedding_model = Embedding_online.__loadModelInstance__() 
        self.db_connector = db_connector 

        self.all_docs_by_id = {}
        self.all_queries_by_id = {} 

        self.spell = SpellChecker() 

        self.chroma_client = chromadb.PersistentClient(path=N_GRAM_CHROMADB_PATH)
        self.suggestions_collection_name = f"{dataset_name.replace('/', '_')}_ngrams"
        
        self._load_or_build_index()

    def _get_index_file_path(self):
        return os.path.join(INDEX_SAVE_PATH_BASE, self.dataset_name.replace('/', '_'), "query_suggestion_chromadb_indexed.flag")

    def _load_index(self):
        flag_path = self._get_index_file_path()
        if os.path.exists(flag_path):
            try:
                self.suggestions_collection = self.chroma_client.get_collection(name=self.suggestions_collection_name)
                print(f"Query suggestion ChromaDB collection '{self.suggestions_collection_name}' loaded successfully.")
                
                self.all_docs_by_id = {doc_id: text for doc_id, text in self.db_connector.get_all_document_ids_and_texts(self.dataset_name, cleaned=False, sample_ratio=1.0, limit=None)}
                queries_list, _ = load_queries_and_qrels(self.dataset_name)
                self.all_queries_by_id = {query.query_id: query.text for query in queries_list}
                print(f"Loaded {len(self.all_docs_by_id)} raw docs and {len(self.all_queries_by_id)} raw queries for snippets.")

                return True
            except Exception as e:
                print(f"Failed to load ChromaDB collection or original texts: {e}. Rebuilding index.")
                return False
        return False

    def _build_index(self):
        print(f"Building query suggestion index with embeddings for {self.dataset_name} in ChromaDB...")
        
        try:
            self.suggestions_collection = self.chroma_client.get_or_create_collection(name=self.suggestions_collection_name)
        except Exception as e:
            print(f"Error creating/getting ChromaDB collection: {e}. Aborting index build.")
            return

        existing_ids = self.suggestions_collection.get(limit=1)['ids'] 
        if existing_ids: 
            print(f"ChromaDB collection '{self.suggestions_collection_name}' has existing data. Clearing...")
            self.suggestions_collection.delete(where={}) 
            print(f"Cleared existing data in ChromaDB collection '{self.suggestions_collection_name}'.")
        else:
            print(f"ChromaDB collection '{self.suggestions_collection_name}' is already empty. No data to clear.")

        docs_list_raw = self.db_connector.get_all_document_ids_and_texts(self.dataset_name, cleaned=False, sample_ratio=1.0, limit=None)
        queries_list_from_ir_datasets, _ = load_queries_and_qrels(self.dataset_name) 

        self.all_docs_by_id = {doc_id: text for doc_id, text in docs_list_raw}
        self.all_queries_by_id = {query.query_id: query.text for query in queries_list_from_ir_datasets}
        print(f"Loaded {len(self.all_docs_by_id)} raw docs and {len(self.all_queries_by_id)} raw queries for snippets.")

        SAMPLE_DOCS_FOR_NGRAMS = 1.0
        print(f"Loading cleaned documents (sample_ratio={SAMPLE_DOCS_FOR_NGRAMS}) from database for N-grams...")
        docs_list_for_ngrams = self.db_connector.get_all_document_ids_and_texts(
            self.dataset_name, 
            cleaned=True,
            sample_ratio=SAMPLE_DOCS_FOR_NGRAMS, 
            limit=None 
        )
        
        all_text_sources_for_ngrams = []
        
        if not docs_list_for_ngrams:
            print(f"Warning: No cleaned documents found in DB for {self.dataset_name}. N-grams will be limited.")
        for doc_id, text in docs_list_for_ngrams:
            if isinstance(text, str):
                all_text_sources_for_ngrams.append((text, doc_id, 'doc'))
        
        if not all_text_sources_for_ngrams:
            print(f"No text data available to build query suggestion index for {self.dataset_name}.")
            return

        N_GRAM_PROCESSING_BATCH_SIZE = 10000 
        MAX_SOURCES_PER_NGRAM = 10 
        CHROMA_ADD_SUB_BATCH_SIZE = 1000 

        total_ngrams_added = 0

        for i in range(0, len(all_text_sources_for_ngrams), N_GRAM_PROCESSING_BATCH_SIZE):
            batch_start_time = time.time()
            batch_texts_with_source_info = all_text_sources_for_ngrams[i:i + N_GRAM_PROCESSING_BATCH_SIZE]
            
            current_batch_ngrams_meta = {} 

            def extract_ngrams_batch_func(args): 
                text, source_id, source_type = args
                processed_tokens = word_tokenize(text.strip()) 
                ngrams_list = []
                for token in processed_tokens:
                    ngrams_list.append((token, source_id, source_type))
                if len(processed_tokens) > 1:
                    for j in range(len(processed_tokens) - 1):
                        ngrams_list.append((f"{processed_tokens[j]} {processed_tokens[j+1]}", source_id, source_type))
                if len(processed_tokens) > 2:
                    for j in range(len(processed_tokens) - 2):
                        ngrams_list.append((f"{processed_tokens[j]} {processed_tokens[j+1]} {processed_tokens[j+2]}", source_id, source_type))
                return ngrams_list

            with ThreadPoolExecutor(max_workers=24) as executor: #os.cpu_count() or 4) as executor: 
                all_ngrams = executor.map(extract_ngrams_batch_func, batch_texts_with_source_info)
                for ngram_list in all_ngrams:
                    for gram_text, source_id, source_type in ngram_list:
                        meta = current_batch_ngrams_meta.setdefault(gram_text, {'sources': set(), 'count': 0})
                        if len(meta['sources']) < MAX_SOURCES_PER_NGRAM:
                            meta['sources'].add((source_id, source_type))
                        meta['count'] += 1
            
            batch_ngrams_texts = []
            batch_ngrams_ids = []
            batch_ngrams_metadatas = []

            MAX_NGRAMS_TO_EMBED_PER_BATCH = 75000 
            sorted_batch_ngrams = sorted(current_batch_ngrams_meta.items(), key=lambda item: item[1]['count'], reverse=True)

            for gram_text, meta in sorted_batch_ngrams[:MAX_NGRAMS_TO_EMBED_PER_BATCH]: 
                batch_ngrams_texts.append(gram_text)
                batch_ngrams_ids.append(str(uuid4()))
                source_refs_str = ",".join([f"{sid}:{stype}" for sid, stype in meta['sources']])
                batch_ngrams_metadatas.append({'text': gram_text, 'sources': source_refs_str, 'count': meta['count'], 'source_type': list(meta['sources'])[0][1] if meta['sources'] else 'doc_fallback'}) 

            if not batch_ngrams_texts:
                print(f"Skipping empty N-gram batch {i//N_GRAM_PROCESSING_BATCH_SIZE + 1}.")
                continue

            print(f"Generating embeddings for {len(batch_ngrams_texts)} N-grams in batch {i//N_GRAM_PROCESSING_BATCH_SIZE + 1}...")
            n_gram_embeddings = self.embedding_model.encode(batch_ngrams_texts, convert_to_tensor=False, device='cuda') 

            for j in range(0, len(batch_ngrams_ids), CHROMA_ADD_SUB_BATCH_SIZE):
                sub_batch_ids = batch_ngrams_ids[j:j+CHROMA_ADD_SUB_BATCH_SIZE]
                sub_batch_embeddings = n_gram_embeddings[j:j+CHROMA_ADD_SUB_BATCH_SIZE]
                sub_batch_metadatas = batch_ngrams_metadatas[j:j+CHROMA_ADD_SUB_BATCH_SIZE]
                
                self.suggestions_collection.add(
                    embeddings=sub_batch_embeddings.tolist(),
                    metadatas=sub_batch_metadatas,
                    ids=sub_batch_ids
                )
                total_ngrams_added += len(sub_batch_ids)
            print(f"Processed source batch {i//N_GRAM_PROCESSING_BATCH_SIZE + 1}. Total N-grams added so far: {total_ngrams_added}. Batch time: {time.time() - batch_start_time:.2f} seconds.")
            
            del batch_texts_with_source_info
            del current_batch_ngrams_meta
            del batch_ngrams_texts
            del batch_ngrams_ids
            del batch_ngrams_metadatas
            del n_gram_embeddings 
            gc.collect()

        print(f"Finished processing all text sources. Total N-grams added to ChromaDB: {total_ngrams_added}")
        
        os.makedirs(os.path.dirname(self._get_index_file_path()), exist_ok=True)
        with open(self._get_index_file_path(), 'w') as f:
            f.write("indexed")
        print("Query suggestion index built and saved successfully in ChromaDB.")


    def _load_or_build_index(self):
        if not self._load_index():
            self._build_index()

    def _extract_snippet(self, full_text_from_db: str, phrase: str, window_size: int = 150) -> str: # **تعديل: زيادة window_size للمقتطف أكثر**
        """
        يستخرج مقتطفاً حول عبارة معينة في نص كامل (يُستخدم للمقتطف الفعلي الذي يظهر بجانب الاقتراح).
        """
        if not full_text_from_db or not phrase:
            return ""
        
        full_text_lower = full_text_from_db.lower()
        phrase_lower = phrase.lower()

        try:
            start_index = full_text_lower.index(phrase_lower)
            end_index = start_index + len(phrase_lower)

            snippet_start = max(0, start_index - window_size)
            snippet_end = min(len(full_text_lower), end_index + window_size)

            snippet = full_text_from_db[snippet_start:snippet_end]
            
            if snippet_start > 0:
                snippet = "..." + snippet
            if snippet_end < len(full_text_lower):
                snippet = snippet + "..."
            
            return snippet.replace('\n', ' ').strip() 
        except ValueError:
            return full_text_from_db[:window_size * 2].replace('\n', ' ').strip() + "..." if len(full_text_from_db) > window_size * 2 else full_text_from_db.replace('\n', ' ').strip()


    def _get_autocomplete_phrase_from_snippet(self, original_raw_text: str, matched_ngram_text: str, target_word_count: int = 8) -> str: # **تعديل: تقليل target_word_count هنا!**
        """
        يحاول استخراج عبارة إكمال تلقائي قصيرة وطبيعية (هدفها 5-8 كلمات) من النص الأصلي
        بناءً على الـ N-gram المطابق، مع تحديد طولها بعدد الكلمات.
        """
        if not original_raw_text or not matched_ngram_text:
            return matched_ngram_text.replace('\n', ' ').strip() 
        
        original_words = word_tokenize(original_raw_text)
        matched_ngram_words = word_tokenize(matched_ngram_text)

        if not matched_ngram_words:
            return matched_ngram_text.replace('\n', ' ').strip()

        for i in range(len(original_words) - len(matched_ngram_words) + 1):
            if original_words[i:i + len(matched_ngram_words)] == matched_ngram_words:
                
                phrase_start_token_idx = max(0, i - (target_word_count - len(matched_ngram_words)) // 2)
                
                phrase_end_token_idx = min(len(original_words), i + len(matched_ngram_words) + (target_word_count - len(matched_ngram_words)) // 2 + (target_word_count - len(matched_ngram_words)) % 2)
                
                if (phrase_end_token_idx - phrase_start_token_idx) > target_word_count:
                    phrase_end_token_idx = phrase_start_token_idx + target_word_count
                
                phrase_tokens = original_words[phrase_start_token_idx:phrase_end_token_idx]
                phrase = " ".join(phrase_tokens).strip()

                phrase = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', phrase) 
                
                try:
                    matched_ngram_text_lower = matched_ngram_text.lower()
                except Exception:
                    matched_ngram_text_lower = str(matched_ngram_text).lower()

                if matched_ngram_text_lower in phrase.lower() and \
                   (len(phrase.split()) > len(matched_ngram_text.split()) or \
                    (len(phrase.split()) == len(matched_ngram_text.split()) and phrase.lower() == matched_ngram_text_lower)):
                    return phrase
                else:
                    return matched_ngram_text.replace('\n', ' ').strip() 

        return matched_ngram_text.replace('\n', ' ').strip() 


    def get_suggestions(self, query_prefix: str, top_k: int = 10) -> list[tuple[str, str]]:
        if not hasattr(self, 'suggestions_collection') or self.suggestions_collection is None:
            print("ChromaDB suggestions collection not initialized. Please rebuild index.")
            return []

        corrected_query_tokens = []
        for word in query_prefix.split():
            corrected_word = self.spell.correction(word.lower()) 
            if corrected_word:
                corrected_query_tokens.append(corrected_word)
            else:
                corrected_query_tokens.append(word) 

        processed_prefix_after_spellcheck = " ".join(corrected_query_tokens)

        processed_prefix_final_tokens = self.preprocessor.preprocess_text(processed_prefix_after_spellcheck, remove_stopwords_flag=True) 
        processed_prefix = " ".join(processed_prefix_final_tokens)


        if not processed_prefix:
            return []

        query_embedding = self.embedding_model.encode(processed_prefix, convert_to_tensor=False, device='cuda') 

        search_results = self.suggestions_collection.query(
            query_embeddings=[query_embedding.tolist()], 
            n_results=top_k * 5, 
            include=['metadatas', 'distances']
        )
        
        candidate_suggestions_with_details = []

        if (
            search_results and
            search_results.get('ids') is not None and
            search_results.get('metadatas') is not None and
            search_results.get('distances') is not None and
            isinstance(search_results['ids'], list) and
            isinstance(search_results['metadatas'], list) and
            isinstance(search_results['distances'], list) and
            len(search_results['ids']) > 0 and
            len(search_results['metadatas']) > 0 and
            len(search_results['distances']) > 0
        ):
            for i in range(len(search_results['ids'][0])):
                try:
                    gram_text = search_results['metadatas'][0][i]['text']
                    source_refs_str = search_results['metadatas'][0][i]['sources']
                    distance = search_results['distances'][0][i]
                    embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                    similarity = 1 - (distance / (embedding_dim**0.5)) if embedding_dim else 1 - distance

                    source_references = set()
                    if isinstance(source_refs_str, str):
                        for ref_str in source_refs_str.split(','):
                            if ref_str:
                                try:
                                    sid, stype = ref_str.split(':', 1)
                                    source_references.add((sid, stype))
                                except ValueError:
                                    pass

                    candidate_suggestions_with_details.append((gram_text, similarity, source_references))
                except Exception:
                    continue

        COUNT_WEIGHT = 0.1 
        for i, (gram_text, sim, refs) in enumerate(candidate_suggestions_with_details):
            original_gram_count = 0
            if (
                search_results and
                search_results.get('metadatas') is not None and
                isinstance(search_results['metadatas'], list) and
                len(search_results['metadatas']) > 0 and
                isinstance(search_results['metadatas'][0], list) and
                len(search_results['metadatas'][0]) > i and
                isinstance(search_results['metadatas'][0][i], dict)
            ):
                val = search_results['metadatas'][0][i].get('count', 0)
                if val is not None:
                    try:
                        original_gram_count = float(val)
                    except Exception:
                        original_gram_count = 0
                else:
                    original_gram_count = 0
            try:
                weighted_score = sim + COUNT_WEIGHT * np.log1p(original_gram_count)
            except Exception:
                weighted_score = sim
            candidate_suggestions_with_details[i] = (gram_text, weighted_score, refs)


        candidate_suggestions_with_details.sort(key=lambda x: x[1], reverse=True)
        
        final_suggestions_with_snippets = []
        seen_suggestions_text_for_diversity = set() 
        seen_source_ids_for_diversity = set() 

        for suggestion_gram_text, score, source_references in candidate_suggestions_with_details: 
            
            display_suggestion_text = ""
            display_snippet_text = ""
            current_suggestion_source_id = None

            if source_references:
                source_id, source_type = next(iter(source_references)) 
                current_suggestion_source_id = source_id 

                original_text = ""
                if source_type == 'doc':
                    original_text = self.db_connector.get_document_text_by_id(source_id, self.dataset_name, cleaned=False)
                elif source_type == 'query' and source_id in self.all_queries_by_id:
                     original_text = self.all_queries_by_id[source_id] 

                if original_text:
                    display_suggestion_text = self._get_autocomplete_phrase_from_snippet(original_text, suggestion_gram_text)
                    display_snippet_text = self._extract_snippet(original_text, suggestion_gram_text)
                
                if not display_suggestion_text:
                    display_suggestion_text = suggestion_gram_text 
                    display_snippet_text = self._extract_snippet(original_text if original_text is not None else "", suggestion_gram_text) 
                
                if display_suggestion_text in seen_suggestions_text_for_diversity: 
                    continue

                if current_suggestion_source_id: 
                    if current_suggestion_source_id in seen_source_ids_for_diversity and len([s_snip for s_text, s_snip in final_suggestions_with_snippets if s_snip and current_suggestion_source_id in s_snip]) >= 1: 
                        continue
                
                final_suggestions_with_snippets.append((display_suggestion_text, display_snippet_text))
                seen_suggestions_text_for_diversity.add(display_suggestion_text)
                if current_suggestion_source_id:
                    seen_source_ids_for_diversity.add(current_suggestion_source_id)
            
            if len(final_suggestions_with_snippets) >= top_k: 
                break
        
        return final_suggestions_with_snippets

