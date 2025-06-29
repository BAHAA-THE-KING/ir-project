import dill
from rank_bm25 import BM25Okapi
from services.online_vectorizers.inverted_index import InvertedIndex
from services.processing.bm25_preprocessing import AntiqueTextProcessor, QuoraTextProcessor
from loader import load_dataset
import math
from collections import namedtuple

class BM25_online:
    __bm25instance__ : dict[str, BM25Okapi] = {}
    __invertedIndex__ : dict[str, InvertedIndex] = {}
    @staticmethod
    def loadInstance(dataset_name : str):
        if BM25_online.__bm25instance__[dataset_name] == None:
            with open(f"data/{dataset_name}/bm25_model.dill", "rb") as f:
                BM25_online.__bm25instance__[dataset_name] = dill.load(f) 
    @staticmethod
    def loadInvertedIndex(dataset_name : str):
        if BM25_online.__invertedIndex__[dataset_name] == None:
            with open(f"data/{dataset_name}/inverted_index.dill", "rb") as f:
                inverted_index = InvertedIndex()
                ii = dill.load(f)
                inverted_index.index = ii.index
                inverted_index.doc_lengths = ii.doc_lengths
                inverted_index.N = ii.N
                BM25_online.__invertedIndex__[dataset_name] = inverted_index

    @staticmethod
    def bm25_search(dataset_name: str, query: str, top_k: int = 10, with_inverted_index: bool = False) -> list[tuple[int, float, str]]:
        # Load the model and the documents
        BM25_online.loadInstance(dataset_name)
        bm25 = BM25_online.__bm25instance__[dataset_name]
        docs = load_dataset(dataset_name)
        if with_inverted_index:
            BM25_online.loadInvertedIndex(dataset_name)
            inverted_index = BM25_online.__invertedIndex__[dataset_name]

        # Execute the query
        if dataset_name == "antique":
            query_tokens = AntiqueTextProcessor.preprocess_text(query)
        elif dataset_name == "quora":
            query_tokens = QuoraTextProcessor.preprocess_text(query)
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        if with_inverted_index:
            documents_sharing_terms_with_query = inverted_index.get_documents_sharing_terms_with_query(query_tokens)
            scores = bm25.get_batch_scores(query_tokens, documents_sharing_terms_with_query)
        else:
            scores = bm25.get_scores(query_tokens)

        # Sort the results
        if with_inverted_index:
            top_indices = sorted(list(enumerate(documents_sharing_terms_with_query)), key=lambda  elm: scores[elm[0]], reverse=True)[:top_k]
        else:
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []

        # Display the results
        if with_inverted_index:
            for elm in top_indices:
                text = docs[elm[1]].text
                results.append((docs[elm[1]].doc_id, scores[elm[0]], text))
        else:
            for idx in top_indices:
                text = docs[idx].text
                results.append((docs[idx].doc_id, scores[idx], text))
        
        return results

    @staticmethod
    def calc_dcg(rel, rank):
        return ((2 ** rel) - 1) / math.log10(rank + 1)

    @staticmethod
    def evaluate_bm25(dataset_name, queries, qrels, K = 10):
        nDCG = []

        for i in range(len(queries)):
            query = queries[i]
            # print(f"Query: {query.text}")
            # print(f"Query: {bm25_preprocess_text(query.text)}")
            
            # Search using BM25
            results = BM25_online.bm25_search(dataset_name, query.text, K, True)
            # for i, res in enumerate(results):
                # print(f"Result #{i} {res[1]}: {res[2]}")
                # print(f"Result #{i} {res[1]}: {bm25_preprocess_text(res[2])}")

            # Find relevant documents for this query
            relevant_qrels = [qrel for qrel in qrels if qrel.query_id == query.query_id]
            relevant_qrels = sorted(relevant_qrels, key=lambda x: x.relevance, reverse=True)
            # for i, qrel in enumerate(relevant_qrels[:K]):
            #     doc = [doc for doc in docs if qrel.doc_id == doc.doc_id][0]
                # print(f"Qrel #{i} {qrel.relevance}: {doc.text}")
                # print(f"Qrel #{i} {qrel.relevance}: {bm25_preprocess_text(doc.text)}")
            
            DCG = [
                calc_dcg(
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
            
            iDCG = [calc_dcg(qrel.relevance, i+1) for i, qrel in enumerate(relevant_qrels[:K])]
            
            res = sum(DCG) 
            ires = sum(iDCG) 
            
            print("")
            print(f"query: {i}")
            print(f"nDCG: {res}")
            print(f"iDCG: {ires}")
            print(f"nDCG: {res/ires*100}%")
            nDCG.append(res/ires)
        
        print(f"Average nDCG: {sum(nDCG)/len(nDCG)*100}%")
