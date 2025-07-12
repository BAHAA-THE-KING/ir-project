import ir_datasets
from src.config import DATASETS

from typing import TypeAlias

from collections import namedtuple

Doc = namedtuple('Doc', ['doc_id', 'text'])
Query = namedtuple('Query', ['query_id', 'text'])
Qrel = namedtuple('Qrel', ['query_id', 'doc_id', 'relevance', 'iteration'])

def load_dataset(name: str) -> list[Doc]:
    dataset = ir_datasets.load(DATASETS[name]['ir_datasets_id'])
    
    docs = list(dataset.docs_iter())
    return docs

def load_queries_and_qrels(name: str) -> tuple[list[Query], list[Qrel]]:
    dataset_test = ir_datasets.load(DATASETS[name]['ir_datasets_test_id'])
    
    queries = list(dataset_test.queries_iter())
    qrels = list(dataset_test.qrels_iter())
    return queries, qrels

def load_dataset_with_queries(name: str) -> tuple[list[Doc], list[Query], list[Qrel]]:
    dataset = ir_datasets.load(DATASETS[name]['ir_datasets_id'])
    dataset_test = ir_datasets.load(DATASETS[name]['ir_datasets_test_id'])
    
    docs = list(dataset.docs_iter())
    queries = list(dataset_test.queries_iter())
    qrels = list(dataset_test.qrels_iter())
    
    return docs, queries, qrels