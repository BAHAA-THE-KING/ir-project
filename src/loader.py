import ir_datasets
from config import DATASETS

def load_dataset(name: str):
    dataset = ir_datasets.load(DATASETS[name]['ir_datasets_id'])
    
    docs = list(dataset.docs_iter())
    return docs

def load_queries_and_qrels(name: str):
    dataset_test = ir_datasets.load(DATASETS[name]['ir_datasets_test_id'])
    
    queries = list(dataset_test.queries_iter())
    qrels = list(dataset_test.qrels_iter())
    return queries, qrels

def load_dataset_with_queries(name: str):
    dataset = ir_datasets.load(DATASETS[name]['ir_datasets_id'])
    dataset_test = ir_datasets.load(DATASETS[name]['ir_datasets_test_id'])
    
    docs = list(dataset.docs_iter())
    queries = list(dataset_test.queries_iter())
    qrels = list(dataset_test.qrels_iter())
    
    return docs, queries, qrels