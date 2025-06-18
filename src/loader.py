import ir_datasets

def load_dataset(name: str):
    dataset = ir_datasets.load(name)
    
    docs = list(dataset.docs_iter())
    return docs

def load_dataset_with_queries(name: str):
    dataset = ir_datasets.load(name)
    dataset_test = ir_datasets.load(name+'/test')
    
    docs = list(dataset.docs_iter())
    queries = list(dataset_test.queries_iter())
    qrels = list(dataset_test.qrels_iter())
    return docs, queries, qrels