import ir_datasets

def load_dataset(name: str = "antique"):
    dataset = ir_datasets.load(name)
    dataset_test = ir_datasets.load(name+'/test')
    
    docs = list(dataset.docs_iter())
    queries = list(dataset_test.queries_iter())
    qrels = list(dataset_test.qrels_iter())
    return docs, queries, qrels