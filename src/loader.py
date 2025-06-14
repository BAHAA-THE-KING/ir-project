import ir_datasets
import codecs

def load_dataset(name: str = "antique"):
    dataset = ir_datasets.load(name)
    dataset_test = ir_datasets.load(name+'/test')
    
    # Use UTF-8 encoding when reading the dataset
    docs = list(dataset.docs_iter())
    queries = list(dataset_test.queries_iter())
    qrels = list(dataset_test.qrels_iter())
    return docs, queries, qrels