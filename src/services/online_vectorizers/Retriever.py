import math

def calc_dcg(relevance, rank):
    return ((2 ** relevance) - 1) / math.log10(rank + 1)

class Retriever:
    def search(self, dataset_name: str, query: str, top_k: int = 10, with_index: bool = True) -> list[tuple[int, float, str]]:
        raise NotImplementedError()
    
    def evaluateNDCG(self, dataset_name, queries, qrels, docs, K = 10, print_more = False):
        nDCG = []

        for i in range(len(queries)):
            query = queries[i]
            if print_more:
                preprocess_text = preprocess_text
                print(f"Query: {query.text}")
                print(f"Query: {preprocess_text(query.text)}")
            
            # Search
            results = self.search(dataset_name, query.text, K, True)
            if print_more:
                for i, res in enumerate(results):
                    print(f"Result #{i} {res[1]}: {res[2]}")
                    print(f"Result #{i} {res[1]}: {preprocess_text(res[2])}")

            # Find relevant documents for this query
            relevant_qrels = [qrel for qrel in qrels if qrel.query_id == query.query_id]
            relevant_qrels = sorted(relevant_qrels, key=lambda x: x.relevance, reverse=True)
            if print_more:
                for i, qrel in enumerate(relevant_qrels[:K]):
                    doc = [doc for doc in docs if qrel.doc_id == doc.doc_id][0]
                    print(f"Qrel #{i} {qrel.relevance}: {doc.text}")
                    print(f"Qrel #{i} {qrel.relevance}: {preprocess_text(doc.text)}")
            
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
            print(f"query: {i+1}/{len(queries)}")
            print(f"DCG: {res}")
            print(f"iDCG: {ires}")
            print(f"nDCG: {res/ires*100}%")
            nDCG.append(res/ires)
            print(f"Average nDCG: {sum(nDCG)/len(nDCG)*100}%")
        
        print(f"Final Average nDCG: {sum(nDCG)/len(nDCG)*100}%")

    def evaluateMAP(self):
        raise NotImplementedError()