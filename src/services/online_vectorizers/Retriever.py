import math

def calc_dcg(relevance, rank):
    return ((2 ** relevance) - 1) / math.log10(rank + 1)

class Retriever:
    def search(self, dataset_name: str, query: str, top_k: int = 10, with_index: bool = True) -> list[tuple[str, float, str]]:
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

    def evaluateMRR(self, dataset_name, queries, qrels, K = 100, print_more = False):
        MRR = []
        for i in range(len(queries)):
            query=queries[i]
            results = self.search(dataset_name, query.text, K, True)
            
            firstRank = 100
            for ii, res in enumerate(results):
                if len([qrel for qrel in qrels if qrel.query_id == query.query_id and qrel.doc_id == res[0] and qrel.relevance != 0]) == 1:
                    firstRank = ii + 1
            
            MRR.append(1/firstRank)
            
            if print_more:
                print()
                print(f"Query: {i+1}/{len(queries)}")
                print(f"Current MRR: {sum(MRR)/len(MRR)}")
        
        MRR = sum(MRR)/len(MRR)
        if print_more:
            print(f"MRR: {MRR}")
        return MRR
    
    def evaluateMAP(self, dataset_name, queries, qrels, K = 10, print_more = False):
        AP = []
        for i in range(len(queries)):
            query = queries[i]
            results = self.search(dataset_name, query.text, K, True)

            relevant_num = 0
            precision_sum = 0
            for res in results:
                if len([qrel for qrel in qrels if qrel.query_id == query.query_id and qrel.doc_id == res[0] and qrel.relevance != 0]) == 1:
                    relevant_num += 1
                    precision_sum += relevant_num / (i + 1)
            if relevant_num > 0:
                AP.append(precision_sum / relevant_num)
            if print_more:
                print()
                print(f'Query: {i+1}/{len(queries)}')
                if len(AP) > 0:
                    print(f'AP = {sum(AP) / len(AP) * 100}')
        MAP = sum(AP) / len(AP)
        if print_more:
            print(f'MAP={MAP}')
        return MAP
