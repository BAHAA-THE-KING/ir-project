import math
from services.processing.text_preprocessor import TextPreprocessor

def calc_dcg(relevance, rank):
    return ((2 ** relevance) - 1) / math.log2(rank + 1)

class Retriever:
    def search(self, dataset_name: str, query: str, top_k: int = 10, with_index: bool = True) -> list[tuple[str, float, str]]:
        raise NotImplementedError()
    
    def evaluateNDCG(self, dataset_name, queries, qrels, docs, K = 10, print_more = False):
        nDCG = []

        for i in range(len(queries)):
            query = queries[i]
            if print_more:
                preprocess_text = TextPreprocessor.getInstance().preprocess_text
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
            nDCG.append(res/ires)
            
            if print_more:
                print("")
                print(f"query: {i+1}/{len(queries)}")
                print(f"DCG: {res}")
                print(f"iDCG: {ires}")
                print(f"nDCG: {res/ires*100}%")
            if print_more:
                print(f"Average nDCG: {sum(nDCG)/len(nDCG)*100}%")
        
        nDCG = sum(nDCG)/len(nDCG)*100

        if print_more:
            print(f"Final Average nDCG: {nDCG}%")

        return nDCG

    def evaluateMRR(self, dataset_name, queries, qrels, K = 100, print_more = False):
        MRR = []

        cleaned_qrels: dict[str, dict[str, int]] = {}
        for qrel in qrels:
            if qrel.query_id not in cleaned_qrels.keys():
                cleaned_qrels[qrel.query_id] = {}
            cleaned_qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

        for i in range(len(queries)):
            query = queries[i]
            results = self.search(dataset_name, query.text, K, True)
            
            firstRank = 100
            for ii, res in enumerate(results):
                if res[0] in cleaned_qrels[query.query_id].keys() and cleaned_qrels[query.query_id][res[0]] > 0:
                    firstRank = ii + 1
                    break
            
            MRR.append(1/firstRank)
            
            if print_more:
                print()
                print(f"Query: {i+1}/{len(queries)}")
                print(f"Current MRR: {sum(MRR) / len(MRR) * 100}")
        
        MRR = sum(MRR) / len(MRR) * 100
        if print_more:
            print(f"MRR: {MRR}%")
        return MRR
    
    def evaluateMAP(self, dataset_name, queries, qrels,docs, K = 10, print_more = False):
        MAP = []

        cleaned_qrels: dict[str, dict[str, int]] = {}
        for qrel in qrels:
            if qrel.query_id not in cleaned_qrels.keys():
                cleaned_qrels[qrel.query_id] = {}
            cleaned_qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
        
        for i in range(len(queries)):
            query = queries[i]
            if print_more:
                print()
                print(f'Query: {i+1}/{len(queries)}')
                print(query.text)

            results = self.search(dataset_name, query.text, K, True)
            if print_more:
                print([res[0] for res in results])
                print([qrel.doc_id+f": {qrel.relevance}" for qrel in qrels if qrel.query_id == query.query_id])
                print("results")
                for doc in [doc for doc in docs if doc.doc_id in [res[0] for res in results]]:
                    print(doc.doc_id+" "+doc.text)
                print("qrels")
                koko = [qrel.doc_id for qrel in qrels if qrel.query_id == query.query_id]
                for doc in [doc for doc in docs if doc.doc_id in koko]:
                    print(doc.doc_id+" "+doc.text)

            relevant_num = 0
            precision_sum = 0
            for ii, res in enumerate(results):
                if res[0] in cleaned_qrels[query.query_id].keys() and cleaned_qrels[query.query_id][res[0]] > 0:
                    relevant_num += 1
                    precision_sum += relevant_num / (ii + 1)
            if print_more:
                print(precision_sum)
            if relevant_num > 0:
                MAP.append(precision_sum / relevant_num)
            if print_more:
                if len(MAP) > 0:
                    print(f'MAP = {sum(MAP) / len(MAP) * 100}')
        if len(MAP) > 0:
            MAP = sum(MAP) / len(MAP) * 100
        else:
            MAP = 0
        if print_more:
            print(f'MAP={MAP}%')
        return MAP

    def evaluateAll(self, dataset_name, queries, qrels, K = 10):
            MRR = []
            MAP = []
            nDCG = []

            cleaned_qrels: dict[str, dict[str, int]] = {}
            for qrel in qrels:
                if qrel.query_id not in cleaned_qrels.keys():
                    cleaned_qrels[qrel.query_id] = {}
                cleaned_qrels[qrel.query_id][qrel.doc_id] = qrel.relevance

            for i in range(len(queries)):
                query = queries[i]
                results = self.search(dataset_name, query.text, K * 10, True)
               
                # MRR calc
                firstRank = 100
                for j, res in enumerate(results):
                    if res[0] in cleaned_qrels[query.query_id] and cleaned_qrels[query.query_id][res[0]] > 0:
                        # MRR calc
                        firstRank = j + 1
                        break
                
                # MAP calc
                relevant_num = 0
                precision_sum = 0
                # nDCG calc
                DCG = []
                iDCG = []
                results = results[:K]
                for jj, ress in enumerate(results):
                    if ress[0] in cleaned_qrels[query.query_id] and cleaned_qrels[query.query_id][ress[0]] > 0:
                        # MAP calc
                        relevant_num += 1
                        precision_sum += relevant_num / (jj + 1)
                        # nDCG calc
                        DCG.append(calc_dcg(cleaned_qrels[query.query_id][ress[0]], jj+1))
                    else:
                        # nDCG calc
                        DCG.append(calc_dcg(0, jj+1))
                
                # MRR calc
                MRR.append(1 / firstRank)
                
                # MAP calc
                if relevant_num > 0:
                    MAP.append(precision_sum / relevant_num)
                
                # nDCG calc
                iDCG = [calc_dcg(qrel[1], iii+1) for iii, qrel in enumerate(sorted(list(cleaned_qrels[query.query_id].items()),key = lambda qrel:qrel[1], reverse=True)[:K])]
                res = sum(DCG)
                ires = sum(iDCG)
                nDCG.append(res/ires)
               
            # MRR
            MRR = sum(MRR) / len(MRR) * 100
            
            # MAP    
            if len(MAP) > 0:
                MAP = sum(MAP) / len(MAP) * 100
            else:
                MAP = 0
            
            #nDCG
            nDCG = sum(nDCG) / len(nDCG) * 100
            
            return {
                "MRR": MRR,
                "MAP": MAP,
                "nDCG": nDCG
            }