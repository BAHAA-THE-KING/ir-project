from typing import List, Tuple, Dict

def hybrid_search(query: str, docs: Dict, queries: Dict, qrels: Dict, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Performs a hybrid search combining multiple ranking methods.
    Currently returns dummy results for testing.
    
    Args:
        query: The search query string
        top_k: Number of top results to return (default: 10)
        
    Returns:
        List of tuples containing (document_id, score)
    """
    results = [(i.doc_id, 0.8) for i in docs[0:3]]
    return results[:top_k]
