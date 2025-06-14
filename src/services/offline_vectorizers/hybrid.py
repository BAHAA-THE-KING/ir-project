from typing import List, Tuple

def hybrid_search(query: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Performs a hybrid search combining multiple ranking methods.
    Currently returns dummy results for testing.
    
    Args:
        query: The search query string
        top_k: Number of top results to return (default: 10)
        
    Returns:
        List of tuples containing (document_id, score)
    """
    results = [("dummy_doc_1", 0.8), ("dummy_doc_2", 0.6)]
    return results[:top_k]
