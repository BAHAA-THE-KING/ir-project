from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import time
import sys
import os
from src.services.processing.text_preprocessor import TextPreprocessor
import httpx

# Adjust sys.path to allow imports from the 'src' directory
# This assumes api_main.py is in the project root, and 'src' is a subdirectory.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if os.path.join(project_root, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(project_root, 'src'))

try:
    from src.gui.ir_engine import IREngine, SearchModel
    from src.config import DATASETS
    print("Successfully imported IREngine and DATASETS from src.")
except Exception as e:
    print(f"Error importing modules: {e}. Please ensure your Python path is correctly configured.")
    print("If you are running this from a different directory, you might need to manually add your project root and src directory to sys.path.")
    # Fallback/Mock for demonstration if imports fail due to environment setup
    # This MockIREngine is updated to match the new search signature required by the API.
    class SearchModel:
        TFIDF = "TF-IDF"
        BM25 = "BM25"
        HYBRID = "Hybrid"
        EMBEDDING = "Embedding"
        @staticmethod
        def list_values():
            return [SearchModel.TFIDF, SearchModel.BM25, SearchModel.HYBRID, SearchModel.EMBEDDING]

    class MockIREngine:
        def __init__(self):
            print("Using Mock IREngine.")
            self.current_dataset = "antique" # Mock dataset active by default
            self.mock_datasets = {
                "antique": {"name": "antique", "description": "Question-answer dataset (mock)", "ir_datasets_id": "antique", "ir_datasets_test_id": "antique/test/non-offensive"},
            }
            self.docs = {
                "doc1": type('obj', (object,), {'doc_id': 'doc1', 'text': 'This is the full text of document 1 for the mock search.'}),
                "doc2": type('obj', (object,), {'doc_id': 'doc2', 'text': 'Here is the content of document number 2 in the mock system.'}),
                "doc3": type('obj', (object,), {'doc_id': 'doc3', 'text': 'Document 3 contains some interesting information for testing purposes.'})
            }
            self.queries = []
            self.qrels = []

        def get_available_datasets(self) -> List[str]:
            return list(self.mock_datasets.keys())

        def get_dataset_stats(self, dataset_name: str) -> Dict:
            # Simulate stats for mock datasets, now takes dataset_name
            dataset_config = self.mock_datasets.get(dataset_name, {})
            return {
                "name": dataset_config.get("name", "Unknown"),
                "description": dataset_config.get("description", "No description"),
                "num_docs": len(self.docs) * 10, # Mock large numbers
                "num_queries": 50,
                "num_qrels": 200
            }

        # Updated search method to match new signature
        def search(self, model_name: str, query: str, top_k: int = 10,
                   use_inverted_index: bool = False, use_vector_store: bool = False,
                   include_cluster_info: bool = False) -> List[Tuple[str, float, str]]:
            print(f"Mock search for '{query}' with model '{model_name}'. Inverted Index: {use_inverted_index}, Vector Store: {use_vector_store}, Cluster Info: {include_cluster_info}")
            # Simulate some results based on mock documents
            mock_results = [
                ("doc1", 0.95, self.docs["doc1"].text[:80] + "..."),
                ("doc2", 0.88, self.docs["doc2"].text[:80] + "..."),
                ("doc3", 0.75, self.docs["doc3"].text[:80] + "...")
            ]
            return [(item[0], item[1], item[2]) for item in mock_results[:top_k]]

        def get_document(self, doc_id: str) -> Optional[str]:
            print(f"Mock: Getting document {doc_id}")
            doc_obj = self.docs.get(doc_id)
            return doc_obj.text if doc_obj else None

    IREngine = MockIREngine
    DATASETS = {k: v for k, v in IREngine().mock_datasets.items()} # Use mock datasets for consistency

app = FastAPI(
    title="Information Retrieval System API",
    description="API for searching documents using various IR models."
)

# Initialize the IR Engine globally to maintain state across requests
ir_engine = IREngine()

# Pydantic models for request and response validation
class SearchRequest(BaseModel):
    model: str # Specify model for this search
    query: str
    dataset: str # Add dataset as a required string argument
    top_k: int = 10
    use_inverted_index: bool = False # For TF-IDF/BM25
    use_vector_store: bool = False # For Embedding models
    include_cluster_info: bool = False # For future cluster integration

class SearchResultItem(BaseModel):
    doc_id: str
    score: float
    snippet: str

class SearchResponse(BaseModel):
    query: str
    time_taken: float
    results: List[SearchResultItem]

# --- New Preprocess Endpoint ---
from fastapi import Body

class PreprocessRequest(BaseModel):
    query: str

class PreprocessResponse(BaseModel):
    preprocessed: List[str]

@app.post("/preprocess", response_model=PreprocessResponse, summary="Preprocess a query using TextPreprocessor")
async def preprocess_query(request: PreprocessRequest):
    preprocessor = TextPreprocessor.getInstance()
    processed = preprocessor.preprocess_text(request.query)
    # Always return a list, even if empty or if processed is a string
    if isinstance(processed, list):
        return PreprocessResponse(preprocessed=processed)
    elif isinstance(processed, str):
        return PreprocessResponse(preprocessed=[processed] if processed else [])
    else:
        return PreprocessResponse(preprocessed=[])

# --- API Endpoints ---

# ✅ /datasets
@app.get("/datasets", response_model=List[str], summary="Get available dataset names")
async def get_available_datasets():
    """
    Retrieves a list of all available dataset names supported by the IR system.
    """
    return ir_engine.get_available_datasets()


# ✅ /search
@app.post("/search", response_model=SearchResponse, summary="Perform a search query")
async def perform_search(request: SearchRequest):
    """
    Performs a search using the specified model and the currently loaded dataset.

    - **model**: The search model to use (e.g., "TF-IDF", "BM25", "Hybrid", "Embedding").
    - **query**: The search query string.
    - **dataset**: The dataset to use for the search.
    - **top_k**: The number of top relevant documents to retrieve (default: 10).
    - **use_inverted_index**: Boolean indicating whether to use an inverted index for TF-IDF/BM25 models (default: False).
    - **use_vector_store**: Boolean indicating whether to use a vector store for Embedding models (default: False).
    - **include_cluster_info**: Boolean for future functionality to include cluster information in results (default: False).
    """
    start_time = time.time()
    try:
        # Switch to the requested dataset before searching
        ir_engine.change_dataset(request.dataset)
        # Call the /preprocess endpoint to get the preprocessed query
        async with httpx.AsyncClient() as client:
            preprocess_response = await client.post(
                "http://127.0.0.1:8000/preprocess",
                json={"query": request.query}
            )
            preprocess_data = preprocess_response.json()
            preprocessed_query = preprocess_data["preprocessed"]
        # Use the preprocessed query for searching (join tokens back to string if needed)
        query_for_search = " ".join(preprocessed_query)
        results = ir_engine.search(
            model_name=request.model,
            query=query_for_search,
            top_k=request.top_k,
            use_inverted_index=request.use_inverted_index,
            use_vector_store=request.use_vector_store,
            include_cluster_info=request.include_cluster_info
        )
        formatted_results = [
            SearchResultItem(doc_id=str(item[0]), score=float(item[1]), snippet=str(item[2]))
            for item in results
        ]
        end_time = time.time()
        time_taken = end_time - start_time
        return SearchResponse(
            query=request.query,
            time_taken=time_taken,
            results=formatted_results
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# ✅ /document/{doc_id}
@app.get("/document/{doc_id}", response_model=str, summary="Get full text content of a document")
async def get_document_content(doc_id: str):
    """
    Retrieves the full text content of a document given its document ID.
    """
    try:
        document_text = ir_engine.get_document(doc_id)
        if document_text is None:
            raise HTTPException(status_code=404, detail=f"Document with ID '{doc_id}' not found.")
        return document_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

# To run this FastAPI application:
# 1. Save the code as api_main.py in your project's root directory.
# 2. Make sure src/gui/ir_engine.py has its 'search' method updated as described above.
# 3. Open your terminal in the project root and run: uvicorn api_main:app --reload --port 8000
# 4. Access interactive docs at: http://127.0.0.1:8000/docs