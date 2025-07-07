from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import time
import sys
import os

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
except ImportError as e:
    print(f"Error importing modules: {e}. Please ensure your Python path is correctly configured.")
    print("If you are running this from a different directory, you might need to manually add your project root and src directory to sys.path.")
    # Fallback/Mock for demonstration if imports fail due to environment setup
    # In a real application, you'd want to fix the import paths.
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
            self.current_dataset = "antique"
            self.current_model = SearchModel.TFIDF
            self.mock_datasets = {
                "antique": {"name": "antique", "description": "Question-answer dataset (mock)", "ir_datasets_id": "antique", "ir_datasets_test_id": "antique/test/non-offensive"},
                "wikir/en1k": {"name": "wikir/en1k", "description": "Wikipedia-based dataset (mock)", "ir_datasets_id": "wikir/en1k", "ir_datasets_test_id": "wikir/en1k/test"}
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

        def get_dataset_info(self, dataset_name: str) -> Dict:
            return self.mock_datasets.get(dataset_name)

        def change_dataset(self, dataset_name: str) -> None:
            if dataset_name not in self.mock_datasets:
                raise ValueError(f"Dataset {dataset_name} not found in mock datasets")
            self.current_dataset = dataset_name
            print(f"Mock: Changed dataset to {dataset_name}")

        def get_dataset_stats(self) -> Dict:
            # Simulate stats for mock datasets
            dataset_config = self.mock_datasets.get(self.current_dataset, {})
            return {
                "name": dataset_config.get("name", "Unknown"),
                "description": dataset_config.get("description", "No description"),
                "num_docs": len(self.docs) * 10, # Mock large numbers
                "num_queries": 50,
                "num_qrels": 200
            }

        def get_available_models(self) -> List[str]:
            return SearchModel.list_values()

        def change_model(self, model_name: str) -> None:
            if model_name not in SearchModel.list_values():
                raise ValueError(f"Model {model_name} not found in mock models")
            self.current_model = model_name
            print(f"Mock: Changed model to {model_name}")

        def get_current_model(self) -> str:
            return self.current_model

        def search(self, query: str, top_k: int = 10, with_index: bool = False) -> List[Tuple[str, float, str]]:
            print(f"Mock search for '{query}' using model '{self.current_model}' and top_k={top_k}. with_index={with_index}")
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
    description="API for searching documents using various IR models and managing datasets."
)

# Initialize the IR Engine globally to maintain state across requests
ir_engine = IREngine()

# Pydantic models for request and response validation
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    # This flag will control the use of inverted index for TF-IDF/BM25
    # or vector store for Embedding models within IREngine.search
    use_index: bool = False

class SearchResultItem(BaseModel):
    doc_id: str
    score: float
    snippet: str

class SearchResponse(BaseModel):
    query: str
    time_taken: float
    results: List[SearchResultItem]

class DatasetInfoResponse(BaseModel):
    name: str
    description: str

class DatasetStatsResponse(BaseModel):
    name: str
    description: str
    num_docs: int
    num_queries: int
    num_qrels: int

@app.get("/datasets", response_model=List[str], summary="Get available dataset names")
async def get_available_datasets():
    """
    Retrieves a list of all available dataset names supported by the IR system.
    """
    return ir_engine.get_available_datasets()

@app.get("/datasets/info/{dataset_name}", response_model=DatasetInfoResponse, summary="Get information about a specific dataset")
async def get_dataset_info(dataset_name: str):
    """
    Retrieves detailed information about a specific dataset.
    """
    info = DATASETS.get(dataset_name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found")
    return DatasetInfoResponse(name=info["name"], description=info["description"])

@app.put("/datasets/current/{dataset_name}", summary="Change the current active dataset")
async def change_current_dataset(dataset_name: str):
    """
    Sets the current active dataset for subsequent search operations.
    """
    try:
        ir_engine.change_dataset(dataset_name)
        return {"message": f"Dataset changed to '{dataset_name}' successfully."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/datasets/stats", response_model=DatasetStatsResponse, summary="Get statistics of the current dataset")
async def get_current_dataset_stats():
    """
    Retrieves statistics (number of documents, queries, relevance judgments)
    for the currently active dataset.
    """
    try:
        stats = ir_engine.get_dataset_stats()
        return DatasetStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dataset stats: {str(e)}")

@app.get("/models", response_model=List[str], summary="Get available search model names")
async def get_available_models():
    """
    Retrieves a list of all available search model names (e.g., TF-IDF, BM25, Hybrid, Embedding).
    """
    return ir_engine.get_available_models()

@app.put("/models/current/{model_name}", summary="Change the current active search model")
async def change_current_model(model_name: str):
    """
    Sets the current active search model for subsequent search operations.
    """
    try:
        ir_engine.change_model(model_name)
        return {"message": f"Search model changed to '{model_name}' successfully."}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models/current", response_model=str, summary="Get the current active search model")
async def get_current_model():
    """
    Retrieves the name of the currently active search model.
    """
    return ir_engine.get_current_model()

@app.post("/search", response_model=SearchResponse, summary="Perform a search query")
async def perform_search(request: SearchRequest):
    """
    Performs a search using the currently selected model and dataset.

    - **query**: The search query string.
    - **top_k**: The number of top relevant documents to retrieve (default: 10).
    - **use_index**: Boolean indicating whether to use an underlying index/vector store (default: False).
                     This maps to the 'checkbox with vector store' and 'checkbox with cluster' if clustering
                     is integrated as an indexing strategy.
    """
    start_time = time.time()
    try:
        # Pass use_index to IREngine's search method
        results = ir_engine.search(request.query, request.top_k, with_index=request.use_index)

        # Ensure results are in the expected format (doc_id, score, snippet)
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

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

# To run this FastAPI application, you will use Uvicorn.
# 1. Open your terminal or command prompt.
# 2. Navigate to your project's root directory (where api_main.py is located).
# 3. Run the command: uvicorn api_main:app --reload --port 8000
#    --reload enables auto-reloading on code changes.
#    --port 8000 sets the server to run on port 8000 (you can change this).
#
# Once running, you can access the interactive API documentation (Swagger UI) at:
# http://127.0.0.1:8000/docs
# or ReDoc at:
# http://127.0.0.1:8000/redoc