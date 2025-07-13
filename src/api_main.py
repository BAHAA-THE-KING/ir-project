import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now import from src
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import time
import httpx
from services.processing.text_preprocessor import TextPreprocessor
from ir_engine import ir_engine, SearchModel
from config import DATASETS

from services.query_suggestion_service import QuerySuggestionService
from database.db_connector import DBConnector
from fastapi import Request
import logging

# Set up logging for timing
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Information Retrieval System API",
    description="API for searching documents using various IR models."
)

# --- Timing Middleware ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    import time
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logging.info(f"[TIMING] {request.method} {request.url.path} took {process_time:.4f} seconds")
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global variables for singleton instances
ir_engine_instance = None
preprocessor_instance = None
db_connector_instance = None
suggestion_services = {}

def initialize_services():
    """Initialize all services once at startup"""
    global ir_engine_instance, preprocessor_instance, db_connector_instance, suggestion_services
    
    logging.info("Initializing IR services...")
    
    # Initialize IR Engine
    try:
        ir_engine_instance = ir_engine()
        logging.info("IR Engine initialized")
    except Exception as e:
        logging.error(f"Failed to initialize IR Engine: {e}")
        logging.warning("IR Engine initialization failed - search features may be limited")
        ir_engine_instance = None
    
    # Initialize singleton instances
    preprocessor_instance = TextPreprocessor.getInstance()
    logging.info("TextPreprocessor initialized")
    
    # Initialize database connector
    try:
        db_path = "data/ir_project_data.db"  # Database is in the data directory
        db_connector_instance = DBConnector(db_path)
        db_connector_instance.connect()
        logging.info("Database connector initialized")
    except Exception as e:
        logging.error(f"Failed to initialize database connector: {e}")
        logging.warning("Database connector initialization failed - some features may be limited")
        db_connector_instance = None
    
    # Preload QuerySuggestionService for all datasets
    logging.info("Loading QuerySuggestionService instances...")
    for dataset in DATASETS.keys():
        try:
            if db_connector_instance is None:
                logging.warning(f"Skipping QuerySuggestionService for dataset '{dataset}' - database connector not available")
                continue
            start_time = time.time()
            suggestion_services[dataset] = QuerySuggestionService(dataset, preprocessor_instance, db_connector_instance)
            load_time = time.time() - start_time
            logging.info(f"Loaded QuerySuggestionService for dataset '{dataset}' in {load_time:.2f} seconds")
        except Exception as e:
            logging.error(f"Failed to load QuerySuggestionService for dataset '{dataset}': {e}")
            logging.warning(f"Skipping QuerySuggestionService for dataset '{dataset}' - suggestion features will be disabled")
            # Don't add to suggestion_services if it fails, but continue with other services
    
    logging.info(f"Successfully loaded {len(suggestion_services)} QuerySuggestionService instances")

def get_suggestion_service(dataset: str) -> QuerySuggestionService:
    """Get a suggestion service for the specified dataset"""
    if dataset not in suggestion_services:
        raise ValueError(f"Dataset '{dataset}' is not available. Available datasets: {list(suggestion_services.keys())}")
    return suggestion_services[dataset]

# Pydantic models for request and response validation
class SearchRequest(BaseModel):
    model: str # Specify model for this search
    dataset_name: str # Add dataset as a required string argument
    query: str
    top_k: int = 10
    use_inverted_index: bool = False # For TF-IDF/BM25
    use_vector_store: bool = False # For Embedding models

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

# ✅ /search
@app.post("/search", response_model=SearchResponse, summary="Perform a search query")
async def perform_search(request: SearchRequest):
    """
    Performs a search using the specified model and the currently loaded dataset.

    - **model**: The search model to use (e.g., "TF-IDF", "BM25", "Hybrid", "Embedding").
    - **query**: The search query string.
    - **dataset_name**: The dataset to use for the search.
    - **top_k**: The number of top relevant documents to retrieve (default: 10).
    - **use_inverted_index**: Boolean indicating whether to use an inverted index for TF-IDF/BM25 models (default: False).
    - **use_vector_store**: Boolean indicating whether to use a vector store for Embedding models (default: False).
    """
    start_time = time.time()
    try:
        # Skip preprocessing for TF-IDF model, use original query
        if request.model == "TF-IDF":
            query_for_search = request.query
            preprocess_time = 0.0
        else:
            # Call the /preprocess endpoint to get the preprocessed query
            preprocess_start = time.time()
            async with httpx.AsyncClient() as client:
                preprocess_response = await client.post(
                    "http://127.0.0.1:8000/preprocess",
                    json={"query": request.query}
                )
                preprocess_data = preprocess_response.json()
                preprocessed_query = preprocess_data["preprocessed"]
            preprocess_time = time.time() - preprocess_start
            # Use the preprocessed query for searching (join tokens back to string if needed)
            query_for_search = " ".join(preprocessed_query)
        
        search_start = time.time()
        if ir_engine_instance is None:
            raise HTTPException(status_code=500, detail="IR Engine not initialized")
        results = ir_engine_instance.search(
            model_name=request.model,
            dataset_name=request.dataset_name,
            query=query_for_search,
            top_k=request.top_k,
            use_inverted_index=request.use_inverted_index,
            use_vector_store=request.use_vector_store,
        )
        search_time = time.time() - search_start
        formatted_results = [
            SearchResultItem(doc_id=str(item[0]), score=float(item[1]), snippet=str(item[2]))
            for item in results
        ]
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"[TIMING] /search: preprocess={preprocess_time:.4f}s, search={search_time:.4f}s, total={total_time:.4f}s")
        return SearchResponse(
            query=request.query,
            time_taken=total_time,
            results=formatted_results
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# ✅ /document/{doc_id}
@app.get("/document/{doc_id}", response_model=str, summary="Get full text content of a document")
async def get_document_content(doc_id: str, dataset_name: str):
    """
    Retrieves the full text content of a document given its document ID and dataset name.
    """
    try:
        if db_connector_instance is None:
            raise HTTPException(status_code=500, detail="Database connector not initialized")
        
        # Try to get the document from the database
        document_text = db_connector_instance.get_document_text_by_id(doc_id, dataset_name, cleaned=False)
        if document_text is None:
            # Try cleaned version if raw not found
            document_text = db_connector_instance.get_document_text_by_id(doc_id, dataset_name, cleaned=True)
        
        if document_text is None:
            raise HTTPException(status_code=404, detail=f"Document with ID '{doc_id}' not found in dataset '{dataset_name}'.")
        
        return document_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

class SuggestRequest(BaseModel):
    query: str
    dataset: str
    top_k: int = 10

class SuggestionItem(BaseModel):
    suggestion: str
    snippet: str

class SuggestResponse(BaseModel):
    suggestions: List[SuggestionItem]

# --- Suggestion Endpoint ---
@app.post("/suggest", response_model=SuggestResponse, summary="Get query suggestions for autocomplete")
async def suggest_query(request: SuggestRequest):
    start_time = time.time()
    try:
        suggestion_service = get_suggestion_service(request.dataset)
        suggestions = suggestion_service.get_suggestions(request.query, top_k=request.top_k)
        time_taken = time.time() - start_time
        logging.info(f"[TIMING] /suggest: time_taken={time_taken:.4f}s")
        return SuggestResponse(suggestions=[SuggestionItem(suggestion=s[0], snippet=s[1]) for s in suggestions])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Suggestion failed: {str(e)}")

# --- Health Check Endpoint ---
@app.get("/health", summary="Check API health and loaded services")
async def health_check():
    """Check the health of the API and verify that all services are loaded"""
    try:
        health_status = {
            "status": "healthy",
            "services": {
                "ir_engine": ir_engine_instance is not None,
                "preprocessor": preprocessor_instance is not None,
                "db_connector": db_connector_instance is not None,
                "suggestion_services": {
                    dataset: service is not None 
                    for dataset, service in suggestion_services.items()
                }
            },
            "loaded_datasets": list(suggestion_services.keys()),
            "total_suggestion_services": len(suggestion_services)
        }
        
        # Check if all critical services are loaded
        all_services_loaded = all([
            ir_engine_instance is not None,
            preprocessor_instance is not None,
            db_connector_instance is not None,
            len(suggestion_services) > 0
        ])
        
        if not all_services_loaded:
            health_status["status"] = "degraded"
            health_status["message"] = "Some services failed to load"
        
        return health_status
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Startup event to initialize services
@app.on_event("startup")
async def startup_event():
    """Initialize all services when the API starts"""
    logging.info("Starting API initialization...")
    initialize_services()
    logging.info("API initialization complete")

@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources when the API shuts down"""
    if db_connector_instance:
        db_connector_instance.close()
        logging.info("Database connection closed")

# To run this FastAPI application:
# 1. Save the code as api_main.py in your project's root directory.
# 2. Make sure src/ir_engine.py has its 'search' method updated as described above.
# 3. Open your terminal in the project root and run: uvicorn api_main:app --reload --port 8000
# 4. Access interactive docs at: http://127.0.0.1:8000/docs