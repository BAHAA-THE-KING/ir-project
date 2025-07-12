import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import from src
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import time
import httpx
from src.services.processing.text_preprocessor import TextPreprocessor
from src.gui.ir_engine import IREngine, SearchModel
from src.config import DATASETS
from src.services.query_suggestion_service import QuerySuggestionService
from src.database.db_connector import DBConnector
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

# Initialize the IR Engine globally to maintain state across requests
ir_engine = IREngine()

# Singleton instances
preprocessor = TextPreprocessor.getInstance()
db_path = "./ir_project_data.db"
db_connector = DBConnector(db_path)
db_connector.connect()

# Preload QuerySuggestionService for all datasets
suggestion_services = {}
for dataset in DATASETS.keys():
    try:
        suggestion_services[dataset] = QuerySuggestionService(dataset, preprocessor, db_connector)
        print(f"Loaded QuerySuggestionService for dataset: {dataset}")
    except Exception as e:
        print(f"Failed to load QuerySuggestionService for dataset {dataset}: {e}")

def get_suggestion_service(dataset):
    if dataset not in suggestion_services:
        raise ValueError(f"Dataset '{dataset}' is not available or failed to load.")
    return suggestion_services[dataset]

# Pydantic models for request and response validation
class SearchRequest(BaseModel):
    model: str # Specify model for this search
    query: str
    dataset: str # Add dataset as a required string argument
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
    """
    start_time = time.time()
    try:
        # Switch to the requested dataset before searching
        ir_engine.change_dataset(request.dataset)
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
        results = ir_engine.search(
            model_name=request.model,
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Suggestion failed: {str(e)}")

@app.on_event("shutdown")
def shutdown_event():
    db_connector.close()

# To run this FastAPI application:
# 1. Save the code as api_main.py in your project's root directory.
# 2. Make sure src/gui/ir_engine.py has its 'search' method updated as described above.
# 3. Open your terminal in the project root and run: uvicorn api_main:app --reload --port 8000
# 4. Access interactive docs at: http://127.0.0.1:8000/docs