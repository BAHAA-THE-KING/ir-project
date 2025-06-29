# Information Retrieval System

A comprehensive information retrieval system with multiple vectorization methods including TF-IDF, BM25, and hybrid approaches.

## Features

- **Multiple Vectorization Methods**: TF-IDF, BM25, and Hybrid search
- **GUI Interface**: PyQt6-based graphical user interface
- **Dataset Support**: Integration with ir_datasets for various datasets
- **Text Processing**: Advanced preprocessing with NLTK
- **Model Persistence**: Save and load trained models

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Install from source

```bash
# Clone the repository
git clone <repository-url>
cd ir_project

# Install the package
pip install -e .
```

### Install dependencies only

```bash
pip install -r src/requirements.txt
```

## Usage

### Command Line

```bash
# Run the main application
python src/main.py

# Or if installed as a package
ir-project
```

### GUI Application

```bash
# Run the GUI application
python src/gui/gui.py
```

### Programmatic Usage

```python
from src.config import DEFAULT_DATASET
from src.loader import load_dataset_with_queries
from src.services.online_vectorizers.bm25 import bm25_search

# Load dataset
docs, queries, qrels = load_dataset_with_queries(DEFAULT_DATASET)

# Perform search
results = bm25_search(DEFAULT_DATASET, "your query here", top_k=10)
```

## Project Structure

```
src/
├── __init__.py              # Main package
├── main.py                  # Main application entry point
├── config.py               # Configuration settings
├── loader.py               # Dataset loading utilities
├── evaluation.py           # Evaluation metrics
├── gui/                    # GUI components
│   ├── __init__.py
│   ├── gui.py             # Main GUI application
│   └── ir_engine.py       # Search engine interface
└── services/              # Core services
    ├── __init__.py
    ├── online_vectorizers/  # Real-time search services
    │   ├── __init__.py
    │   ├── bm25.py
    │   ├── hybrid.py
    │   ├── tfidf.py
    │   ├── embedding.py
    │   └── inverted_index.py
    ├── offline_vectorizers/ # Model training services
    │   ├── __init__.py
    │   ├── bm25.py
    │   ├── hybrid.py
    │   ├── tfidf.py
    │   └── embedding.py
    └── processing/         # Text processing utilities
        ├── __init__.py
        ├── preprocessing.py
        └── docs_processor.py
```

## Available Datasets

The system supports various datasets through ir_datasets:

- **antique**: Question-answer dataset with natural questions from real users
- **beir/quora**: Quora question pairs dataset from the BEIR benchmark

## Search Models

### BM25
Best Matching 25 (BM25) is a ranking function used by search engines to rank matching documents according to their relevance to a given search query.

### TF-IDF
Term Frequency-Inverse Document Frequency is a numerical statistic that reflects how important a word is to a document in a collection.

### Embedding (BERT)

The system supports dense retrieval using BERT-based embeddings. Documents and queries are encoded into dense vectors using a pre-trained BERT model, and similarity is computed (typically via cosine similarity) to rank results.

### Hybrid
Combines multiple ranking methods for improved search results.

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Lint code
flake8 src/
```

### Adding New Vectorizers

1. Create a new file in `src/services/offline_vectorizers/` for training
2. Create a new file in `src/services/online_vectorizers/` for search
3. Update the respective `__init__.py` files
4. Add the new model to `SearchModel` enum in `src/gui/ir_engine.py`

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- [ir_datasets](https://github.com/allenai/ir_datasets) for dataset access
- [rank_bm25](https://github.com/dorianbrown/rank_bm25) for BM25 implementation
- [NLTK](https://www.nltk.org/) for text processing
