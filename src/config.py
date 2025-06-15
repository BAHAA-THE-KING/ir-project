# Dataset configurations
DATASETS = {
    "antique": {
        "name": "antique",
        "description": "Question-answer dataset with natural questions from real users",
        "ir_datasets_id": "antique"
    },
    "wikir": {
        "name": "wikir",
        "description": "Wikipedia-based information retrieval dataset with 1K articles",
        "ir_datasets_id": "wikir/en1k"
    }
}

# Default dataset to use if none specified
DEFAULT_DATASET = "antique"
