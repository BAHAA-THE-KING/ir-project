# Dataset configurations
DATASETS = {
    "antique": {
        "name": "antique",
        "description": "Question-answer dataset with natural questions from real users",
        "ir_datasets_id": "antique",
        "ir_datasets_test_id": "antique/test"
    },
    "wikir/en1k": {
        "name": "wikir/en1k",
        "description": "Wikipedia-based information retrieval dataset with 1K articles",
        "ir_datasets_id": "wikir/en1k",
        "ir_datasets_test_id": "wikir/en1k/test"
    }
}

# Default dataset to use if none specified
DEFAULT_DATASET = "antique"
