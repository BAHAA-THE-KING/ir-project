# Dataset configurations
DATASETS = {
    "antique": {
        "name": "antique",
        "description": "Question-answer dataset with natural questions from real users",
        "ir_datasets_id": "antique",
        "ir_datasets_test_id": "antique/test"
    },
    "quora": {
        "name": "beir/quora",
        "description": "Quora question pairs dataset from the BEIR benchmark",
        "ir_datasets_id": "beir/quora",
        "ir_datasets_test_id": "beir/quora/test"
    }
}

# Default dataset to use if none specified
DEFAULT_DATASET = "antique"
