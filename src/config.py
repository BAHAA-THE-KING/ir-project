# Dataset configurations
DATASETS = {
    "antique": {
        "name": "antique",
        "description": "Question-answer dataset with natural questions from real users",
        "ir_datasets_id": "antique"
    },
    "msmarco": {
        "name": "msmarco",
        "description": "Large-scale passage ranking dataset from Bing queries",
        "ir_datasets_id": "beir/msmarco"
    }
}

# Default dataset to use if none specified
DEFAULT_DATASET = "antique"
