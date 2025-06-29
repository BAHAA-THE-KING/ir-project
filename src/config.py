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
    },
    "webis": {
        "name": "beir/webis-touche2020/v2",
        "description": "Webis Touché 2020 (v2) dataset from the BEIR benchmark",
        "ir_datasets_id": "beir/webis-touche2020/v2",
        "ir_datasets_test_id": "beir/webis-touche2020/v2"
    }
}

# Default dataset to use if none specified
DEFAULT_DATASET = "antique"
