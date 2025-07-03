import os
import dill
from rank_bm25 import BM25Okapi
from services.processing.text_preprocessor import TextPreprocessor

class BM25_offline:
    def bm25_train(self, docs, dataset_name):
        # Train the model
        bm25 = BM25Okapi([doc.text for doc in docs], tokenizer = TextPreprocessor.getInstance().preprocess_text)

        # Save the model and the documents
        path = f"data/{dataset_name}/bm25_model.dill"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'a').close()
        with open(path, "wb") as f:
            dill.dump(bm25, f)

        print("✅ Model trained and saved")