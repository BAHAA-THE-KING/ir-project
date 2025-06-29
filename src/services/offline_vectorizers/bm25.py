import os
import dill
from rank_bm25 import BM25Okapi
from services.processing.bm25_preprocessing import AntiqueTextProcessor, QuoraTextProcessor, WebisTextProcessor, RecreationTextProcessor, WikirTextProcessor

class BM25_offline:
    @staticmethod
    def bm25_train(docs, dataset_name):
        if dataset_name == "antique":
            preprocess_text = AntiqueTextProcessor.preprocess_text
        elif dataset_name == 'quora':
            preprocess_text = QuoraTextProcessor.preprocess_text
        elif dataset_name == "webis":
            preprocess_text = WebisTextProcessor.preprocess_text
        elif dataset_name == "recreation":
            preprocess_text = RecreationTextProcessor.preprocess_text
        elif dataset_name == "wikir":
            preprocess_text = WikirTextProcessor.preprocess_text
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        # Train the model
        bm25 = BM25Okapi([doc.text for doc in docs], tokenizer = preprocess_text)

        # Save the model and the documents
        path = f"data/{dataset_name}/bm25_model.dill"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'a').close()
        with open(path, "wb") as f:
            dill.dump(bm25, f)

        print("✅ Model trained and saved")