import sys
import os


project_root = "./" 
src_path = os.path.join(project_root, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)


from services.query_suggestion_service import QuerySuggestionService
from services.processing.text_preprocessor import TextPreprocessor
from services.online_vectorizers.embedding import Embedding_online 
from config import DATASETS 
from database.db_connector import DBConnector


def test_suggestions(dataset_name, preprocessor, db_connector):
    print(f"\n--- Testing Query Suggestions for dataset: {dataset_name} ---")


    suggestion_service = QuerySuggestionService(dataset_name, preprocessor, db_connector)

    while True:
        user_input = input(f"Enter a query prefix for {dataset_name} (type 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            break

        if user_input:
            suggestions_with_snippets = suggestion_service.get_suggestions(user_input, top_k=5)
            if suggestions_with_snippets:
                print("Suggestions:")
                for i, (suggestion_text, snippet_text) in enumerate(suggestions_with_snippets):
                    print(f"  {i+1}. {suggestion_text}")
                    if snippet_text:
                        print(f"     Snippet: \"{snippet_text}\"")
            else:
                print("No suggestions found.")
        else:
            print("Please enter something to get suggestions.")

def main():
    # تهيئة المعالج المسبق
    text_preprocessor_instance = TextPreprocessor.getInstance()
    
    db_path = "./ir_project_data.db" 
    db_connector_instance = DBConnector(db_path)
    db_connector_instance.connect() 


    test_suggestions("antique", text_preprocessor_instance, db_connector_instance)


    test_suggestions("quora", text_preprocessor_instance, db_connector_instance) 

  
    db_connector_instance.close()

    print("\nTesting complete.")

if __name__ == "__main__":
    main()