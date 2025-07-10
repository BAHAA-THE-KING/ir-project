import sys
import os

# أضف مسار المشروع إلى Python path إذا لم يكن مضافاً لتتمكن من استيراد الوحدات
# تأكد أن المسار صحيح بناءً على هيكل مشروعك
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from services.query_suggestion_service import QuerySuggestionService
from services.processing.text_preprocessor import TextPreprocessor

def test_suggestions(dataset_name, preprocessor):
    print(f"\n--- Testing Query Suggestions for dataset: {dataset_name} ---")
    
    # تهيئة خدمة اقتراح الكويري للمجموعة المختارة
    # سيقوم بتحميل الفهرس المحفوظ أو بناءه إذا لم يكن موجوداً
    suggestion_service = QuerySuggestionService(dataset_name, preprocessor)

    while True:
        user_input = input(f"Enter a query prefix for {dataset_name} (type 'exit' to quit): ").strip()
        if user_input.lower() == 'exit':
            break
        
        if user_input:
            suggestions = suggestion_service.get_suggestions(user_input, top_k=5)
            if suggestions:
                print("Suggestions:")
                for i, suggestion in enumerate(suggestions):
                    print(f"  {i+1}. {suggestion}")
            else:
                print("No suggestions found.")
        else:
            print("Please enter something to get suggestions.")

def main():
    # تهيئة المعالج المسبق
    text_preprocessor_instance = TextPreprocessor()

    # اختبر لمجموعة البيانات "antique"
    test_suggestions("antique", text_preprocessor_instance)

    # اختبر لمجموعة البيانات "quora"
    # test_suggestions("quora", text_preprocessor_instance)

    print("\nTesting complete.")

if __name__ == "__main__":
    main()