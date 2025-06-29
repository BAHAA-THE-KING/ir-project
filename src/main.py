from services.online_vectorizers.embedding import embedding_search
from loader import load_queries_and_qrels
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))



def main():
    queries, qrels = load_queries_and_qrels('antique')
    embedding_search('antique' , queries[57].text)
if __name__ == "__main__":
    main()

