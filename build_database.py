import sqlite3
import os
import sys

# Dynamically adjust sys.path to allow imports from the 'src' directory
# This makes the script runnable from anywhere within the project structure,
# assuming 'src' is a direct subdirectory of the project root.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up until 'src' directory is found or root is reached
project_root = current_dir
while "src" not in os.listdir(project_root) and project_root != os.path.dirname(project_root):
    project_root = os.path.dirname(project_root)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if os.path.join(project_root, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(project_root, 'src'))

# Import necessary modules from your project
from src.loader import load_dataset # 
from src.services.processing.text_preprocessor import TextPreprocessor # [cite: 31, 32]

# Define the database file name
DATABASE_NAME = "ir_project_data.db"

def create_tables(conn):
    """
    Creates the necessary tables in the SQLite database if they don't already exist.
    Each table will store documents with 'id', 'doc_id', and 'text' columns.
    """
    cursor = conn.cursor()
    tables = {
        "antique_raw": "CREATE TABLE IF NOT EXISTS antique_raw (id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, text TEXT)",
        "antique_cleaned": "CREATE TABLE IF NOT EXISTS antique_cleaned (id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, text TEXT)",
        "quora_raw": "CREATE TABLE IF NOT EXISTS quora_raw (id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, text TEXT)",
        "quora_cleaned": "CREATE TABLE IF NOT EXISTS quora_cleaned (id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT, text TEXT)"
    }
    for table_name, create_sql in tables.items():
        cursor.execute(create_sql)
    conn.commit()
    print("✅ Tables created or already exist.")

def insert_documents(conn, table_name, docs, preprocessor=None):
    """
    Inserts documents into the specified table.
    If a preprocessor is provided, the text will be cleaned before insertion.
    Documents are inserted sequentially to ensure 'id' corresponds to original order.
    """
    cursor = conn.cursor()
    print(f"Inserting documents into {table_name}...")
    
    # Iterate through documents and insert them. The AUTOINCREMENT will handle the 'id'.
    # By inserting in the order they are loaded, the 'id' in the database
    # will be (original_list_index + 1). Thus, original_list_index = id - 1.
    for doc in docs: # 
        text_to_insert = doc.text # 
        if preprocessor:
            # Preprocess the text: TextPreprocessor.preprocess_text returns a list of tokens,
            # so join them back into a string for storage. 
            processed_tokens = preprocessor.preprocess_text(doc.text) # 
            text_to_insert = " ".join(processed_tokens)

        cursor.execute(f"INSERT INTO {table_name} (doc_id, text) VALUES (?, ?)",
                       (doc.doc_id, text_to_insert)) # 
    conn.commit()
    print(f"✅ {len(docs)} documents inserted into {table_name}.")

def build_database():
    """
    Connects to the database, creates tables, and populates them with raw
    and preprocessed data from 'antique' and 'quora' datasets.
    """
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        create_tables(conn)

        # Initialize the text preprocessor
        text_preprocessor = TextPreprocessor.getInstance() # 

        datasets_to_process = ["antique", "quora"] # [cite: 74, 30]

        for dataset_name in datasets_to_process:
            print(f"\n--- Processing dataset: {dataset_name} ---")
            
            # Load raw documents for the current dataset
            docs = load_dataset(dataset_name) # 
            
            # Insert raw documents into the corresponding table
            insert_documents(conn, f"{dataset_name}_raw", docs)
            
            # Insert cleaned (preprocessed) documents into the corresponding table
            insert_documents(conn, f"{dataset_name}_cleaned", docs, preprocessor=text_preprocessor)

    except sqlite3.Error as e:
        print(f"❌ SQLite error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("\n🎉 Database building complete!")
            print(f"Database saved as: {DATABASE_NAME}")

if __name__ == "__main__":
    build_database()