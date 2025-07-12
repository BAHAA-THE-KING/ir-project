# src/database/db_connector.py

import sqlite3
from typing import List, Tuple, Dict, Optional # تأكد من استيراد Optional

class DBConnector:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """Open a connection to the database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"Successfully connected to database: {self.db_path}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            self.conn = None

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print(f"Connection to database closed: {self.db_path}")
            self.conn = None

    def _execute_query(self, query: str, params: Tuple = ()) -> List[Tuple]:
        """Helper function to execute read queries."""
        if not self.conn:
            self.connect() 
            if not self.conn: 
                print("Database connection not established.")
                return []
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error executing query: {query} - {e}")
            return []

    def get_document_text_by_id(self, doc_id: str, dataset_name: str, cleaned: bool = False) -> Optional[str]:
        """
        Retrieve the text of a document (original or cleaned) based on document ID and dataset name.
        :param doc_id: Document ID.
        :param dataset_name: Dataset name (e.g., 'antique', 'quora').
        :param cleaned: If True, retrieve cleaned text; otherwise, retrieve original text.
        :return: Document text or None if not found.
        """
        table_name = f"{dataset_name}_cleaned" if cleaned else f"{dataset_name}_raw"
        query = f"SELECT text FROM {table_name} WHERE doc_id = ?"
        
        results = self._execute_query(query, (doc_id,))
        if results:
            return results[0][0]
        return None

    def get_all_document_ids_and_texts(self, dataset_name: str, cleaned: bool = False, sample_ratio: float = 1.0, limit: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        Retrieve all document IDs and texts from a table, with support for sampling.
        :param dataset_name: Dataset name.
        :param cleaned: If True, retrieve cleaned text; otherwise, retrieve original text.
        :param sample_ratio: Random sample ratio (between 0.0 and 1.0). 1.0 means all documents.
        :param limit: Maximum number of documents to fetch after sampling.
        :return: List of (doc_id, text).
        """
        table_name = f"{dataset_name}_cleaned" if cleaned else f"{dataset_name}_raw"
        
        # Get total number of rows for sampling
        count_query = f"SELECT COUNT(*) FROM {table_name}"
        total_rows = self._execute_query(count_query)[0][0]
        
        fetch_limit = total_rows # Default is all rows
        if sample_ratio < 1.0:
            fetch_limit = int(total_rows * sample_ratio)
        
        if limit is not None: # If explicit limit is set
            fetch_limit = min(fetch_limit, limit)
            
        # Use ORDER BY RANDOM() LIMIT to ensure random sampling
        query = f"SELECT doc_id, text FROM {table_name} ORDER BY RANDOM() LIMIT {fetch_limit}"
        
        print(f"Fetching {fetch_limit} documents (sample_ratio={sample_ratio}, limit={limit}) from {table_name}...")
        return self._execute_query(query)