import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QComboBox, QPushButton, QTextEdit, 
                            QLabel, QSplitter, QListWidget, QListWidgetItem, QLineEdit)
from PyQt6.QtCore import Qt
from src.gui.ir_engine import IREngine

class IRMainWindow(QMainWindow):
    def __init__(self):
        print("Initializing IRMainWindow...")
        super().__init__()
        self.ir_engine = IREngine()
        self.setup_ui()
        print("IRMainWindow initialized successfully")
        
    def setup_ui(self):
        print("Setting up UI...")
        self.setWindowTitle("IR Project GUI")
        self.setMinimumSize(1200, 800)
        
        # Create the main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create dataset and model selection area
        selection_layout = QHBoxLayout()
        
        # Dataset selection
        dataset_label = QLabel("Select Dataset:")
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(self.ir_engine.get_available_datasets())
        self.dataset_combo.setCurrentText(self.ir_engine.current_dataset)
        load_button = QPushButton("Load Dataset")
        load_button.clicked.connect(self.load_dataset)
        
        # Model selection
        model_label = QLabel("Select Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.ir_engine.get_available_models())
        self.model_combo.setCurrentText(self.ir_engine.get_current_model())
        self.model_combo.currentTextChanged.connect(self.change_model)
        
        selection_layout.addWidget(dataset_label)
        selection_layout.addWidget(self.dataset_combo)
        selection_layout.addWidget(load_button)
        selection_layout.addStretch()
        selection_layout.addWidget(model_label)
        selection_layout.addWidget(self.model_combo)
        
        # Create search area
        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter your search query here...")
        
        # Add top_k input
        top_k_label = QLabel("Top K:")
        self.top_k_input = QLineEdit()
        self.top_k_input.setPlaceholderText("10")
        self.top_k_input.setFixedWidth(60)
        self.top_k_input.setText("10")  # Default value
        
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.perform_search)
        
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(top_k_label)
        search_layout.addWidget(self.top_k_input)
        search_layout.addWidget(search_button)
        
        # Create results area
        results_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel for search results
        self.results_list = QListWidget()
        self.results_list.itemClicked.connect(self.show_document)
        
        # Right panel for document content
        self.document_view = QTextEdit()
        self.document_view.setReadOnly(True)
        
        results_splitter.addWidget(self.results_list)
        results_splitter.addWidget(self.document_view)
        results_splitter.setSizes([400, 800])
        
        # Add all layouts to main layout
        layout.addLayout(selection_layout)
        layout.addLayout(search_layout)
        layout.addWidget(results_splitter)
        
        # Show initial dataset stats
        self.update_status()
        print("UI setup completed")
    
    def load_dataset(self):
        try:
            dataset_name = self.dataset_combo.currentText()
            print(f"Loading dataset: {dataset_name}")
            self.ir_engine.change_dataset(dataset_name)
            self.update_status()
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            self.statusBar().showMessage(f"Error loading dataset: {str(e)}")
    
    def update_status(self):
        stats = self.ir_engine.get_dataset_stats()
        status = (f"Dataset: {stats['name']} - {stats['description']} | "
                 f"Documents: {stats['num_docs']} | "
                 f"Queries: {stats['num_queries']} | "
                 f"Relevance judgments: {stats['num_qrels']}")
        self.statusBar().showMessage(status)
        print(f"Status updated: {status}")
    
    def perform_search(self):
        query = self.search_input.text()
        if not query:
            return
        
        try:
            # Get top_k value, default to 10 if invalid
            try:
                top_k = int(self.top_k_input.text())
                if top_k <= 0:
                    raise ValueError("Top K must be positive")
            except ValueError:
                top_k = 10
                self.top_k_input.setText("10")
                self.statusBar().showMessage("Invalid Top K value, using default (10)")
            
            print(f"Performing search for query: {query} with top_k: {top_k}")
            results = self.ir_engine.search(query, top_k)
            self.results_list.clear()
            
            for doc_id, score in results:
                item = QListWidgetItem(f"Doc_ID: {doc_id}, Score: {score:.2f}")
                item.setData(Qt.ItemDataRole.UserRole, doc_id)
                self.results_list.addItem(item)
                
        except Exception as e:
            print(f"Error performing search: {str(e)}")
            self.statusBar().showMessage(f"Error performing search: {str(e)}")
    
    def show_document(self, item):
        try:
            doc_id = item.data(Qt.ItemDataRole.UserRole)
            print(f"Showing document: {doc_id}")
            doc_content = self.ir_engine.get_document(doc_id)
            if doc_content:
                self.document_view.setText(doc_content)
            else:
                self.document_view.setText("Document not found")
        except Exception as e:
            print(f"Error displaying document: {str(e)}")
            self.statusBar().showMessage(f"Error displaying document: {str(e)}")

    def change_model(self, model_name: str):
        try:
            print(f"Changing model to: {model_name}")
            self.ir_engine.change_model(model_name)
            self.statusBar().showMessage(f"Changed search model to: {model_name}")
        except Exception as e:
            print(f"Error changing model: {str(e)}")
            self.statusBar().showMessage(f"Error changing model: {str(e)}")

if __name__ == "__main__":
    print("Starting application...")
    app = QApplication(sys.argv)
    window = IRMainWindow()
    window.show()
    print("Window shown, entering event loop...")
    sys.exit(app.exec()) 