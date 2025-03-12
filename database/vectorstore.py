# vectorstore.py
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
import json

class VectorStore:
    def __init__(self, persist_directory="../data/embeddings/chroma_db"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the Chroma database
        """
        self.embedding_function = OpenAIEmbeddings()
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize or load the database
        if not os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3")):
            self.db = Chroma(
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
            self.db.persist()
        else:
            # Load existing database
            self.db = Chroma(
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory
            )
    
    def add_documents(self, chunks):
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with text and metadata
        """
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Add documents to the database
        self.db.add_texts(texts=texts, metadatas=metadatas)
        self.db.persist()
        print(f"Added {len(chunks)} documents to vector store")
    
    def load_from_chunks_file(self, chunks_path="../data/processed/chunks.json"):
        """
        Load chunks from a JSON file and add them to the vector store.
        
        Args:
            chunks_path: Path to the chunks JSON file
        """
        with open(chunks_path, 'r') as f:
            chunks = json.load(f)
        
        self.add_documents(chunks)
        return len(chunks)
    
    def get_db(self):
        """
        Get the Chroma database instance.
        
        Returns:
            Chroma database instance
        """
        return self.db

if __name__ == "__main__":
    # Initialize the vector store
    vector_store = VectorStore()
    
    # Load chunks and add them to the vector store
    vector_store.load_from_chunks_file()
    print("Vector store created and populated successfully")