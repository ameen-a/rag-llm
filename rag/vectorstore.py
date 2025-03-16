from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, persist_directory=None):
        """Initialise Chroma vector store for RAG"""
        # fix path handling to be relative to the project root
        if persist_directory is None:
            # use absolute path based on this file's location
            persist_directory = os.path.join(
                Path(__file__).resolve().parent.parent, "data/embeddings/chroma_db"
            )

        self.embedding_function = OpenAIEmbeddings()
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        if not os.path.exists(os.path.join(self.persist_directory, "chroma.sqlite3")):
            # create DB if it doesn't exist
            self.db = Chroma(
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory,
            )
            self.db.persist()
        else:
            # load DB if it already exists
            self.db = Chroma(
                embedding_function=self.embedding_function,
                persist_directory=self.persist_directory,
            )

    def add_documents(self, chunks):
        """Add document chunked embeedings to vector store"""
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        self.db.add_texts(texts=texts, metadatas=metadatas)
        self.db.persist()
        logger.info(f"Added {len(chunks)} docs to vector store")

    def load_from_chunks_file(self, chunks_path=None):
        """Load embedding chunks from JSON to the vector store"""
        if chunks_path is None:
            chunks_path = os.path.join(
                Path(__file__).resolve().parent.parent, "data/processed/chunks.json"
            )

        with open(chunks_path, "r") as f:
            chunks = json.load(f)

        self.add_documents(chunks)
        return len(chunks)

    def get_db(self):
        return self.db
