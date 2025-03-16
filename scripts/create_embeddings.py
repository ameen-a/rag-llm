import os
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.embeddings import Embeddings
from database.chunking import DocumentChunker
from rag.vectorstore import VectorStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def create_embeddings():
    """Create chunks and embeddings from articles"""

    # paths
    articles_file = os.path.join(
        Path(__file__).resolve().parent.parent, "data/processed/articles.json"
    )
    chunks_file = os.path.join(
        Path(__file__).resolve().parent.parent, "data/processed/chunks.json"
    )
    output_file = os.path.join(
        Path(__file__).resolve().parent.parent,
        "data/embeddings/chunks_with_embeddings.json",
    )

    # create output directories
    os.makedirs(os.path.dirname(chunks_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # step 1: create chunks from articles
    logger.info(f"Loading articles from {articles_file}")
    try:
        with open(articles_file, "r") as f:
            documents = json.load(f)
        logger.info(f"Loaded {len(documents)} articles")

        # create chunks
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(documents)

        # save chunks
        chunker.save_chunks(chunks, output_path=chunks_file)

        logger.info(f"Chunking process complete - {len(chunks)} chunks created")
    except FileNotFoundError:
        logger.error("Please run extract_data.py first to generate the articles file")
        sys.exit(1)

    # step 2: create embeddings for chunks
    logger.info(f"Loading chunks from {chunks_file}")
    with open(chunks_file, "r") as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")

    # initialise embeddings class and create embeddings
    embeddings = Embeddings()
    chunks_with_embeddings = embeddings.create_embeddings_for_chunks(
        chunks, output_path=output_file
    )

    vector_store = VectorStore()
    vector_store.load_from_chunks_file(chunks_file)
    logger.info("Vector store populated with chunks")

    logger.info(
        f"Embedding process complete - {len(chunks_with_embeddings)} chunks processed"
    )

    return chunks_with_embeddings


if __name__ == "__main__":
    create_embeddings()
