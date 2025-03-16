import json
import os
import logging
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


logger = logging.getLogger(__name__)

# get API keys
load_dotenv()


class Embeddings:
    def __init__(self, model_name="text-embedding-3-small"):

        self.embedding_model = OpenAIEmbeddings(model=model_name)

    def embed_text(self, text):
        """Embed a single text string"""

        return self.embedding_model.embed_query(text)

    def create_embeddings_for_chunks(
        self, chunks, output_path="../data/embeddings/chunks_with_embeddings.json"
    ):
        """Create and save embeddings for each chunk to JSON file"""

        logger.info(f"Creating embeddings for {len(chunks)} chunks")

        # create embedding column for each chunk
        for chunk in chunks:
            chunk["embedding"] = self.embed_text(chunk["text"])

        # save embeddings to JSON
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(chunks, f, indent=2)

        logger.info(f"Created and saved embeddings for {len(chunks)} chunks")
        return chunks


if __name__ == "__main__":
    # load chunks
    with open("../data/processed/chunks.json", "r") as f:
        chunks = json.load(f)
    logger.info(f"Loaded {len(chunks)} chunks")

    embeddings = Embeddings()
    embeddings.create_embeddings_for_chunks(chunks)
