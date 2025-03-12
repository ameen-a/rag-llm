import json
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# get API keys
load_dotenv()

def create_embeddings_for_chunks(chunks, output_path="../data/embeddings/chunks_with_embeddings.json"):
    """Generate embeddings for each chunk and save them."""
    # Initialize the embedding model
    embedding_model = OpenAIEmbeddings()  # Or another embedding model
    
    # Create embeddings for each chunk
    for chunk in chunks:
        chunk['embedding'] = embedding_model.embed_query(chunk['text'])
    
    # Save the chunks with embeddings
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(chunks, f, indent=2)
    
    print(f"Created and saved embeddings for {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":
    # Load the chunks
    with open("../data/processed/chunks.json", 'r') as f:
        chunks = json.load(f)
    
    # Create embeddings
    create_embeddings_for_chunks(chunks)