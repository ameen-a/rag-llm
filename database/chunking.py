from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os
import uuid

# MAKE THIS A CLASS
def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into chunks for more effective retrieval.
    
    Args:
        documents: List of document dictionaries with 'title' and 'body' fields
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    all_chunks = []
    
    for doc in documents:
        doc_id = doc.get('id', str(uuid.uuid4()))
        title = doc.get('title', '')
        body = doc.get('body', '')
        url = doc.get('url', '')
        
        # Combine title and body for context, but mark the title specially
        full_text = f"TITLE: {title}\n\n{body}"
        
        # Split the text into chunks
        chunks = text_splitter.split_text(full_text)
        
        # Create a chunk object for each text chunk
        for i, chunk_text in enumerate(chunks):
            chunk = {
                'text': chunk_text,
                'metadata': {
                    'doc_id': doc_id,
                    'title': title,
                    'url': url,
                    'chunk_index': i,
                    'chunk_count': len(chunks)
                }
            }
            all_chunks.append(chunk)
    
    return all_chunks

def save_chunks(chunks, output_path="../data/processed/chunks.json"):
    """Save the chunks to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(chunks, f, indent=2)
    
    print(f"Saved {len(chunks)} chunks to {output_path}")
    return output_path

if __name__ == "__main__":
    # Load the documents
    with open("../data/processed/articles.json", 'r') as f:
        documents = json.load(f)
    
    # Chunk the documents
    chunks = chunk_documents(documents)


    
    # Save the chunks
    save_chunks(chunks)