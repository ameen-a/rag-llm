from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os
import uuid
import logging

logger = logging.getLogger(__name__)

class DocumentChunker:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """Split documents into chunks for better retrieval"""

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def chunk_documents(self, documents):
        """Create chunk objects for each document"""

        all_chunks = []
        
        for doc in documents:
            doc_id = doc.get('id', str(uuid.uuid4()))
            title = doc.get('title', '')
            body = doc.get('body', '')
            url = doc.get('url', '')
            
            # combine title and body for improved document context
            full_text = f"TITLE: {title}\n\n{body}"
            
            # use LangChain's splitter to define chunks
            chunks = self.text_splitter.split_text(full_text)
            
            # create a chunk object for each chunk
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
    
    def save_chunks(self, chunks, output_path="../data/processed/chunks.json"):
        """Save chunks to JSON"""

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(chunks, f, indent=2)
        
        logger.info(f"saved {len(chunks)} chunks to {output_path}")
        return output_path

if __name__ == "__main__":

    # load the documents
    with open("../data/processed/articles.json", 'r') as f:
        documents = json.load(f)
    
    # create chunks
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)
    
    # save chunks
    chunker.save_chunks(chunks)