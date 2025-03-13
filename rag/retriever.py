from langchain_openai import OpenAIEmbeddings
from vectorstore import VectorStore
import logging

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, vector_store=None, persist_directory="../data/embeddings/chroma_db"):
        """
        Initialize the retriever.
        
        Args:
            vector_store: VectorStore instance or None to create a new one
            persist_directory: Directory where the Chroma database is stored
        """
        if vector_store is None:
            self.vector_store = VectorStore(persist_directory)
        else:
            self.vector_store = vector_store
            
        self.db = self.vector_store.get_db()
        self.embedding_function = OpenAIEmbeddings()
    
    def retrieve(self, query, k=3):
        """
        Retrieve documents similar to the query.
        
        Args:
            query: Query text
            k: Number of documents to return
            
        Returns:
            List of retrieved documents with their relevance scores
        """
        results = self.db.similarity_search_with_relevance_scores(query, k=k)
        return results
    
    def retrieve_with_filter(self, query, filter_dict, k=3):
        """
        Retrieve documents similar to the query with metadata filtering.
        
        Args:
            query: Query text
            filter_dict: Dictionary of metadata filters
            k: Number of documents to return
            
        Returns:
            List of retrieved documents with their relevance scores
        """
        results = self.db.similarity_search_with_relevance_scores(
            query, 
            k=k,
            filter=filter_dict
        )
        return results
    
    def format_retrieved_documents(self, results):
        """
        Format retrieved documents into a more readable structure.
        
        Args:
            results: List of (document, score) tuples from retrieval
            
        Returns:
            List of dictionaries with document content and metadata
        """
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'relevance_score': score
            })
        return formatted_results

if __name__ == "__main__":
    
    # initialize retriever with existing vector store
    retriever = Retriever()
    
    # test retrieval query
    query = "what kind of weight loss can I expect with GLP-1?"
    results = retriever.retrieve(query, k=3)
    
    # Print the results
    logger.info(f"\nQuery: {query}")
    formatted_results = retriever.format_retrieved_documents(results)
    
    for i, result in enumerate(formatted_results):
        logger.info(f"\nResult {i+1} (Score: {result['relevance_score']:.4f})")
        logger.info(f"Title: {result['metadata']['title']}")
        logger.info(f"Content: {result['content']}...")
        logger.info("-" * 80)