from langchain_openai import OpenAIEmbeddings
from rag.vectorstore import VectorStore
import logging

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, vector_store=None, persist_directory=None):
        """Initialise the retriever with a vector store"""
        if vector_store is None:
            self.vector_store = VectorStore(persist_directory)
        else:
            self.vector_store = vector_store

        self.db = self.vector_store.get_db()
        self.embedding_function = OpenAIEmbeddings()

    def retrieve(self, query, k=3):
        """Get similar documents using LangChain's retriever"""
        results = self.db.similarity_search_with_relevance_scores(query, k=k)
        return results

    def format_retrieved_documents(self, results):
        """Format retrieved documents into a more readable structure"""
        formatted_results = []
        for doc, score in results:
            formatted_results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": score,
                }
            )
        return formatted_results
