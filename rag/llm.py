from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from retriever import Retriever
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class RAG:
    def __init__(self, model_name="gpt-4o", temperature=0):
        """
        initialize the rag system.
        
        args:
            model_name: name of the llm model to use
            temperature: temperature parameter for the llm
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.retriever = Retriever()
    
    def _format_context(self, context_docs):
        """
        format retrieved documents into a context string.
        
        args:
            context_docs: list of retrieved document dictionaries
            
        returns:
            formatted context string
        """
        context_parts = []
        for i, doc in enumerate(context_docs):
            title = doc['metadata'].get('title', 'Untitled')
            content = doc['content']
            context_parts.append(f"Source: {title}\n{content}\n")
        
        return "\n".join(context_parts)
    


    def answer_question(self, query, k=3):
        """
        answer a question using retrieval-augmented generation.
        
        args:
            query: user question
            k: number of documents to retrieve
            
        returns:
            dictionary with question, answer, and context
        """
        # retrieve relevant documents
        results = self.retriever.retrieve(query, k=k)
        context_docs = self.retriever.format_retrieved_documents(results)
        
        # format context
        context_str = self._format_context(context_docs)
        
        # create messages for the llm
        system_message = SystemMessage(content="""
        you are a helpful assistant that answers questions based on the provided context.
        if the answer cannot be determined from the context, acknowledge that and provide
        general information if possible. always cite your sources when appropriate.
        """)
        
        human_message = HumanMessage(content=f"""
        please answer the following question using the provided context:
        
        question: {query}
        
        context:
        {context_str}
        """)
        
        # get response from llm
        response = self.llm.invoke([system_message, human_message])
        
        return {
            "question": query,
            "answer": response.content,
            "context": context_docs
        }

if __name__ == "__main__":
    # Initialize the RAG system
    rag = RAG()
    
    # Ask a question
    query = "using the documents provided, answer the following question: what kind of weight loss can I expect with GLP-1? give me A PERCENTAGE. Also tell me where you for this data from"
    result = rag.answer_question(query)
    
    # Print the result
    # print(f"Question: {result['question']}")
    print("\nAnswer:")
    print(result['answer'])
    
    # # Optionally, print the context sources
    # print("\nContext Sources:")
    # for i, doc in enumerate(result['context']):
    #     print(f"{i+1}. {doc['metadata']['title']} (Score: {doc['relevance_score']:.4f})")
