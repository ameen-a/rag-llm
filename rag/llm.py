from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from rag.retriever import Retriever
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class RAG:
    def __init__(self, model_name="gpt-4o", temperature=0):
        """
        Orchestrate LLM and retriever for RAG
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.retriever = Retriever()
    
    def _format_context(self, context_docs):
        """Format retrieved docs for prompt"""
        context_parts = []
        for i, doc in enumerate(context_docs):
            title = doc['metadata'].get('title', 'Untitled')
            content = doc['content']
            context_parts.append(f"Source: {title}\n{content}\n")
        
        return "\n".join(context_parts)
    
    def answer_question(self, query, k=3):
        """Answer query with RAG"""

        # get relevant docs
        results = self.retriever.retrieve(query, k=k)
        context_docs = self.retriever.format_retrieved_documents(results)
        
        # create prompt
        context_str = self._format_context(context_docs)
        system_message = SystemMessage(content="""
        You are a helpful RAG-based assistant that answers questions based on the provided context.
        If the answer cannot be determined from the context, acknowledge that and provide
        general information if possible. You must ALWAYS cite your sources with the document title. 
        """)
        human_message = HumanMessage(content=f"""
        Please answer the following question using the provided context:
        
        Question: {query}
        
        Context:
        {context_str}
        """)
        
        # query LLM
        response = self.llm.invoke([system_message, human_message])
        
        return {
            "question": query,
            "answer": response.content,
            "context": context_docs
        }

# if __name__ == "__main__":
    # try:
    #     from rag.retriever import Retriever
    # except ImportError:
    #     from retriever import Retriever
    

    # rag = RAG()
    
    # query = "using the documents provided, answer the following question: what kind of weight loss can I expect with GLP-1? give me A PERCENTAGE. Also tell me where you for this data from"
    # result = rag.answer_question(query)
    # # print(query)
    # # print(result)
    # # Print the result
    # logger.info(f"Question: {result['question']}")
    # logger.info(f"\nAnswer:")
    # logger.info(result['answer'])
    
    # # # # Optionally, print the context sources
    # # print("\nContext Sources:")
    # # for i, doc in enumerate(result['context']):
    # #     print(f"{i+1}. {doc['metadata']['title']} (Score: {doc['relevance_score']:.4f})")
