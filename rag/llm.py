from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from rag.retriever import Retriever
from rag.prompts import SYSTEM_PROMPT, HUMAN_PROMPT, format_context
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
        return format_context(context_docs)
    
    def answer_question(self, query, k=3):
        """Answer query with RAG"""

        # get relevant docs
        results = self.retriever.retrieve(query, k=k)
        context_docs = self.retriever.format_retrieved_documents(results)
        
        # create prompt
        context_str = self._format_context(context_docs)
        system_message = SystemMessage(content=SYSTEM_PROMPT)
        human_message = HumanMessage(content=HUMAN_PROMPT.format(
            query=query,
            context_str=context_str
        ))
        
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
