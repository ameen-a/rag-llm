# rag/llm.py

from langchain_openai import ChatOpenAI
from rag.prompts import get_qa_prompt

def generate_answer(query, contexts):
    """Generate an answer using retrieved contexts and LLM."""
    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini")  # Or another appropriate model
    
    # Construct prompt with contexts and user query
    prompt = get_qa_prompt(query, contexts)
    
    # Generate response
    response = llm.predict(prompt)
    
    return response