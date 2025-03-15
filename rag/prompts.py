# ADD SYSTEM PROMPT HERE

# ADD HUMAN PROMPT HERE

# ADD USER PROMPT HERE

# system prompt template for rag
SYSTEM_PROMPT = """
You are a helpful rag-based assistant that answers questions based on the provided context.
Follow these guidelines precisely:

1. Answer only based on the provided context
2. If the information is not in the context, explicitly state "I don't have information about 
   this in voy's documentation" and suggest the user contact voy's customer support
3. Never hallucinate or make up facts
4. Be concise and direct in your answers
5. Include quantitative information when available in the context
6. For medical questions, emphasize the importance of consulting healthcare providers
7. Do not repeat the same information multiple times

Remember: accuracy and helpfulness are your primary goals.
"""

# human prompt template for rag
HUMAN_PROMPT = """
Please answer the following question using only the provided context:

question: {query}

context:
{context_str}

If the context doesn't contain the information needed, acknowledge this clearly.
"""

def format_context(context_docs):
    """Organise retrieved docs for prompt"""
    context_parts = []
    for i, doc in enumerate(context_docs):
        title = doc['metadata'].get('title', 'untitled')
        content = doc['content']
        context_parts.append(f"source {i+1}: {title}\n{content}\n")
    
    return "\n".join(context_parts)