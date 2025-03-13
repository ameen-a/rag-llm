import os
import sys
import json
import logging
from pathlib import Path


# add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.llm import RAG
from rag.constants import MODEL_NAME, TEMPERATURE, K

# configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# initialize the RAG system
logger.info(f"initializing rag system with model: {MODEL_NAME}")
rag = RAG(model_name=MODEL_NAME, temperature=TEMPERATURE)

# get query from user
query = "using the documents provided, answer the following question: what kind of weight loss can I expect with GLP-1? give me A PERCENTAGE. Also, you MUST tell me where you got this data from"

# process the query
result = rag.answer_question(query, k=K)

# display results
logger.info("="*80)
logger.info(f"QUESTION: {result['question']}")
logger.info("="*80)
logger.info(f"ANSWER:")
logger.info(result['answer'])
logger.info("="*80)
# display context sources
logger.info("CONTEXT SOURCES:")
for i, doc in enumerate(result['context']):
    logger.info(f"{i+1}. {doc['metadata'].get('title', 'Untitled')} (Score: {doc['relevance_score']:.4f})")
logger.info("="*80)

logger.info(f"Query processing complete")