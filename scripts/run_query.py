import os
import sys
import json
import logging
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.llm import RAG
from rag.constants import MODEL_NAME, TEMPERATURE, K

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

TEST_QUERY = "What payment options are available?"


def process_query(query):
    """Run the user's prompt"""
    logger.info(f"Initialising rag system with model: {MODEL_NAME}")
    rag = RAG(model_name=MODEL_NAME, temperature=TEMPERATURE)

    # process query
    result = rag.answer_question(query, k=K)

    # main results
    logger.info("=" * 80)
    logger.info(f"QUESTION: {result['question']}")
    logger.info("=" * 80)
    logger.info(f"ANSWER:")
    logger.info(result["answer"])
    logger.info("=" * 80)

    # show retrieved documents
    logger.info("CONTEXT SOURCES:")
    for i, doc in enumerate(result["context"]):
        logger.info(
            f"{i+1}. {doc['metadata'].get('title', 'Untitled')} (Score: {doc['relevance_score']:.4f})"
        )
    logger.info("=" * 80)

    logger.info(f"Query processing complete")
    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a query through the rag system")
    parser.add_argument(
        "--query", "-q", type=str, help="Query to process", default=TEST_QUERY
    )
    args = parser.parse_args()

    query = args.query
    process_query(query)
