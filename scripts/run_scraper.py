#!/usr/bin/env python
import os
import sys
import json
import logging
import argparse
from pathlib import Path

# Add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.get_data import VoyZendeskAPI

def main():
    """Run the Zendesk API extractor and save the results."""
    parser = argparse.ArgumentParser(description="Extract FAQ data from Voy Zendesk Help Center")
    parser.add_argument(
        "--output", 
        default="../data/processed/articles.json",
        help="Output file path for processed articles"
    )
    parser.add_argument(
        "--raw-dir", 
        default="../data/raw",
        help="Directory to save raw article responses"
    )
    parser.add_argument(
        "--log-level", 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize the API extractor
    api = VoyZendeskAPI()
    
    # Run the extraction - using only the extract_all_articles method
    logger.info("Starting Zendesk API extraction")
    articles = api.extract_all_articles(save_raw=True, raw_dir=args.raw_dir)
    
    # Save processed articles
    logger.info(f"Saving {len(articles)} processed articles to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(articles, f, indent=2)
    
    logger.info("Extraction complete")

if __name__ == "__main__":
    main()