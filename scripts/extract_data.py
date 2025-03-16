import os
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.get_data import VoyZendeskAPI

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# make output directory
output_file = "../data/processed/articles.json"
raw_dir = "../data/raw"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# extract articles
api = VoyZendeskAPI()
articles = api.extract_all_articles(save_raw=True, raw_dir=raw_dir)

# save articles
with open(output_file, "w") as f:
    json.dump(articles, f, indent=2)

logger.info(f"Extraction complete - {len(articles)} articles saved")
