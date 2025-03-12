import os
import sys
import json
import logging
from pathlib import Path

# add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.get_data import VoyZendeskAPI

# configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# output paths
output_file = "../data/processed/articles.json"
raw_dir = "../data/raw"

# create output directory
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# initialize API and extract articles
api = VoyZendeskAPI()
articles = api.extract_all_articles(save_raw=True, raw_dir=raw_dir)

# save articles into single JSON
with open(output_file, 'w') as f:
    json.dump(articles, f, indent=2)

print(f"Extraction complete - {len(articles)} articles saved")