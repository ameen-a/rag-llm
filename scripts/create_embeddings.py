import os
import sys
import json
import logging
from pathlib import Path

# add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.embeddings import Embeddings

# configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# input and output paths
input_file = "../data/processed/chunks.json"
output_file = "../data/embeddings/chunks_with_embeddings.json"

# create output directory
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# load chunks
logger.info(f"Loading chunks from {input_file}")
with open(input_file, 'r') as f:
    chunks = json.load(f)
logger.info(f"Loaded {len(chunks)} chunks")

# initialize embeddings class and create embeddings
embeddings = Embeddings()
chunks_with_embeddings = embeddings.create_embeddings_for_chunks(chunks, output_path=output_file)

logger.info(f"embedding process complete - {len(chunks_with_embeddings)} chunks processed")