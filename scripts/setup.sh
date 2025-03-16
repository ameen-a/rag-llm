#!/bin/bash

# create and activate virtual environment
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/embeddings/chroma_db

# dependencies
echo "Installing python dependencies..."
pip install -r requirements.txt

# create .env file if it doesn't exist
if [ ! -f .env ]; then
  echo "Creating .env file..."
  echo "OPENAI_API_KEY=" > .env
  echo "Add your OpenAI API key to the .env file"
fi

# extract data and create embeddings
echo "Extracting data..."
python scripts/extract_data.py

echo "Creating embeddings..."
python scripts/create_embeddings.py

echo "Setup complete. Run 'source venv/bin/activate && python scripts/run_app.py' to start the application, or 'source venv/bin/activate && python scripts/run_evals.py' for evals"
