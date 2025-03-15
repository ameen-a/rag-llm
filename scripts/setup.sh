#!/bin/bash
# setup script for rag application

# create and activate virtual environment
echo "creating virtual environment..."
python -m venv venv
source venv/bin/activate

# create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/embeddings/chroma_db

# install python dependencies
echo "installing python dependencies..."
pip install -r requirements.txt

# create .env file if it doesn't exist
if [ ! -f .env ]; then
  echo "creating .env file..."
  echo "OPENAI_API_KEY=" > .env
  echo "please add your openai api key to the .env file"
fi

# extract data and create embeddings
echo "extracting data..."
python scripts/extract_data.py

echo "creating embeddings..."
python scripts/create_embeddings.py

echo "setup complete! run 'source venv/bin/activate && python scripts/run_app.py' to start the application"
