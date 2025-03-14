#!/bin/bash

# setup.sh - Automates the RAG system setup and query process

# display step information with formatting
function display_step() {
    echo ""
    echo "============================================================"
    echo "STEP $1: $2"
    echo "============================================================"
    echo ""
}

# check if python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    display_step 1 "Creating virtual environment"
    python3 -m venv venv
    echo "virtual environment created."
fi

# activate virtual environment
display_step 2 "Activating virtual environment"
source venv/bin/activate

# install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    display_step 3 "Installing dependencies"
    pip install -r requirements.txt
else
    display_step 3 "Installing required packages"
    # install minimum required packages
    pip install langchain langchain_openai openai python-dotenv requests
fi

# create necessary directories
display_step 4 "Creating directory structure"
mkdir -p data/raw data/processed data/embeddings

# extract data from zendesk
display_step 5 "Extracting data from Zendesk API"
python scripts/extract_data.py

# create embeddings
display_step 6 "Creating document chunks and embeddings"
python scripts/create_embeddings.py

# # create vector store
# display_step 7 "Loading embeddings into vector store"
# python -c "from rag.vectorstore import VectorStore; vs = VectorStore(); vs.load_from_chunks_file()"

# run query
display_step 8 "Running query against the RAG system"
python scripts/run_query.py

# deactivate virtual environment
deactivate

echo ""
echo "Setup complete! The RAG system has been initialized and queried."
echo ""
