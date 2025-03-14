import os
from flask import Flask, request, jsonify, Response, stream_with_context, render_template
from flask_cors import CORS
import json
import time
import logging
from rag.llm import RAG
from rag.constants import MODEL_NAME, TEMPERATURE, K
from dotenv import load_dotenv

# load environment variables
load_dotenv()

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# initialize flask app
app = Flask(__name__, 
            template_folder='../frontend/templates',
            static_folder='../frontend/static')
CORS(app)  # enable cors for all routes

# initialize rag system
rag = RAG(model_name=MODEL_NAME, temperature=TEMPERATURE)

@app.route('/')
def index():
    # serve the main chat interface
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    # get the query from request
    data = request.json
    query = data.get('message', '')
    chat_history = data.get('history', [])
    
    if not query:
        return jsonify({"error": "no message provided"}), 400
    
    # stream the response
    def generate():
        # process the query with rag
        result = rag.answer_question(query, k=K)
        
        # in a real streaming implementation, you would yield chunks as they're generated
        # for now, we'll simulate streaming with the complete answer
        answer = result['answer']
        
        # simple simulation of streaming by sending one word at a time
        words = answer.split()
        for i, word in enumerate(words):
            # yield the word followed by space (except for last word)
            suffix = " " if i < len(words) - 1 else ""
            yield word + suffix
            time.sleep(0.05)  # small delay to simulate streaming
            
    return Response(stream_with_context(generate()), content_type='text/plain')

@app.route('/api/sources', methods=['POST'])
def sources():
    # get the query from request
    data = request.json
    query = data.get('message', '')
    
    if not query:
        return jsonify({"error": "no message provided"}), 400
    
    # get sources for the query
    result = rag.answer_question(query, k=K)
    sources = result['context']
    
    return jsonify({"sources": sources})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
