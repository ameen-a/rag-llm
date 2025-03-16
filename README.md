## Voy - RAG case study
*Author: [Ameen Ahmed](https://github.com/ameen-a)*

This system extracts content from Voy's [Zendesk FAQ pages](https://joinvoy.zendesk.com/hc/en-gb), processes it into searchable chunks, and uses an LLM to answer user questions based on the most relevant chunks. We also include an evaluation script to measure the performance of the system and reduce hallucinations.

### Architecture
The system contains the following components:
#### Data Pipeline
- **Extraction**: Fetches content from Voy's Zendesk knowledge base using their API
- **Processing**: Cleans the HTML, normalises text, and splits the documents into chunks
- **Embedding**: Converts text chunks into vector embeddings using OpenAI's embedding model (`text-embedding-3-small`)
- **Storage**: Stores embeddings in a ChromaDB vector database for semantic similarity search

#### Question Answering Flow
1. **Query Processing**: The user submits a question through web interface or CLI
2. **Retrieval**: The system finds the most relevant document chunks using vector similarity search
3. **Context Formation**: The retrieved chunks are formatted into a context prompt
4. **Generation**: The LLM, `GPT-4o`, generates an answer based on the context and question
5. **Evaluation**: The system evaluates answers for factual accuracy and hallucination potential using the RAGAS framework

### Evaluation Strategy
The system is evaluated using the RAGAS framework which captures the following metrics:
- **RAGAS Score**: Measures the quality of the answer based on relevance, accuracy, and consistency
- **Factual Correctness**: Checks if the answer is factually correct
- **Hallucination**: Evaluates if the answer contains any hallucinatory content

there is a script which does X Y Z and

- i tried to reduce it answering questions that aren't relevant to the app by doing simple prompt engineering

Talk about context recall 


### Libraries

- **LangChain**: For RAG pipeline components, text splitting, and LLM integration
- **OpenAI**: For vector embeddings model and LLM
- **ChromaDB**: Vector database for storing and retrieving embeddings
- **Flask**: Web server for the browser interface
- **RAGAS**: Evaluation framework for measuring RAG system performance
- **Pandas/Matplotlib**: For analysis of evaluation results
- **BeautifulSoup**: For HTML parsing

### Setup 



Evaluation Examples
The system has been tested with sample questions:
"What payment options are available?"
"What is the titration pathway for wegovy?"
"How much weight loss can I expect with GLP-1 medications?"
Evaluation metrics include faithfulness, answer relevancy, and context precision to ensure responses are accurate and helpful.
Future Improvements
Implement user feedback loop to improve retrieval and answer quality
Add more robust hallucination detection mechanisms
Expand document sources beyond FAQs to include more comprehensive information
Build a more sophisticated frontend with chat history




