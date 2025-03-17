# Voy - RAG Case Study
*Author: [Ameen Ahmed](https://github.com/ameen-a)*

This work extracts content from Voy's [Zendesk FAQ pages](https://joinvoy.zendesk.com/hc/en-gb), processes it into searchable chunks, and uses an LLM to answer user questions based on the most relevant chunks. It also includes an evaluation script to measure the performance of the system and mitigate against hallucinations.

## Architecture
<!-- The system contains the following components:
#### Data Pipeline
- **Extraction**: Fetches content from Voy's Zendesk knowledge base using their API.
- **Processing**: Cleans the HTML, normalises text, and splits the documents into chunks.
- **Embedding**: Converts text chunks into vector embeddings using OpenAI's embedding model, `text-embedding-3-small`.
- **Storage**: Stores embeddings in a ChromaDB vector database for semantic similarity search. -->


The system contains a data pipeline that begins with **extraction**, which fetches content from Voy's Zendesk knowledge base using their API, then continues with **processing** to clean the HTML, normalize text, and split the documents into chunks, followed by **embedding** that converts these text chunks into vector embeddings using OpenAI's `text-embedding-3-small`, and finally concludes with **storage**, where the embeddings are kept in a ChromaDB vector database for semantic similarity search.

#### Question Answering Flow
1. **Query Processing**: The user submits a question through web interface or CLI.
2. **Retrieval**: The system finds the most relevant document chunks using vector similarity search.
3. **Context Formation**: The retrieved chunks are formatted into a context prompt.
4. **Generation**: The LLM, `GPT-4o`, generates an answer based on the context and question.
5. **Evaluation**: The system evaluates answers for factual accuracy and hallucination potential using the RAGAS framework.

## Evaluation Strategy
The system is evaluated using the RAGAS framework which captures the several important metrics, including _factuality_, _answer relevancy_, and _context precision_. The `run_evals.py` script runs the evals on three test questions (with ground truth answers) found in `data/evals/eval_dataset.json` and performs a simple analysis [_see image below_].

![image](data/eval_results/rag_evaluation_results.png)

 This dataset also contains the ground truth answers for each question, which are used to calculate certain RAGAS metrics like _faithfulness_ and _answer relevancy_. A particularly important metric is **_context precision_**. It measures the proportion of the retrieved context that is relevant to the question, and is a good measure of how much the retrieved context is actually useful for answering the question. A full list of metrics and their calculation methods are available in the [RAGAS documentation](https://docs.ragas.io/en/latest/references/metrics/). 


An evaluation pipeline is essential to any LLM project, particularly if the LLM is user-facing. It requires some work to create ground truth answers, but the benefits are that any arbitrary model, prompt template, hyperparameter, RAG approach, or technique (like fine-tuning) can be systematically evaluated. It's also essential to have tracing functionality: knowing how users interact with the LLM is necessary to meaningfully improve it over time and catch failure cases. An experiment tracking tool like [Weights & Biases](https://wandb.ai/site) provides all of these features, as well as other useful tools like unit testing outputs, model versioning, and hyperparameter tuning.

## Future Improvements

- Given the time constaints, a trivial implementation of embedding and retrieval was used. I would revisit this and use a hybrid approach (keyword + semantic search), with an embedding model that is better suited to document-based Q&A. 
- The current chunking strategy is also naive: it splits the documents arbitrarily into chunks of 1000 tokens (with some overlap). A better approach might be to split by article given they are relatively short. This would also clean up the output by avoiding overlapped sources. 
- Add some agentic functionality: an evaluator LLM could be used a final check to flag any potential hallucinations and misrepresentations of the context, and a prompt formatting LLM could be used to format the user query in a way that is more likely to be helpful. If the corpus size is huge, one could split the documents into different vector databases and have an LLM route the query to the most relevant one.
- As mentioned, a proper **eval pipeline**.

### Libraries

- **LangChain**: For RAG pipeline components, text splitting, and LLM integration
- **OpenAI**: For vector embeddings model and LLM
- **ChromaDB**: Vector database for storing and retrieving embeddings
- **Flask**: Web server for the browser interface
- **RAGAS**: Evaluation framework for measuring RAG system performance
- **Pandas/Matplotlib**: For analysis of evaluation results
- **BeautifulSoup**: For HTML parsing

### Setup

The `setup.sh` script will download the Voy Zendesk FAQ data, process it, and create the vector embeddings. It will also create a virtual environment and install the necessary libraries.

```bash
# first, add your openai api key to the .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env

# then from root, run the setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

Running the Application

The system can be used in two ways:

```bash
# start the web interface
python scripts/run_app.py
# then visit http://127.0.0.1:5000 in your browser

# or use the command line interface
python scripts/run_query.py --query "your question here"
```

#### Evaluation

To measure the system's performance:

```bash
# run the evaluation pipeline
python scripts/run_evals.py
```

This will generate performance metrics in `data/eval_results`.
