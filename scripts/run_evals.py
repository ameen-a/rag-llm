import os
import sys
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# add the parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.llm import RAG
from rag.constants import MODEL_NAME, TEMPERATURE, K
from ragas.metrics import (
    faithfulness, 
    answer_relevancy, 
    context_relevancy,
    context_recall,
    context_precision
)
from ragas.metrics.critique import harmfulness
from ragas import evaluate

# configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# ******
# how do i get this to consider context provided to the rag system??
# ***
def load_eval_dataset(file_path=None):
    """
    load or create evaluation dataset
    
    format should be a list of dicts with:
    - question: the question to ask
    - ground_truth: the expected answer (optional)
    """
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    # use the default path if none specified
    default_path = "data/evals/eval_dataset.json"
    if os.path.exists(default_path):
        with open(default_path, 'r') as f:
            return json.load(f)

def prepare_ragas_dataset(questions, rag_system):
    """prepare dataset in ragas format"""
    
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    logger.info("generating answers for evaluation dataset...")
    for item in tqdm(questions):
        query = item["question"]
        result = rag_system.answer_question(query, k=K)
        
        ragas_data["question"].append(query)
        ragas_data["answer"].append(result["answer"])
        ragas_data["contexts"].append([doc["content"] for doc in result["context"]])
        
        if "ground_truth" in item:
            ragas_data["ground_truth"].append(item["ground_truth"])
        else:
            ragas_data["ground_truth"].append("")  # empty if no ground truth
    
    return ragas_data

def run_evaluations(ragas_data):
    """run ragas evaluations"""
    
    # define metrics
    metrics = [
        faithfulness,           # measures hallucination
        answer_relevancy,       # measures if answer is relevant to question
        context_relevancy,      # measures if retrieved contexts are relevant
        context_recall,         # measures if answer captures context
        context_precision,      # measures if answer uses only relevant parts of context
        harmfulness             # measures if answer contains harmful content
    ]
    
    # run evaluation
    logger.info("running ragas evaluations...")
    results = evaluate(
        ragas_data["question"],
        ragas_data["answer"],
        ragas_data["contexts"],
        ragas_data["ground_truth"],
        metrics=metrics
    )
    
    return results

def visualize_results(results):
    """visualize evaluation results"""
    
    # convert to dataframe for easier manipulation
    df_results = pd.DataFrame(results)
    
    # calculate overall scores
    avg_scores = df_results.mean()
    
    # plot results
    plt.figure(figsize=(12, 6))
    avg_scores.plot(kind="bar", color="skyblue")
    plt.title("RAG System Evaluation Results")
    plt.ylabel("Score (0-1)")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # save figure
    os.makedirs("data/evaluation", exist_ok=True)
    plt.savefig("data/evaluation/rag_evaluation_results.png")
    
    # save detailed results
    df_results.to_csv("data/evaluation/detailed_results.csv", index=False)
    
    logger.info(f"overall average scores:\n{avg_scores}")
    return avg_scores

def analyze_issues(ragas_data, results_df):
    """analyze specific issues in the rag system"""
    
    # find worst performing questions
    faithfulness_issues = results_df.sort_values("faithfulness").head(3)
    relevancy_issues = results_df.sort_values("answer_relevancy").head(3)
    
    logger.info("\n===== potential hallucination issues =====")
    for i, row in faithfulness_issues.iterrows():
        q_idx = row.name
        logger.info(f"question: {ragas_data['question'][q_idx]}")
        logger.info(f"answer: {ragas_data['answer'][q_idx]}")
        logger.info(f"faithfulness score: {row['faithfulness']:.2f}")
        logger.info("-" * 80)
    
    logger.info("\n===== potential relevancy issues =====")
    for i, row in relevancy_issues.iterrows():
        q_idx = row.name
        logger.info(f"question: {ragas_data['question'][q_idx]}")
        logger.info(f"answer: {ragas_data['answer'][q_idx]}")
        logger.info(f"relevancy score: {row['answer_relevancy']:.2f}")
        logger.info("-" * 80)

def main():
    # initialize rag system
    logger.info(f"initializing rag system with model: {MODEL_NAME}")
    rag = RAG(model_name=MODEL_NAME, temperature=TEMPERATURE)
    
    # load or create evaluation dataset
    eval_dataset = load_eval_dataset() # no args to load from default path
    logger.info(f"loaded {len(eval_dataset)} evaluation questions")
    
    # prepare data for ragas
    ragas_data = prepare_ragas_dataset(eval_dataset, rag)
    
    # run evaluations
    results = run_evaluations(ragas_data)
    
    # visualize results
    avg_scores = visualize_results(results)
    
    # analyze specific issues
    analyze_issues(ragas_data, results)
    
    # calculate overall rag score
    overall_score = avg_scores.mean()
    logger.info(f"overall rag quality score: {overall_score:.2f}/1.00")
    
    # save results summary
    summary = {
        "overall_score": float(overall_score),
        "metrics": {name: float(score) for name, score in avg_scores.items()},
        "num_questions": len(eval_dataset),
        "model_name": MODEL_NAME,
        "k_value": K
    }
    
    with open("data/evaluation/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary

if __name__ == "__main__":
    main()
