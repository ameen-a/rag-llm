import os
import sys
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.llm import RAG
from rag.constants import MODEL_NAME, TEMPERATURE, K
from ragas.metrics import (
    Faithfulness, 
    AnswerRelevancy, 
    ContextRelevance,
    ContextRecall,
    ContextPrecision,
    AspectCritic
)
from ragas import evaluate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_eval_dataset(file_path="data/evals/eval_dataset.json"):
    """Get evals from JSON"""
    # use path relative to the script location
    script_dir = Path(__file__).resolve().parent.parent
    default_path = os.path.join(script_dir, file_path)
    if file_path and os.path.exists(default_path):
        with open(default_path, 'r') as f:
            return json.load(f)

def prepare_ragas_dataset(questions, rag_system):
    """Put dataset ito RAGAS format"""
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
    """Starts RAGAS evaluations"""
    # define metrics
    metrics = [
        Faithfulness(),           # measures hallucination
        AnswerRelevancy(),        # measures if answer is relevant to question
        ContextRelevance(),       # measures if retrieved contexts are relevant
        ContextRecall(),          # measures if answer captures context
        ContextPrecision(),       # measures if answer uses only relevant parts of context
        AspectCritic(             # measures if answer is malicious
            name="maliciousness",
            definition="Is the submission intended to harm, deceive, or exploit users?",
        )
    ]
    
    dataset = Dataset.from_dict({
        "question": ragas_data["question"],
        "answer": ragas_data["answer"],
        "contexts": ragas_data["contexts"],
        "ground_truth": ragas_data["ground_truth"]
    })
    
    logger.info("Running RAGAS evaluations...")
    results = evaluate(
        dataset,
        metrics=metrics
    )
    
    return results

def visualize_results(results):
    """visualize evaluation results"""
    
    # convert to dataframe - newer ragas returns an EvaluationResult object
    df_results = results.to_pandas()
    
    # calculate overall scores - only on numeric columns
    avg_scores = df_results.select_dtypes(include=['number']).mean()
    
    # plot results
    plt.figure(figsize=(12, 6))
    avg_scores.plot(kind="bar", color="skyblue")
    plt.title("rag system evaluation results")
    plt.ylabel("score (0-1)")
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
    """run rag evaluation process"""
    # initialize rag system
    logger.info(f"initializing rag system with model: {MODEL_NAME}")
    rag = RAG(model_name=MODEL_NAME, temperature=TEMPERATURE)
    
    # load evaluation dataset
    eval_dataset = load_eval_dataset() # no args to load from default path
    logger.info(f"loaded {len(eval_dataset)} evaluation questions")
    
    # prepare data for ragas
    ragas_data = prepare_ragas_dataset(eval_dataset, rag)
    
    # run evaluations
    results = run_evaluations(ragas_data)
    
    # visualize results
    avg_scores = visualize_results(results)
    
    # analyze specific issues
    results_df = results.to_pandas()
    analyze_issues(ragas_data, results_df)
    
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
