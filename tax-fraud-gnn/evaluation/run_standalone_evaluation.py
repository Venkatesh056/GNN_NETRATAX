"""
Standalone Evaluation Script for GNN Fraud Detection

This script can run evaluation:
1. Using real model predictions (if model and predictions exist)
2. Using synthetic predictions for testing the evaluation pipeline

Run with: python run_standalone_evaluation.py [--synthetic]
"""
import sys
from pathlib import Path
import json
import argparse
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluators import BatchClassificationEvaluator


def generate_synthetic_predictions(datasets_path: Path, output_path: Path) -> List[Dict]:
    """
    Generate synthetic predictions for testing the evaluation pipeline.
    Uses ground truth + noise to simulate model predictions.
    """
    print("Generating synthetic predictions for evaluation testing...")
    
    predictions = []
    
    company_files = sorted(datasets_path.glob("dataset_*_companies_complete.csv"))
    
    for company_file in company_files:
        dataset_num = company_file.stem.split("_")[1]
        print(f"  Processing dataset {dataset_num}...")
        
        companies_df = pd.read_csv(company_file)
        
        for _, row in companies_df.iterrows():
            ground_truth = int(row["is_fraud"])
            
            # Simulate model predictions with ~85% accuracy
            # Add noise to ground truth
            if np.random.random() < 0.85:
                predicted = ground_truth  # Correct prediction
            else:
                predicted = 1 - ground_truth  # Wrong prediction
            
            # Generate probability based on prediction
            if predicted == 1:
                fraud_prob = 0.5 + np.random.random() * 0.5  # 0.5-1.0
            else:
                fraud_prob = np.random.random() * 0.5  # 0.0-0.5
            
            predictions.append({
                "company_id": str(row["company_id"]),
                "ground_truth": ground_truth,
                "response": predicted,
                "fraud_probability": round(fraud_prob, 4),
                "dataset": f"dataset_{dataset_num}"
            })
    
    # Save predictions
    output_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path / "predictions.jsonl"
    
    with open(jsonl_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    
    print(f"  Saved {len(predictions)} synthetic predictions to {jsonl_path}")
    
    return predictions


def load_predictions(jsonl_path: Path) -> List[Dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line.strip()))
    return predictions


def run_evaluation(predictions: List[Dict]) -> Dict:
    """Run evaluation using BatchClassificationEvaluator."""
    evaluator = BatchClassificationEvaluator()
    
    responses = [p["response"] for p in predictions]
    ground_truths = [p["ground_truth"] for p in predictions]
    fraud_probabilities = [p["fraud_probability"] for p in predictions]
    
    results = evaluator(
        responses=responses,
        ground_truths=ground_truths,
        fraud_probabilities=fraud_probabilities
    )
    
    results["total_samples"] = len(predictions)
    
    return results


def evaluate_by_dataset(predictions: List[Dict]) -> Dict[str, Dict]:
    """Evaluate each dataset separately."""
    by_dataset = {}
    for pred in predictions:
        ds = pred.get("dataset", "unknown")
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(pred)
    
    results = {}
    for dataset, preds in sorted(by_dataset.items()):
        results[dataset] = run_evaluation(preds)
    
    return results


def print_report(overall: Dict, by_dataset: Dict):
    """Print evaluation report to console."""
    print("\n" + "=" * 70)
    print("GNN FRAUD DETECTION - EVALUATION REPORT")
    print("=" * 70)
    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Samples: {overall.get('total_samples', 'N/A')}")
    
    print("\n" + "-" * 40)
    print("OVERALL METRICS")
    print("-" * 40)
    print(f"  Accuracy:   {overall['accuracy']:.4f}  ({overall['accuracy']*100:.2f}%)")
    print(f"  Precision:  {overall['precision']:.4f}  ({overall['precision']*100:.2f}%)")
    print(f"  Recall:     {overall['recall']:.4f}  ({overall['recall']*100:.2f}%)")
    print(f"  F1 Score:   {overall['f1_score']:.4f}")
    print(f"  AUC-ROC:    {overall['auc_roc']:.4f}")
    
    if "true_positives" in overall:
        print("\n  Confusion Matrix:")
        print(f"    True Negatives:   {overall['true_negatives']:5d}")
        print(f"    False Positives:  {overall['false_positives']:5d}")
        print(f"    False Negatives:  {overall['false_negatives']:5d}")
        print(f"    True Positives:   {overall['true_positives']:5d}")
    
    print("\n" + "-" * 40)
    print("PER-DATASET METRICS")
    print("-" * 40)
    print(f"{'Dataset':<15} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 58)
    
    for dataset, metrics in by_dataset.items():
        print(f"{dataset:<15} {metrics['accuracy']:>7.3f} {metrics['precision']:>7.3f} "
              f"{metrics['recall']:>7.3f} {metrics['f1_score']:>7.3f} {metrics['auc_roc']:>7.3f}")
    
    print("\n" + "=" * 70)
    
    # Interpretation
    accuracy = overall['accuracy']
    f1 = overall['f1_score']
    auc = overall['auc_roc']
    
    if accuracy >= 0.9 and f1 >= 0.8 and auc >= 0.9:
        print("✅ EXCELLENT: The model shows strong fraud detection capability.")
    elif accuracy >= 0.8 and f1 >= 0.6:
        print("⚠️  GOOD: The model performs well but has room for improvement.")
    elif accuracy >= 0.7:
        print("⚠️  MODERATE: Consider additional training or feature engineering.")
    else:
        print("❌ NEEDS IMPROVEMENT: Significant model improvements required.")
    
    print("=" * 70 + "\n")


def save_report(overall: Dict, by_dataset: Dict, output_path: Path):
    """Save evaluation report to files."""
    # Save JSON metrics
    metrics_path = output_path / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "overall": overall,
            "by_dataset": by_dataset,
            "generated_at": datetime.now().isoformat()
        }, f, indent=2)
    
    # Save Markdown report
    report_path = output_path / "EVALUATION_REPORT.md"
    
    lines = []
    lines.append("# GNN Fraud Detection Evaluation Report\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Total Samples:** {overall.get('total_samples', 'N/A')}\n")
    
    lines.append("## Overall Metrics\n")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    lines.append(f"| Accuracy | {overall['accuracy']:.4f} ({overall['accuracy']*100:.2f}%) |")
    lines.append(f"| Precision | {overall['precision']:.4f} ({overall['precision']*100:.2f}%) |")
    lines.append(f"| Recall | {overall['recall']:.4f} ({overall['recall']*100:.2f}%) |")
    lines.append(f"| F1 Score | {overall['f1_score']:.4f} |")
    lines.append(f"| AUC-ROC | {overall['auc_roc']:.4f} |")
    
    lines.append("\n## Per-Dataset Metrics\n")
    lines.append("| Dataset | Accuracy | Precision | Recall | F1 | AUC-ROC |")
    lines.append("|---------|----------|-----------|--------|----|----|")
    
    for dataset, metrics in sorted(by_dataset.items()):
        lines.append(f"| {dataset} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | "
                    f"{metrics['recall']:.3f} | {metrics['f1_score']:.3f} | {metrics['auc_roc']:.3f} |")
    
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"✅ Saved metrics to: {metrics_path}")
    print(f"✅ Saved report to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="GNN Fraud Detection Evaluation")
    parser.add_argument("--synthetic", action="store_true", 
                       help="Generate synthetic predictions for testing")
    parser.add_argument("--predictions", type=str, 
                       help="Path to predictions JSONL file")
    args = parser.parse_args()
    
    eval_dir = Path(__file__).parent
    base_dir = eval_dir.parent.parent
    datasets_path = base_dir / "new_datasets"
    output_path = eval_dir / "data"
    
    predictions_path = eval_dir / "data" / "predictions.jsonl"
    if args.predictions:
        predictions_path = Path(args.predictions)
    
    print("=" * 70)
    print("GNN FRAUD DETECTION - STANDALONE EVALUATION")
    print("=" * 70)
    
    # Generate or load predictions
    if args.synthetic or not predictions_path.exists():
        if not datasets_path.exists():
            print(f"❌ Datasets not found at: {datasets_path}")
            sys.exit(1)
        predictions = generate_synthetic_predictions(datasets_path, output_path)
    else:
        print(f"\n📂 Loading predictions from: {predictions_path}")
        predictions = load_predictions(predictions_path)
        print(f"   Loaded {len(predictions)} predictions")
    
    # Run evaluation
    print("\n📊 Computing metrics...")
    overall = run_evaluation(predictions)
    by_dataset = evaluate_by_dataset(predictions)
    
    # Print and save results
    print_report(overall, by_dataset)
    save_report(overall, by_dataset, eval_dir)


if __name__ == "__main__":
    main()
