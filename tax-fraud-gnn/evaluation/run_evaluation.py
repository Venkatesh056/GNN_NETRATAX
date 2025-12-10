"""
GNN Fraud Detection Evaluation Runner

Runs the complete evaluation pipeline:
1. Loads predictions from JSONL file
2. Applies evaluators to compute metrics
3. Generates comprehensive evaluation report
"""
import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Add evaluation module to path
sys.path.insert(0, str(Path(__file__).parent))

from evaluators import (
    AccuracyEvaluator,
    PrecisionRecallF1Evaluator,
    AUCROCEvaluator,
    BatchClassificationEvaluator,
    aggregate_precision_recall_f1,
    compute_auc_roc
)


def load_predictions(jsonl_path: Path) -> List[Dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(jsonl_path, "r") as f:
        for line in f:
            predictions.append(json.loads(line.strip()))
    return predictions


def evaluate_single_sample_approach(predictions: List[Dict]) -> Dict[str, float]:
    """
    Evaluate using single-sample evaluators and aggregate.
    
    This approach follows the Azure AI Evaluation SDK pattern of
    evaluating each sample individually then aggregating.
    """
    accuracy_eval = AccuracyEvaluator()
    prf_eval = PrecisionRecallF1Evaluator()
    auc_eval = AUCROCEvaluator()
    
    accuracy_results = []
    prf_results = []
    auc_results = []
    
    for pred in predictions:
        # Accuracy evaluation
        acc_result = accuracy_eval(
            response=pred["response"],
            ground_truth=pred["ground_truth"]
        )
        accuracy_results.append(acc_result)
        
        # Precision/Recall/F1 components
        prf_result = prf_eval(
            response=pred["response"],
            ground_truth=pred["ground_truth"]
        )
        prf_results.append(prf_result)
        
        # AUC-ROC data collection
        auc_result = auc_eval(
            fraud_probability=pred["fraud_probability"],
            ground_truth=pred["ground_truth"]
        )
        auc_results.append(auc_result)
    
    # Aggregate results
    mean_accuracy = np.mean([r["accuracy"] for r in accuracy_results])
    prf_metrics = aggregate_precision_recall_f1(prf_results)
    auc_roc = compute_auc_roc(auc_results)
    
    return {
        "accuracy": mean_accuracy,
        "precision": prf_metrics["precision"],
        "recall": prf_metrics["recall"],
        "f1_score": prf_metrics["f1_score"],
        "auc_roc": auc_roc,
        "total_samples": len(predictions)
    }


def evaluate_batch_approach(predictions: List[Dict]) -> Dict[str, float]:
    """
    Evaluate using batch evaluator (more efficient for sklearn metrics).
    """
    batch_eval = BatchClassificationEvaluator()
    
    responses = [p["response"] for p in predictions]
    ground_truths = [p["ground_truth"] for p in predictions]
    fraud_probabilities = [p["fraud_probability"] for p in predictions]
    
    return batch_eval(
        responses=responses,
        ground_truths=ground_truths,
        fraud_probabilities=fraud_probabilities
    )


def evaluate_by_dataset(predictions: List[Dict]) -> Dict[str, Dict[str, float]]:
    """Evaluate each dataset separately."""
    by_dataset = {}
    for pred in predictions:
        ds = pred.get("dataset", "unknown")
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(pred)
    
    results = {}
    for dataset, preds in by_dataset.items():
        results[dataset] = evaluate_batch_approach(preds)
    
    return results


def generate_report(
    overall_metrics: Dict[str, float],
    dataset_metrics: Dict[str, Dict[str, float]],
    output_path: Path
) -> str:
    """Generate markdown evaluation report."""
    
    report = []
    report.append("# GNN Fraud Detection Evaluation Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Total Samples Evaluated:** {overall_metrics.get('total_samples', 'N/A')}")
    
    report.append("\n## Overall Metrics\n")
    report.append("| Metric | Score |")
    report.append("|--------|-------|")
    report.append(f"| **Accuracy** | {overall_metrics['accuracy']:.4f} ({overall_metrics['accuracy']*100:.2f}%) |")
    report.append(f"| **Precision** | {overall_metrics['precision']:.4f} ({overall_metrics['precision']*100:.2f}%) |")
    report.append(f"| **Recall** | {overall_metrics['recall']:.4f} ({overall_metrics['recall']*100:.2f}%) |")
    report.append(f"| **F1 Score** | {overall_metrics['f1_score']:.4f} |")
    report.append(f"| **AUC-ROC** | {overall_metrics['auc_roc']:.4f} |")
    
    if "true_positives" in overall_metrics:
        report.append("\n### Confusion Matrix\n")
        report.append("```")
        report.append("                  Predicted")
        report.append("                  No Fraud    Fraud")
        report.append(f"Actual No Fraud     {overall_metrics['true_negatives']:5d}    {overall_metrics['false_positives']:5d}")
        report.append(f"       Fraud        {overall_metrics['false_negatives']:5d}    {overall_metrics['true_positives']:5d}")
        report.append("```")
    
    report.append("\n## Per-Dataset Metrics\n")
    report.append("| Dataset | Accuracy | Precision | Recall | F1 | AUC-ROC |")
    report.append("|---------|----------|-----------|--------|----|----|")
    
    for dataset, metrics in sorted(dataset_metrics.items()):
        report.append(
            f"| {dataset} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['f1_score']:.3f} | {metrics['auc_roc']:.3f} |"
        )
    
    report.append("\n## Metric Definitions\n")
    report.append("- **Accuracy**: Proportion of correct predictions (TP + TN) / Total")
    report.append("- **Precision**: Of predicted frauds, how many are actual frauds: TP / (TP + FP)")
    report.append("- **Recall**: Of actual frauds, how many are detected: TP / (TP + FN)")
    report.append("- **F1 Score**: Harmonic mean of Precision and Recall")
    report.append("- **AUC-ROC**: Area under ROC curve, measures discrimination ability")
    
    report.append("\n## Interpretation\n")
    
    accuracy = overall_metrics['accuracy']
    f1 = overall_metrics['f1_score']
    auc = overall_metrics['auc_roc']
    
    if accuracy >= 0.9 and f1 >= 0.8 and auc >= 0.9:
        report.append("✅ **Excellent Performance**: The model shows strong fraud detection capability.")
    elif accuracy >= 0.8 and f1 >= 0.6:
        report.append("⚠️ **Good Performance**: The model performs well but has room for improvement.")
    else:
        report.append("❌ **Needs Improvement**: Consider retraining with more data or adjusting model architecture.")
    
    report_text = "\n".join(report)
    
    # Save report
    report_path = output_path / "EVALUATION_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report_text)
    
    # Also save metrics as JSON
    metrics_path = output_path / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "overall": overall_metrics,
            "by_dataset": dataset_metrics,
            "generated_at": datetime.now().isoformat()
        }, f, indent=2)
    
    return report_text


def run_evaluation(predictions_path: Path = None):
    """Main evaluation runner."""
    print("=" * 60)
    print("GNN FRAUD DETECTION - EVALUATION PIPELINE")
    print("=" * 60)
    
    eval_dir = Path(__file__).parent
    
    if predictions_path is None:
        predictions_path = eval_dir / "data" / "predictions.jsonl"
    
    if not predictions_path.exists():
        print(f"❌ Predictions file not found: {predictions_path}")
        print("Run collect_predictions.py first to generate predictions.")
        return None
    
    # Load predictions
    print(f"\n📂 Loading predictions from: {predictions_path}")
    predictions = load_predictions(predictions_path)
    print(f"   Loaded {len(predictions)} predictions")
    
    # Evaluate overall
    print("\n📊 Computing metrics...")
    overall_metrics = evaluate_batch_approach(predictions)
    overall_metrics["total_samples"] = len(predictions)
    
    # Evaluate per dataset
    dataset_metrics = evaluate_by_dataset(predictions)
    
    # Generate report
    print("\n📝 Generating report...")
    report = generate_report(overall_metrics, dataset_metrics, eval_dir)
    
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    
    print(f"\n✅ Report saved to: {eval_dir / 'EVALUATION_REPORT.md'}")
    print(f"✅ Metrics saved to: {eval_dir / 'evaluation_metrics.json'}")
    
    return overall_metrics


if __name__ == "__main__":
    run_evaluation()
