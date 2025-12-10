# GNN Fraud Detection Evaluation Framework

This directory contains the evaluation pipeline for the GNN-based tax fraud detection system.

## Latest Evaluation Results

**Run Date:** 2025-12-10

| Metric | Score |
|--------|-------|
| Accuracy | 85.95% |
| Precision | 0.00% |
| Recall | 0.00% |
| F1 Score | 0.00 |
| AUC-ROC | 0.50 |

**⚠️ Important Finding:** The current model predicts ALL samples as non-fraud. This results in high accuracy (~86%) simply because most companies are non-fraudulent (class imbalance). The model needs to be retrained with:
1. Class weighting or oversampling of fraud cases
2. Better feature engineering
3. Threshold adjustment for fraud classification

## Overview

The evaluation framework measures model performance using standard classification metrics:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Proportion of correct predictions |
| **Precision** | Of predicted frauds, how many are actual frauds |
| **Recall** | Of actual frauds, how many were detected |
| **F1 Score** | Harmonic mean of precision and recall |
| **AUC-ROC** | Area under the ROC curve |

## Files

```
evaluation/
├── collect_predictions.py  # Collects model predictions for all datasets
├── evaluators.py           # Custom evaluator classes
├── run_evaluation.py       # Main evaluation runner
├── data/                   # Collected predictions (JSONL + CSV)
├── EVALUATION_REPORT.md    # Generated evaluation report
└── evaluation_metrics.json # Metrics in JSON format
```

## Quick Start

### 1. Collect Predictions

First, run the model on all datasets to collect predictions:

```bash
cd tax-fraud-gnn
python evaluation/collect_predictions.py
```

This will:
- Load the trained model from `models/best_model.pt`
- Process all 10 datasets from `new_datasets/`
- Save predictions to `evaluation/data/predictions.jsonl`

### 2. Run Evaluation

Then compute evaluation metrics:

```bash
python evaluation/run_evaluation.py
```

This will:
- Load predictions from JSONL
- Compute all metrics
- Generate `EVALUATION_REPORT.md`
- Save JSON metrics to `evaluation_metrics.json`

## Evaluator Classes

### AccuracyEvaluator

Per-sample evaluator that returns 1 for correct prediction, 0 for incorrect.

```python
from evaluators import AccuracyEvaluator

eval = AccuracyEvaluator()
result = eval(response=1, ground_truth=1)  # {"accuracy": 1}
```

### PrecisionRecallF1Evaluator

Per-sample evaluator that returns TP/FP/FN/TN components for aggregation.

```python
from evaluators import PrecisionRecallF1Evaluator, aggregate_precision_recall_f1

eval = PrecisionRecallF1Evaluator()
results = [eval(response=pred, ground_truth=gt) for pred, gt in data]
metrics = aggregate_precision_recall_f1(results)
```

### BatchClassificationEvaluator

Batch evaluator that computes all metrics at once using sklearn.

```python
from evaluators import BatchClassificationEvaluator

eval = BatchClassificationEvaluator()
metrics = eval(
    responses=[0, 1, 1, 0],
    ground_truths=[0, 1, 0, 0],
    fraud_probabilities=[0.1, 0.9, 0.7, 0.3]
)
```

## Output Format

### predictions.jsonl

```jsonl
{"company_id": "C001", "ground_truth": 0, "response": 0, "fraud_probability": 0.12, "dataset": "dataset_01"}
{"company_id": "C002", "ground_truth": 1, "response": 1, "fraud_probability": 0.87, "dataset": "dataset_01"}
```

### evaluation_metrics.json

```json
{
  "overall": {
    "accuracy": 0.85,
    "precision": 0.82,
    "recall": 0.79,
    "f1_score": 0.805,
    "auc_roc": 0.92
  },
  "by_dataset": {
    "dataset_01": {...},
    "dataset_02": {...}
  }
}
```

## Integration with Azure AI Evaluation SDK

The evaluators are designed to be compatible with Azure AI Evaluation SDK patterns:

```python
from azure.ai.evaluation import evaluate

# Use custom evaluators
result = evaluate(
    data="evaluation/data/predictions.jsonl",
    evaluators={
        "accuracy": AccuracyEvaluator(),
        "prf": PrecisionRecallF1Evaluator()
    }
)
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- torch
- torch_geometric (for model loading)
