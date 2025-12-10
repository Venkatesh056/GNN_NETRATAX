"""
GNN Fraud Detection Evaluation Package

This package provides evaluation tools for the GNN-based tax fraud detection system.

Modules:
    - evaluators: Custom evaluator classes for classification metrics
    - collect_predictions: Script to collect model predictions
    - run_evaluation: Main evaluation runner
    - run_standalone_evaluation: Standalone evaluation script

Usage:
    from evaluation.evaluators import BatchClassificationEvaluator
    
    evaluator = BatchClassificationEvaluator()
    results = evaluator(
        responses=[0, 1, 1, 0],
        ground_truths=[0, 1, 0, 0],
        fraud_probabilities=[0.1, 0.9, 0.7, 0.3]
    )
"""

from .evaluators import (
    AccuracyEvaluator,
    PrecisionRecallF1Evaluator,
    AUCROCEvaluator,
    BatchClassificationEvaluator,
    aggregate_precision_recall_f1,
    compute_auc_roc
)

__all__ = [
    'AccuracyEvaluator',
    'PrecisionRecallF1Evaluator',
    'AUCROCEvaluator',
    'BatchClassificationEvaluator',
    'aggregate_precision_recall_f1',
    'compute_auc_roc'
]

__version__ = "1.0.0"
