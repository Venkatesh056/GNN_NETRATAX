"""
Custom Evaluators for GNN Fraud Detection

Implements code-based evaluators for:
- Accuracy
- Precision/Recall/F1
- AUC-ROC

These evaluators follow Azure AI Evaluation SDK patterns but use sklearn
for the actual metric calculations.
"""
import numpy as np
from typing import Dict, Any, List, Union


class AccuracyEvaluator:
    """
    Evaluator for binary classification accuracy.
    
    Compares predicted class to ground truth and returns 1 if match, 0 otherwise.
    Aggregation is handled by the evaluate() API.
    """
    
    id = "accuracy_evaluator"
    display_name = "Accuracy Evaluator"
    
    def __init__(self):
        pass
    
    def __call__(
        self,
        *,
        response: int,
        ground_truth: int,
        **kwargs
    ) -> Dict[str, int]:
        """
        Evaluate single prediction accuracy.
        
        Args:
            response: Predicted class (0 or 1)
            ground_truth: Actual class (0 or 1)
        
        Returns:
            Dict with "accuracy" key (1 if correct, 0 if incorrect)
        """
        is_correct = int(response == ground_truth)
        return {"accuracy": is_correct}


class PrecisionRecallF1Evaluator:
    """
    Evaluator that tracks precision, recall, and F1 for the fraud class.
    
    For batch evaluation, collects TP/FP/FN counts to compute metrics.
    """
    
    id = "precision_recall_f1_evaluator"
    display_name = "Precision/Recall/F1 Evaluator"
    
    def __init__(self):
        pass
    
    def __call__(
        self,
        *,
        response: int,
        ground_truth: int,
        **kwargs
    ) -> Dict[str, Union[int, float]]:
        """
        Evaluate single prediction for precision/recall components.
        
        Returns component values that can be aggregated:
        - tp: True positive (predicted fraud, actually fraud)
        - fp: False positive (predicted fraud, actually not fraud)
        - fn: False negative (predicted not fraud, actually fraud)
        - tn: True negative (predicted not fraud, actually not fraud)
        """
        tp = int(response == 1 and ground_truth == 1)
        fp = int(response == 1 and ground_truth == 0)
        fn = int(response == 0 and ground_truth == 1)
        tn = int(response == 0 and ground_truth == 0)
        
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }


class AUCROCEvaluator:
    """
    Evaluator that collects data for AUC-ROC calculation.
    
    AUC-ROC requires the full probability distribution, so this evaluator
    captures the fraud probability and ground truth for later aggregation.
    """
    
    id = "auc_roc_evaluator"
    display_name = "AUC-ROC Evaluator"
    
    def __init__(self):
        pass
    
    def __call__(
        self,
        *,
        fraud_probability: float,
        ground_truth: int,
        **kwargs
    ) -> Dict[str, float]:
        """
        Capture fraud probability and ground truth for AUC-ROC.
        
        Args:
            fraud_probability: Model's predicted probability of fraud
            ground_truth: Actual class (0 or 1)
        
        Returns:
            Dict with probability and label for aggregation
        """
        return {
            "fraud_probability": fraud_probability,
            "ground_truth_label": ground_truth
        }


class BatchClassificationEvaluator:
    """
    Batch evaluator that computes all classification metrics at once.
    
    This evaluator processes an entire dataset and returns:
    - Accuracy
    - Precision (for fraud class)
    - Recall (for fraud class)
    - F1 Score (for fraud class)
    - AUC-ROC
    """
    
    id = "batch_classification_evaluator"
    display_name = "Batch Classification Metrics"
    
    def __init__(self):
        pass
    
    def __call__(
        self,
        *,
        responses: List[int],
        ground_truths: List[int],
        fraud_probabilities: List[float] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute all classification metrics for a batch.
        
        Args:
            responses: List of predicted classes
            ground_truths: List of actual classes
            fraud_probabilities: Optional list of fraud probabilities for AUC-ROC
        
        Returns:
            Dict with all metric scores
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix
        )
        
        y_true = np.array(ground_truths)
        y_pred = np.array(responses)
        
        results = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0))
        }
        
        # Compute AUC-ROC if probabilities provided
        if fraud_probabilities is not None:
            y_score = np.array(fraud_probabilities)
            try:
                results["auc_roc"] = float(roc_auc_score(y_true, y_score))
            except ValueError:
                # Handle case where all labels are same class
                results["auc_roc"] = 0.5
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        results["true_negatives"] = int(cm[0, 0])
        results["false_positives"] = int(cm[0, 1])
        results["false_negatives"] = int(cm[1, 0])
        results["true_positives"] = int(cm[1, 1])
        
        return results


def aggregate_precision_recall_f1(results: List[Dict]) -> Dict[str, float]:
    """
    Aggregate individual TP/FP/FN counts into precision/recall/F1.
    
    Args:
        results: List of dicts with tp, fp, fn, tn keys
    
    Returns:
        Dict with precision, recall, f1_score
    """
    total_tp = sum(r.get("tp", 0) for r in results)
    total_fp = sum(r.get("fp", 0) for r in results)
    total_fn = sum(r.get("fn", 0) for r in results)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn
    }


def compute_auc_roc(results: List[Dict]) -> float:
    """
    Compute AUC-ROC from collected probabilities and labels.
    
    Args:
        results: List of dicts with fraud_probability and ground_truth_label
    
    Returns:
        AUC-ROC score
    """
    from sklearn.metrics import roc_auc_score
    
    y_true = [r["ground_truth_label"] for r in results]
    y_score = [r["fraud_probability"] for r in results]
    
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return 0.5  # Default for single-class case
