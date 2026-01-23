"""
GNN Model Evaluation Scores
Shows F1 score, accuracy, top features, and inference examples for the fraud detection model
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.gnn_models.train_gnn import GNNFraudDetector


def load_model_and_data():
    """Load the trained model and graph data"""
    data_path = Path(__file__).parent / "data" / "processed"
    models_path = Path(__file__).parent / "models"
    accumulated_path = Path(__file__).parent / "data" / "accumulated"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load companies data
    companies_file = accumulated_path / "companies_accumulated.csv"
    if not companies_file.exists():
        companies_file = data_path / "companies_processed.csv"
    
    companies = pd.read_csv(companies_file)
    print(f"Loaded {len(companies)} companies")
    
    # Load model
    NUM_NODE_FEATURES = 12
    model = GNNFraudDetector(in_channels=NUM_NODE_FEATURES, hidden_channels=64, out_channels=2, model_type="gcn").to(device)
    
    try:
        model.load_state_dict(torch.load(models_path / "best_model.pt", map_location=device))
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None
    
    # Load graph data
    import pickle
    import networkx as nx
    
    try:
        with open(data_path / "graphs" / "networkx_graph.gpickle", "rb") as f:
            G = pickle.load(f)
        print(f"Loaded graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        print(f"Error loading graph: {e}")
        return model, companies, device, None
    
    return model, companies, device, G


def build_graph_data(G, companies, device):
    """Build PyTorch Geometric data from NetworkX graph"""
    from torch_geometric.data import Data
    
    node_list = sorted(list(G.nodes()))
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # Build feature matrix
    x_list = []
    y_list = []
    
    for node in node_list:
        node_data = G.nodes[node]
        features = [
            float(node_data.get("turnover", 0)),
            float(node_data.get("sent_invoices", 0)),
            float(node_data.get("received_invoices", 0)),
            float(node_data.get("total_sent_amount", 0)),
            float(node_data.get("total_received_amount", 0)),
            float(node_data.get("unique_buyers", 0)),
            float(node_data.get("unique_sellers", 0)),
            float(node_data.get("circular_trading_score", 0)),
            float(node_data.get("gst_compliance_rate", 1.0)),
            float(node_data.get("late_filing_count", 0)),
            float(node_data.get("round_amount_ratio", 0)),
            float(node_data.get("buyer_concentration", 0))
        ]
        x_list.append(features)
        y_list.append(int(node_data.get("is_fraud", 0)))
    
    x = torch.tensor(x_list, dtype=torch.float32)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.tensor(y_list, dtype=torch.long)
    
    # Normalize features
    x_min = x.min(dim=0, keepdim=True)[0]
    x_max = x.max(dim=0, keepdim=True)[0]
    x_range = x_max - x_min
    x_range[x_range == 0] = 1
    x = (x - x_min) / x_range
    
    # Build edge index
    edge_index = []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            edge_index.append([node_to_idx[u], node_to_idx[v]])
    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data.to(device), node_list, node_to_idx


def get_feature_names():
    """Return the list of feature names used in the model"""
    return [
        "turnover",
        "sent_invoices",
        "received_invoices", 
        "total_sent_amount",
        "total_received_amount",
        "unique_buyers",
        "unique_sellers",
        "circular_trading_score",
        "gst_compliance_rate",
        "late_filing_count",
        "round_amount_ratio",
        "buyer_concentration"
    ]


def compute_feature_importance(model, device):
    """Estimate feature importance based on model weights"""
    feature_names = get_feature_names()
    
    # Get first layer weights
    try:
        # For GCN, conv1 has a linear transformation
        if hasattr(model, 'conv1') and hasattr(model.conv1, 'lin'):
            weights = model.conv1.lin.weight.detach().cpu().numpy()
        elif hasattr(model, 'conv1') and hasattr(model.conv1, 'weight'):
            weights = model.conv1.weight.detach().cpu().numpy()
        else:
            # Try to access the weight matrix directly
            for name, param in model.named_parameters():
                if 'conv1' in name and 'weight' in name:
                    weights = param.detach().cpu().numpy()
                    break
            else:
                print("Could not find conv1 weights")
                return None
        
        # Compute importance as mean absolute weight per input feature
        if weights.ndim == 2:
            importance = np.mean(np.abs(weights), axis=0)
        else:
            importance = np.mean(np.abs(weights.reshape(-1, len(feature_names))), axis=0)
        
        # Normalize
        importance = importance / importance.sum() * 100
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            "Feature": feature_names[:len(importance)],
            "Importance (%)": importance
        }).sort_values("Importance (%)", ascending=False)
        
        return importance_df
    
    except Exception as e:
        print(f"Error computing feature importance: {e}")
        return None


def run_inference(model, data, device):
    """Run model inference on graph data"""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        proba = torch.softmax(out, dim=1)[:, 1]  # Probability of fraud class
    return proba.cpu().numpy()


def evaluate_model():
    """Main evaluation function"""
    print("=" * 60)
    print("GNN TAX FRAUD DETECTION MODEL - EVALUATION SCORES")
    print("=" * 60)
    print()
    
    # Load model and data
    model, companies, device, G = load_model_and_data()
    
    if model is None:
        print("Failed to load model")
        return
    
    if G is None:
        print("Failed to load graph - cannot run inference")
        return
    
    # Build graph data for inference
    print("\nBuilding graph data for inference...")
    data, node_list, node_to_idx = build_graph_data(G, companies, device)
    
    # Run fresh inference
    print("Running model inference...")
    fraud_proba = run_inference(model, data, device)
    y_true = data.y.cpu().numpy()
    y_pred = (fraud_proba > 0.5).astype(int)
    
    print(f"Inference complete: {len(fraud_proba)} predictions generated")
    
    # Map node predictions back to companies
    node_predictions = dict(zip(node_list, fraud_proba))
    
    # =========================================================================
    # CLASSIFICATION METRICS
    # =========================================================================
    print("\n" + "=" * 60)
    print("CLASSIFICATION METRICS")
    print("=" * 60)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc_roc = roc_auc_score(y_true, fraud_proba)
    except:
        auc_roc = 0.0
    
    print(f"\n  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"  AUC-ROC:   {auc_roc:.4f} ({auc_roc*100:.2f}%)")
    
    # =========================================================================
    # CONFUSION MATRIX
    # =========================================================================
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n                  Predicted")
    print(f"                  Normal  Fraud")
    print(f"  Actual Normal   {cm[0][0]:5d}  {cm[0][1]:5d}")
    print(f"  Actual Fraud    {cm[1][0]:5d}  {cm[1][1]:5d}")
    
    # =========================================================================
    # CLASSIFICATION REPORT
    # =========================================================================
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    print()
    print(classification_report(y_true, y_pred, target_names=["Normal", "Fraud"], zero_division=0))
    
    # =========================================================================
    # TOP FEATURES
    # =========================================================================
    print("\n" + "=" * 60)
    print("TOP FEATURES (by model weight importance)")
    print("=" * 60)
    
    importance_df = compute_feature_importance(model, device)
    
    if importance_df is not None:
        print("\n  Rank  Feature                    Importance")
        print("  " + "-" * 50)
        for i, (_, row) in enumerate(importance_df.iterrows(), 1):
            print(f"  {i:2d}.   {row['Feature']:<25} {row['Importance (%)']:6.2f}%")
    else:
        print("\n  Feature Names (12 total):")
        for i, name in enumerate(get_feature_names(), 1):
            print(f"  {i:2d}. {name}")
    
    # =========================================================================
    # DATA STATISTICS
    # =========================================================================
    print("\n" + "=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)
    
    total = len(y_true)
    fraud_count = y_true.sum()
    normal_count = total - fraud_count
    
    print(f"\n  Total Companies:    {total:,}")
    print(f"  Normal Companies:   {normal_count:,} ({normal_count/total*100:.1f}%)")
    print(f"  Fraud Companies:    {fraud_count:,} ({fraud_count/total*100:.1f}%)")
    
    if G is not None:
        print(f"\n  Graph Nodes:        {G.number_of_nodes():,}")
        print(f"  Graph Edges:        {G.number_of_edges():,}")
        print(f"  Avg Connections:    {G.number_of_edges()/G.number_of_nodes():.2f}")
    
    # Fraud probability distribution
    print(f"\n  Fraud Probability Distribution:")
    print(f"    Min:    {fraud_proba.min():.4f}")
    print(f"    Max:    {fraud_proba.max():.4f}")
    print(f"    Mean:   {fraud_proba.mean():.4f}")
    print(f"    Median: {np.median(fraud_proba):.4f}")
    
    high_risk = (fraud_proba > 0.7).sum()
    med_risk = ((fraud_proba > 0.3) & (fraud_proba <= 0.7)).sum()
    low_risk = (fraud_proba <= 0.3).sum()
    
    print(f"\n  Risk Distribution:")
    print(f"    High Risk (>0.7):     {high_risk:,} ({high_risk/total*100:.1f}%)")
    print(f"    Medium Risk (0.3-0.7): {med_risk:,} ({med_risk/total*100:.1f}%)")
    print(f"    Low Risk (<=0.3):     {low_risk:,} ({low_risk/total*100:.1f}%)")
    
    # =========================================================================
    # INFERENCE EXAMPLES
    # =========================================================================
    print("\n" + "=" * 60)
    print("INFERENCE EXAMPLES - Top High Risk Companies")
    print("=" * 60)
    
    # Get top high risk companies
    sorted_idx = np.argsort(fraud_proba)[::-1]  # Sort descending by fraud probability
    
    print("\n  Rank  GSTIN                       Fraud Prob  Actual  Prediction")
    print("  " + "-" * 70)
    
    for i, idx in enumerate(sorted_idx[:10], 1):
        node = node_list[idx]
        prob = fraud_proba[idx]
        actual = y_true[idx]
        predicted = y_pred[idx]
        actual_str = "FRAUD" if actual == 1 else "Normal"
        pred_str = "FRAUD" if predicted == 1 else "Normal"
        match = "✓" if actual == predicted else "✗"
        print(f"  {i:2d}.   {node:<28} {prob:.4f}     {actual_str:<7} {pred_str:<7} {match}")
    
    # =========================================================================
    # INFERENCE EXAMPLES - Top Low Risk Companies
    # =========================================================================
    print("\n" + "=" * 60)
    print("INFERENCE EXAMPLES - Top Low Risk Companies")
    print("=" * 60)
    
    print("\n  Rank  GSTIN                       Fraud Prob  Actual  Prediction")
    print("  " + "-" * 70)
    
    for i, idx in enumerate(sorted_idx[-10:][::-1], 1):
        node = node_list[idx]
        prob = fraud_proba[idx]
        actual = y_true[idx]
        predicted = y_pred[idx]
        actual_str = "FRAUD" if actual == 1 else "Normal"
        pred_str = "FRAUD" if predicted == 1 else "Normal"
        match = "✓" if actual == predicted else "✗"
        print(f"  {i:2d}.   {node:<28} {prob:.4f}     {actual_str:<7} {pred_str:<7} {match}")
    
    # =========================================================================
    # INFERENCE EXAMPLES - Known Fraud Companies
    # =========================================================================
    print("\n" + "=" * 60)
    print("INFERENCE EXAMPLES - Actual Fraud Companies")
    print("=" * 60)
    
    fraud_indices = np.where(y_true == 1)[0]
    print(f"\n  Total Actual Fraud Companies: {len(fraud_indices)}")
    print(f"  Correctly Predicted as Fraud: {sum(y_pred[fraud_indices] == 1)}")
    print(f"  Missed (False Negatives):     {sum(y_pred[fraud_indices] == 0)}")
    
    # Sort fraud companies by their fraud probability
    fraud_proba_sorted = [(i, fraud_proba[i]) for i in fraud_indices]
    fraud_proba_sorted.sort(key=lambda x: x[1], reverse=True)
    
    print("\n  Rank  GSTIN                       Fraud Prob  Correctly Detected?")
    print("  " + "-" * 70)
    
    for i, (idx, prob) in enumerate(fraud_proba_sorted[:15], 1):
        node = node_list[idx]
        detected = "Yes ✓" if y_pred[idx] == 1 else "No ✗"
        print(f"  {i:2d}.   {node:<28} {prob:.4f}     {detected}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc_roc,
        "confusion_matrix": cm.tolist(),
        "total_companies": total,
        "fraud_count": int(fraud_count),
        "normal_count": int(normal_count)
    }


if __name__ == "__main__":
    results = evaluate_model()
