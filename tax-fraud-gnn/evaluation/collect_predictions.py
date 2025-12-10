"""
Prediction Collection Script for GNN Fraud Detection Evaluation

This script systematically collects model predictions for all datasets
and saves them in a format suitable for evaluation.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gnn_models.train_gnn import GNNFraudDetector


# Number of features (must match model's in_channels)
NUM_FEATURES = 12

def load_model(model_path: Path, device: torch.device, in_channels: int = 12):
    """Load trained GNN model with 12 input features."""
    model = GNNFraudDetector(in_channels=in_channels, hidden_channels=64, out_channels=2, model_type="gcn")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def prepare_features(companies_df: pd.DataFrame, invoices_df: pd.DataFrame = None) -> torch.Tensor:
    """
    Prepare feature tensor from company DataFrame.
    Now uses 12 EXTENDED FEATURES for better fraud detection.
    """
    companies_df = companies_df.copy()
    
    # If invoices provided, compute aggregated features
    if invoices_df is not None and len(invoices_df) > 0:
        companies_df["company_id"] = companies_df["company_id"].astype(str)
        
        invoices_df = invoices_df.copy()
        invoices_df["seller_id"] = invoices_df["seller_id"].astype(str)
        invoices_df["buyer_id"] = invoices_df["buyer_id"].astype(str)
        
        # Seller stats
        seller_stats = invoices_df.groupby("seller_id").agg({
            "amount": "sum",
            "invoice_id": "count"
        }).reset_index()
        seller_stats.columns = ["company_id", "sent_amount", "sent_invoices"]
        
        # Buyer stats
        buyer_stats = invoices_df.groupby("buyer_id").agg({
            "amount": "sum",
            "invoice_id": "count"
        }).reset_index()
        buyer_stats.columns = ["company_id", "received_amount", "received_invoices"]
        
        # Unique partners
        unique_buyers = invoices_df.groupby("seller_id")["buyer_id"].nunique().reset_index()
        unique_buyers.columns = ["company_id", "unique_buyers_computed"]
        
        unique_sellers = invoices_df.groupby("buyer_id")["seller_id"].nunique().reset_index()
        unique_sellers.columns = ["company_id", "unique_sellers_computed"]
        
        # Merge features
        companies_df = companies_df.merge(seller_stats, on="company_id", how="left")
        companies_df = companies_df.merge(buyer_stats, on="company_id", how="left")
        companies_df = companies_df.merge(unique_buyers, on="company_id", how="left")
        companies_df = companies_df.merge(unique_sellers, on="company_id", how="left")
        companies_df.fillna(0, inplace=True)
        
        # Compute turnover
        companies_df["turnover"] = companies_df.get("sent_amount", 0) + companies_df.get("received_amount", 0)
    
    features = []
    for _, row in companies_df.iterrows():
        # 12 features matching the model
        feat = [
            # Basic features (5)
            float(row.get("turnover", row.get("avg_monthly_turnover", 0))),
            float(row.get("sent_invoices", row.get("sent_invoice_count", row.get("total_invoices_sent", 0)))),
            float(row.get("received_invoices", row.get("received_invoice_count", row.get("total_invoices_received", 0)))),
            float(row.get("sent_amount", row.get("total_amount_sent", row.get("total_sent_amount", 0)))),
            float(row.get("received_amount", row.get("total_amount_received", row.get("total_received_amount", 0)))),
            # Extended features (7)
            float(row.get("unique_buyers_computed", row.get("unique_buyers", 0))),
            float(row.get("unique_sellers_computed", row.get("unique_sellers", 0))),
            float(row.get("circular_trading_score", 0)),
            float(row.get("gst_compliance_rate", 1.0)),
            float(row.get("late_filing_count", 0)),
            float(row.get("round_amount_ratio", 0)),
            float(row.get("buyer_concentration", 0))
        ]
        features.append(feat)
    
    x = torch.tensor(features, dtype=torch.float32)
    
    # Normalize features to [0, 1] range
    x_min = x.min(dim=0, keepdim=True)[0]
    x_max = x.max(dim=0, keepdim=True)[0]
    x_range = x_max - x_min
    x_range[x_range == 0] = 1  # Avoid division by zero
    x = (x - x_min) / x_range
    
    return x


def build_edge_index(companies_df: pd.DataFrame, invoices_df: pd.DataFrame = None) -> torch.Tensor:
    """Build edge index from invoice data or create self-loops if no invoices."""
    company_ids = companies_df["company_id"].astype(str).tolist()
    id_to_idx = {cid: idx for idx, cid in enumerate(company_ids)}
    
    edges = []
    
    if invoices_df is not None and len(invoices_df) > 0:
        for _, row in invoices_df.iterrows():
            seller = str(row["seller_id"])
            buyer = str(row["buyer_id"])
            if seller in id_to_idx and buyer in id_to_idx:
                edges.append([id_to_idx[seller], id_to_idx[buyer]])
    
    # Add self-loops if no edges or to ensure connectivity
    for idx in range(len(company_ids)):
        edges.append([idx, idx])
    
    if edges:
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.zeros((2, 0), dtype=torch.long)


def collect_predictions(
    model: torch.nn.Module,
    companies_df: pd.DataFrame,
    invoices_df: pd.DataFrame = None,
    device: torch.device = None
) -> pd.DataFrame:
    """
    Run model inference and collect predictions for all companies.
    
    Returns DataFrame with:
    - company_id
    - ground_truth (is_fraud)
    - predicted_class (0 or 1)
    - fraud_probability (confidence score)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data
    x = prepare_features(companies_df, invoices_df).to(device)
    edge_index = build_edge_index(companies_df, invoices_df).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        probs = torch.softmax(out, dim=1)
        fraud_proba = probs[:, 1].cpu().numpy()
        predictions = (fraud_proba > 0.5).astype(int)
    
    # Build results DataFrame
    results = pd.DataFrame({
        "company_id": companies_df["company_id"].astype(str).values,
        "ground_truth": companies_df["is_fraud"].astype(int).values,
        "predicted_class": predictions,
        "fraud_probability": fraud_proba
    })
    
    return results


def collect_all_datasets(
    datasets_path: Path,
    model_path: Path,
    output_path: Path
):
    """
    Collect predictions for all datasets and save to JSONL format.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = load_model(model_path, device)
    
    all_predictions = []
    
    # Find all dataset files
    company_files = sorted(datasets_path.glob("dataset_*_companies_complete.csv"))
    
    for company_file in company_files:
        dataset_num = company_file.stem.split("_")[1]  # e.g., "01"
        invoice_file = datasets_path / f"dataset_{dataset_num}_invoices.csv"
        
        print(f"\nProcessing dataset {dataset_num}...")
        
        # Load data
        companies_df = pd.read_csv(company_file)
        invoices_df = pd.read_csv(invoice_file) if invoice_file.exists() else None
        
        print(f"  Companies: {len(companies_df)}")
        if invoices_df is not None:
            print(f"  Invoices: {len(invoices_df)}")
        
        # Collect predictions
        results = collect_predictions(model, companies_df, invoices_df, device)
        results["dataset"] = f"dataset_{dataset_num}"
        
        all_predictions.append(results)
        
        # Print quick stats
        accuracy = (results["predicted_class"] == results["ground_truth"]).mean()
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Combine all predictions
    combined = pd.concat(all_predictions, ignore_index=True)
    
    # Save to JSONL format (required by Azure AI Evaluation)
    output_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path / "predictions.jsonl"
    
    with open(jsonl_path, "w") as f:
        for _, row in combined.iterrows():
            record = {
                "company_id": row["company_id"],
                "ground_truth": int(row["ground_truth"]),
                "response": int(row["predicted_class"]),  # For evaluator compatibility
                "fraud_probability": float(row["fraud_probability"]),
                "dataset": row["dataset"]
            }
            f.write(json.dumps(record) + "\n")
    
    print(f"\n✅ Saved {len(combined)} predictions to {jsonl_path}")
    
    # Also save as CSV for easier viewing
    csv_path = output_path / "predictions.csv"
    combined.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV copy to {csv_path}")
    
    return combined


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent.parent
    datasets_path = base_path / "new_datasets"
    model_path = Path(__file__).parent.parent / "models" / "best_model.pt"
    output_path = Path(__file__).parent / "data"
    
    print("=" * 60)
    print("GNN FRAUD DETECTION - PREDICTION COLLECTION")
    print("=" * 60)
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first or specify correct path.")
        sys.exit(1)
    
    collect_all_datasets(datasets_path, model_path, output_path)
