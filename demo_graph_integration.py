"""
Demo: Build heterogeneous graph from generated datasets
Shows integration of new graph_builder with existing infrastructure
"""
import sys
from pathlib import Path
import pandas as pd

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "tax-fraud-gnn"))
sys.path.insert(0, str(Path(__file__).parent / "tax-fraud-gnn" / "src"))

from graph_construction.build_graph import GraphBuilder
from datetime import datetime


def demo_hetero_graph_with_datasets():
    """
    Demonstrate building heterogeneous graph from new_datasets
    """
    print("=" * 70)
    print("HETEROGENEOUS GRAPH CONSTRUCTION DEMO")
    print("=" * 70)
    
    # Load dataset 01
    datasets_path = Path(__file__).parent / "new_datasets"
    companies_path = datasets_path / "dataset_01_companies_complete.csv"
    invoices_path = datasets_path / "dataset_01_invoices.csv"
    
    print(f"\n📂 Loading data...")
    print(f"   Companies: {companies_path}")
    print(f"   Invoices: {invoices_path}")
    
    companies = pd.read_csv(companies_path)
    invoices = pd.read_csv(invoices_path) if invoices_path.exists() else pd.DataFrame()
    
    print(f"\n📊 Dataset shape:")
    print(f"   Companies: {companies.shape}")
    print(f"   Invoices: {invoices.shape if not invoices.empty else 'N/A (pre-aggregated)'}")
    
    # Build heterogeneous graph
    # Create a temporary output directory
    temp_output = Path(__file__).parent / "temp_graphs"
    temp_output.mkdir(exist_ok=True)
    
    builder = GraphBuilder(processed_data_path=str(temp_output))
    
    print(f"\n🔨 Building heterogeneous graph...")
    hetero_data, networkx_graph = builder.build_hetero_graph(
        companies=companies,
        invoices=invoices if not invoices.empty else None,
        relations=None,
        now=datetime(2024, 12, 9)
    )
    
    print(f"\n✅ HETEROGENEOUS GRAPH BUILT")
    print("=" * 70)
    print(hetero_data)
    print("=" * 70)
    
    # Detailed breakdown
    print(f"\n📈 Node Statistics:")
    print(f"   Company nodes: {hetero_data['company'].x.shape[0]}")
    print(f"   Company features: {hetero_data['company'].x.shape[1]} dimensions")
    print(f"      [degree, avg_invoice_amount, transaction_count, pagerank,")
    print(f"       betweenness, registration_age_days, turnover, sent_invoices,")
    print(f"       received_invoices]")
    
    if hasattr(hetero_data, 'node_types') and 'invoice' in hetero_data.node_types:
        print(f"   Invoice nodes: {hetero_data['invoice'].x.shape[0]}")
        print(f"   Invoice features: {hetero_data['invoice'].x.shape[1]} dimensions")
        print(f"      [amount, age_days]")
    
    print(f"\n📊 Edge Statistics:")
    if ('company', 'transacts', 'invoice') in hetero_data.edge_types:
        print(f"   (company → invoice): {hetero_data['company', 'transacts', 'invoice'].edge_index.shape[1]} edges")
    if ('invoice', 'billed_to', 'company') in hetero_data.edge_types:
        print(f"   (invoice → company): {hetero_data['invoice', 'billed_to', 'company'].edge_index.shape[1]} edges")
    if ('company', 'related', 'company') in hetero_data.edge_types:
        print(f"   (company → company): {hetero_data['company', 'related', 'company'].edge_index.shape[1]} edges")
    
    print(f"\n🎯 Fraud Statistics:")
    fraud_labels = hetero_data['company'].y
    fraud_count = (fraud_labels == 1).sum().item()
    total_companies = fraud_labels.shape[0]
    print(f"   Fraudulent companies: {fraud_count} / {total_companies} ({100*fraud_count/total_companies:.2f}%)")
    
    # Sample features
    print(f"\n🔍 Sample Company Features (first 3 companies):")
    import torch
    sample_feats = hetero_data['company'].x[:3]
    feat_names = ["degree", "avg_inv_amt", "txn_count", "pagerank", "betweenness", 
                  "reg_age", "turnover", "sent_inv", "recv_inv"]
    for i, feats in enumerate(sample_feats):
        print(f"\n   Company {i}:")
        for name, val in zip(feat_names, feats.tolist()):
            print(f"      {name:15s}: {val:12.2f}")
    
    print("\n" + "=" * 70)
    print("✅ Demo complete! Heterogeneous graph ready for GNN training.")
    print("=" * 70)
    
    return hetero_data, networkx_graph


def demo_homogeneous_fallback():
    """
    Show existing homogeneous graph construction still works
    """
    print("\n" + "=" * 70)
    print("HOMOGENEOUS GRAPH (BACKWARD COMPATIBILITY)")
    print("=" * 70)
    
    datasets_path = Path(__file__).parent / "new_datasets"
    companies_path = datasets_path / "dataset_01_companies_complete.csv"
    
    companies = pd.read_csv(companies_path)
    
    # Mock invoices from company features (since data is pre-aggregated)
    mock_invoices = pd.DataFrame({
        "invoice_id": [f"I{i}" for i in range(100)],
        "seller_id": companies.sample(100, replace=True)["company_id"].values,
        "buyer_id": companies.sample(100, replace=True)["company_id"].values,
        "amount": [1000 * (i+1) for i in range(100)],
        "itc_claimed": [100 * (i+1) for i in range(100)]
    })
    
    print(f"\n🔨 Building homogeneous graph (existing method)...")
    # Use temp directory
    temp_output = Path(__file__).parent / "temp_graphs"
    builder = GraphBuilder(processed_data_path=str(temp_output))
    G = builder.build_networkx_graph(companies, mock_invoices)
    data, node_list, node_to_idx = builder.networkx_to_pytorch_geometric(G, companies)
    
    print(f"\n✅ HOMOGENEOUS GRAPH BUILT")
    print(f"   Nodes: {data.num_nodes}")
    print(f"   Edges: {data.num_edges}")
    print(f"   Node features: {data.x.shape} (3D: turnover, sent_invoices, received_invoices)")
    print(f"   Edge features: {data.edge_attr.shape if hasattr(data, 'edge_attr') else 'N/A'}")
    
    print("\n" + "=" * 70)
    
    return data, G


if __name__ == "__main__":
    # Run both demos
    print("\n🚀 STARTING GRAPH CONSTRUCTION DEMOS\n")
    
    # Demo 1: New heterogeneous graph
    hetero_data, hetero_nx = demo_hetero_graph_with_datasets()
    
    # Demo 2: Existing homogeneous graph (backward compatibility)
    homo_data, homo_nx = demo_homogeneous_fallback()
    
    print("\n✅ All demos completed successfully!")
    print("\nNext steps:")
    print("  1. Train GNN model on heterogeneous graph with HeteroConv layers")
    print("  2. Use incremental learning to update graph with new datasets")
    print("  3. Deploy fraud detection scoring API")
