"""
Graph Construction Module
Builds NetworkX and PyTorch Geometric graph representations
Supports both homogeneous and heterogeneous graph formats
"""

import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data, HeteroData
from pathlib import Path
import logging
import pickle
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GraphBuilder:
    """Construct and convert transaction graphs to GNN-compatible formats"""
    
    def __init__(self, processed_data_path="../../data/processed"):
        self.data_path = Path(processed_data_path)
        self.graph_path = self.data_path / "graphs"
        self.graph_path.mkdir(exist_ok=True)
        logger.info(f"GraphBuilder initialized. Data path: {self.data_path}")
    
    def load_processed_data(self):
        """Load cleaned data"""
        try:
            companies = pd.read_csv(self.data_path / "companies_processed.csv")
            invoices = pd.read_csv(self.data_path / "invoices_processed.csv")
            logger.info(f"Loaded {len(companies)} companies and {len(invoices)} invoices")
            return companies, invoices
        except FileNotFoundError as e:
            logger.error(f"Processed data files not found: {e}")
            raise
    
    def build_networkx_graph(self, companies, invoices):
        """
        Build NetworkX directed graph:
        - Nodes: Companies with attributes (turnover, location, fraud label, etc.)
        - Edges: Invoices with attributes (amount, ITC claimed)
        """
        G = nx.DiGraph()
        logger.info("Building NetworkX directed graph...")
        
        # Add nodes with attributes
        for _, row in companies.iterrows():
            try:
                G.add_node(
                    int(row["company_id"]),
                    turnover=float(row.get("turnover", 0)),
                    location=str(row.get("location", "Unknown")),
                    is_fraud=int(row.get("is_fraud", 0)),
                    sent_invoices=float(row.get("sent_invoice_count", 0)),
                    received_invoices=float(row.get("received_invoice_count", 0)),
                    total_sent_amount=float(row.get("total_sent_amount", 0)),
                    total_received_amount=float(row.get("total_received_amount", 0))
                )
            except Exception as e:
                logger.warning(f"Error adding node {row.get('company_id')}: {e}")
                continue
        
        logger.info(f"Added {G.number_of_nodes()} nodes")
        
        # Add edges with attributes
        for _, row in invoices.iterrows():
            try:
                seller = int(row["seller_id"])
                buyer = int(row["buyer_id"])
                
                # Only add edge if both nodes exist
                if seller in G and buyer in G:
                    G.add_edge(
                        seller,
                        buyer,
                        amount=float(row.get("amount", 0)),
                        itc_claimed=float(row.get("itc_claimed", 0))
                    )
            except Exception as e:
                logger.warning(f"Error adding edge: {e}")
                continue
        
        logger.info(f"Added {G.number_of_edges()} edges")
        return G
    
    def networkx_to_pytorch_geometric(self, G, companies):
        """
        Convert NetworkX graph to PyTorch Geometric Data object
        - Node features: turnover, sent_invoices, received_invoices
        - Edge indices: directed edges from invoice transactions
        - Edge attributes: invoice amount
        """
        logger.info("Converting to PyTorch Geometric format...")
        
        # Create node list and feature matrix
        node_list = sorted(list(G.nodes()))
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        x_list = []
        y_list = []
        
        for node in node_list:
            node_data = G.nodes[node]
            features = [
                node_data.get("turnover", 0),
                node_data.get("sent_invoices", 0),
                node_data.get("received_invoices", 0)
            ]
            x_list.append(features)
            y_list.append(node_data.get("is_fraud", 0))
        
        x = torch.tensor(x_list, dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.long)
        
        logger.info(f"Node feature matrix shape: {x.shape}")
        
        # Create edge indices and attributes
        edge_index = []
        edge_attr = []
        
        for u, v, data in G.edges(data=True):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            edge_index.append([u_idx, v_idx])
            edge_attr.append([data.get("amount", 0)])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float32)
        
        logger.info(f"Edge index shape: {edge_index.shape}")
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            node_ids=torch.tensor(node_list, dtype=torch.long)
        )
        
        return data, node_list, node_to_idx
    
    def compute_graph_statistics(self, G, data):
        """Compute and log graph statistics"""
        logger.info("=" * 60)
        logger.info("GRAPH STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Number of nodes: {G.number_of_nodes()}")
        logger.info(f"Number of edges: {G.number_of_edges()}")
        logger.info(f"Graph density: {nx.density(G):.4f}")
        
        # Degree statistics
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        logger.info(f"Average in-degree: {np.mean(in_degrees):.2f}")
        logger.info(f"Average out-degree: {np.mean(out_degrees):.2f}")
        
        # Fraud statistics
        fraud_nodes = sum(1 for n in G.nodes() if G.nodes[n].get("is_fraud") == 1)
        logger.info(f"Fraudulent companies: {fraud_nodes} ({100*fraud_nodes/G.number_of_nodes():.2f}%)")
        logger.info("=" * 60)
    
    def build_and_save(self):
        """Execute complete graph building pipeline"""
        try:
            logger.info("=" * 60)
            logger.info("STARTING GRAPH CONSTRUCTION PIPELINE")
            logger.info("=" * 60)
            
            # Load data
            companies, invoices = self.load_processed_data()
            
            # Build NetworkX graph
            G = self.build_networkx_graph(companies, invoices)
            
            # Convert to PyTorch Geometric
            data, node_list, node_to_idx = self.networkx_to_pytorch_geometric(G, companies)
            
            # Compute statistics
            self.compute_graph_statistics(G, data)
            
            # Save graph objects
            logger.info("Saving graph files...")
            torch.save(data, self.graph_path / "graph_data.pt")
            logger.info(f"✓ PyTorch Geometric graph: {self.graph_path / 'graph_data.pt'}")
            
            # Save mappings
            mappings = {
                "node_list": node_list,
                "node_to_idx": node_to_idx
            }
            with open(self.graph_path / "node_mappings.pkl", "wb") as f:
                pickle.dump(mappings, f)
            logger.info(f"✓ Node mappings: {self.graph_path / 'node_mappings.pkl'}")
            
            # Save NetworkX graph
            nx.write_gpickle(G, self.graph_path / "networkx_graph.gpickle")
            logger.info(f"✓ NetworkX graph: {self.graph_path / 'networkx_graph.gpickle'}")
            
            logger.info("=" * 60)
            logger.info("✅ GRAPH CONSTRUCTION COMPLETE")
            logger.info("=" * 60)
            
            return data, G, node_list, node_to_idx
            
        except Exception as e:
            logger.error(f"Error during graph construction: {e}")
            raise
    
    def build_hetero_graph(self, companies, invoices, relations=None, now=None):
        """
        Build heterogeneous graph with companies and invoices as separate node types.
        Integrates the new graph_builder.py logic into existing infrastructure.
        
        Args:
            companies: DataFrame with company data
            invoices: DataFrame with invoice data (seller_id, buyer_id, amount, etc.)
            relations: Optional DataFrame with company-company relations
            now: Reference datetime for age calculations
        
        Returns:
            HeteroData object with typed nodes and edges
        """
        from datetime import datetime
        
        logger.info("Building heterogeneous graph...")
        now_ts = pd.Timestamp(now or datetime.utcnow()).normalize()
        
        # Create MultiDiGraph
        G = nx.MultiDiGraph()
        
        # Add company nodes
        for _, row in companies.iterrows():
            company_id = str(row["company_id"])
            reg_date = pd.to_datetime(row.get("registration_date", pd.NaT))
            reg_age_days = int((now_ts - reg_date).days) if pd.notna(reg_date) else 0
            
            G.add_node(
                company_id,
                node_type="company",
                turnover=float(row.get("avg_monthly_turnover", row.get("turnover", 0))),
                registration_age_days=reg_age_days,
                sent_invoices=float(row.get("sent_invoice_count", 0)),
                received_invoices=float(row.get("received_invoice_count", 0)),
                total_sent_amount=float(row.get("total_sent_amount", 0)),
                total_received_amount=float(row.get("total_received_amount", 0)),
                is_fraud=int(row.get("is_fraud", 0))
            )
        
        # Add invoice nodes and transaction edges if invoices provided
        if invoices is not None and len(invoices) > 0:
            for _, row in invoices.iterrows():
                inv_id = str(row["invoice_id"])
                inv_date = pd.to_datetime(row.get("invoice_date", pd.NaT))
                inv_age_days = int((now_ts - inv_date).days) if pd.notna(inv_date) else 0
                amount = float(row.get("amount", 0))
                
                G.add_node(
                    inv_id,
                    node_type="invoice",
                    amount=amount,
                    age_days=inv_age_days
                )
                
                # seller -> invoice -> buyer edges
                seller = str(row["seller_id"])
                buyer = str(row["buyer_id"])
                
                if seller in G.nodes():
                    G.add_edge(seller, inv_id, relation_type="transaction", amount=amount, date=inv_date, age_days=inv_age_days)
                if buyer in G.nodes():
                    G.add_edge(inv_id, buyer, relation_type="transaction", amount=amount, date=inv_date, age_days=inv_age_days)
        
        # Add company-company relations if provided
        if relations is not None and len(relations) > 0:
            for _, row in relations.iterrows():
                src = str(row["src_company_id"])
                dst = str(row["dst_company_id"])
                if src in G.nodes() and dst in G.nodes():
                    G.add_edge(src, dst, relation_type=row.get("relation_type", "ownership"), amount=0.0, date=pd.NaT)
        
        # Compute company features (centrality, etc.)
        company_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "company"]
        company_features = self._compute_hetero_company_features(G, company_nodes)
        
        # Convert to HeteroData
        hetero_data = self._to_heterodata(G, company_features)
        
        logger.info(f"Hetero graph built: {len(company_nodes)} companies, {hetero_data['company'].x.shape}")
        return hetero_data, G
    
    def _compute_hetero_company_features(self, G, company_nodes):
        """Compute features for companies in heterogeneous graph"""
        # Transaction counts and amounts
        txn_amounts = {n: [] for n in company_nodes}
        txn_counts = {n: 0 for n in company_nodes}
        
        for u, v, d in G.edges(data=True):
            if d.get("relation_type") == "transaction":
                amt = float(d.get("amount", 0.0))
                if u in txn_counts:
                    txn_counts[u] += 1
                    txn_amounts[u].append(amt)
                if v in txn_counts:
                    txn_counts[v] += 1
                    txn_amounts[v].append(amt)
        
        avg_amounts = {n: (float(sum(vals)) / len(vals) if vals else 0.0) for n, vals in txn_amounts.items()}
        
        # Centrality on company subgraph
        company_sub = G.subgraph(company_nodes).to_undirected()
        degrees = dict(G.degree(company_nodes))
        pagerank = nx.pagerank(company_sub) if company_sub.number_of_nodes() > 0 else {}
        betweenness = nx.betweenness_centrality(company_sub) if company_sub.number_of_nodes() > 0 else {}
        
        features = {}
        for n in company_nodes:
            node_data = G.nodes[n]
            features[n] = {
                "degree": float(degrees.get(n, 0)),
                "avg_invoice_amount": float(avg_amounts.get(n, 0.0)),
                "transaction_count": float(txn_counts.get(n, 0)),
                "pagerank": float(pagerank.get(n, 0.0)),
                "betweenness": float(betweenness.get(n, 0.0)),
                "registration_age_days": float(node_data.get("registration_age_days", 0)),
                "turnover": float(node_data.get("turnover", 0)),
                "sent_invoices": float(node_data.get("sent_invoices", 0)),
                "received_invoices": float(node_data.get("received_invoices", 0))
            }
        return features
    
    def _to_heterodata(self, G, company_features):
        """Convert NetworkX multigraph to HeteroData"""
        data = HeteroData()
        
        company_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "company"]
        invoice_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "invoice"]
        
        company_index = {n: i for i, n in enumerate(company_nodes)}
        invoice_index = {n: i for i, n in enumerate(invoice_nodes)}
        
        # Company node features (9 features)
        feat_order = ["degree", "avg_invoice_amount", "transaction_count", "pagerank", "betweenness", 
                      "registration_age_days", "turnover", "sent_invoices", "received_invoices"]
        company_x = [[company_features[n][k] for k in feat_order] for n in company_nodes]
        data["company"].x = torch.tensor(company_x, dtype=torch.float) if company_x else torch.empty((0, 9))
        
        # Company labels
        company_y = [G.nodes[n].get("is_fraud", 0) for n in company_nodes]
        data["company"].y = torch.tensor(company_y, dtype=torch.long)
        
        # Invoice node features (2 features)
        invoice_x = [[float(G.nodes[n].get("amount", 0.0)), float(G.nodes[n].get("age_days", 0.0))] for n in invoice_nodes]
        data["invoice"].x = torch.tensor(invoice_x, dtype=torch.float) if invoice_x else torch.empty((0, 2))
        
        # Helper to build edge blocks
        def edge_block(src_nodes, dst_nodes, src_index, dst_index, edge_filter):
            edge_index = []
            edge_attr = []
            for u, v, edata in edge_filter:
                if u in src_index and v in dst_index:
                    edge_index.append([src_index[u], dst_index[v]])
                    edge_attr.append([
                        float(edata.get("amount", 0.0)),
                        float(edata.get("age_days", 0.0)),
                        float(0 if edata.get("relation_type") == "transaction" else 1)
                    ])
            if not edge_index:
                return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 3), dtype=torch.float)
            return torch.tensor(edge_index, dtype=torch.long).t().contiguous(), torch.tensor(edge_attr, dtype=torch.float)
        
        # company -> invoice edges
        seller_edges = [(u, v, d) for u, v, d in G.edges(data=True) 
                        if G.nodes.get(u, {}).get("node_type") == "company" and 
                        G.nodes.get(v, {}).get("node_type") == "invoice"]
        edge_index, edge_attr = edge_block(company_nodes, invoice_nodes, company_index, invoice_index, seller_edges)
        data["company", "transacts", "invoice"].edge_index = edge_index
        data["company", "transacts", "invoice"].edge_attr = edge_attr
        
        # invoice -> company edges
        buyer_edges = [(u, v, d) for u, v, d in G.edges(data=True)
                       if G.nodes.get(u, {}).get("node_type") == "invoice" and
                       G.nodes.get(v, {}).get("node_type") == "company"]
        edge_index, edge_attr = edge_block(invoice_nodes, company_nodes, invoice_index, company_index, buyer_edges)
        data["invoice", "billed_to", "company"].edge_index = edge_index
        data["invoice", "billed_to", "company"].edge_attr = edge_attr
        
        # company -> company edges
        relation_edges = [(u, v, d) for u, v, d in G.edges(data=True)
                          if G.nodes.get(u, {}).get("node_type") == "company" and
                          G.nodes.get(v, {}).get("node_type") == "company"]
        edge_index, edge_attr = edge_block(company_nodes, company_nodes, company_index, company_index, relation_edges)
        data["company", "related", "company"].edge_index = edge_index
        data["company", "related", "company"].edge_attr = edge_attr
        
        return data


if __name__ == "__main__":
    builder = GraphBuilder()
    data, G, node_list, node_to_idx = builder.build_and_save()
    print(f"\nPyTorch Geometric Data:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Node features: {data.x.shape}")

    print(f"  Edge features: {data.edge_attr.shape}")
