"""
Graph construction utility.
- Reads companies.csv, invoices.csv, relations.csv
- Builds a NetworkX MultiDiGraph with companies and invoices as nodes
- Computes company-level features and converts to PyTorch Geometric HeteroData
"""
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import HeteroData
from datetime import datetime
from typing import Tuple, Dict, List


def _safe_age_days(now_ts: pd.Timestamp, maybe_date) -> int:
    if pd.isna(maybe_date):
        return 0
    return int((now_ts - pd.to_datetime(maybe_date)).days)


def read_inputs(
    companies_path: str,
    invoices_path: str,
    relations_path: str,
    now: datetime | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    now_ts = pd.Timestamp(now or datetime.utcnow()).normalize()
    companies = pd.read_csv(companies_path, parse_dates=["registration_date"])
    invoices = pd.read_csv(invoices_path, parse_dates=["invoice_date"])
    relations = pd.read_csv(relations_path)
    return companies, invoices, relations, now_ts


def build_networkx_graph(
    companies: pd.DataFrame,
    invoices: pd.DataFrame,
    relations: pd.DataFrame,
    now_ts: pd.Timestamp,
) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    # Company nodes
    for row in companies.itertuples(index=False):
        reg_age_days = _safe_age_days(now_ts, getattr(row, "registration_date", pd.NaT))
        G.add_node(
            row.company_id,
            node_type="company",
            registration_date=getattr(row, "registration_date", pd.NaT),
            registration_age_days=reg_age_days,
        )

    # Invoice nodes + transaction edges (seller->invoice->buyer)
    for row in invoices.itertuples(index=False):
        inv_age_days = _safe_age_days(now_ts, getattr(row, "invoice_date", pd.NaT))
        G.add_node(
            row.invoice_id,
            node_type="invoice",
            amount=float(getattr(row, "amount", 0.0)),
            invoice_date=getattr(row, "invoice_date", pd.NaT),
            age_days=inv_age_days,
        )
        G.add_edge(
            getattr(row, "seller_id"),
            row.invoice_id,
            relation_type="transaction",
            amount=float(getattr(row, "amount", 0.0)),
            date=getattr(row, "invoice_date", pd.NaT),
        )
        G.add_edge(
            row.invoice_id,
            getattr(row, "buyer_id"),
            relation_type="transaction",
            amount=float(getattr(row, "amount", 0.0)),
            date=getattr(row, "invoice_date", pd.NaT),
        )

    # Ownership/other relations between companies
    for row in relations.itertuples(index=False):
        G.add_edge(
            getattr(row, "src_company_id"),
            getattr(row, "dst_company_id"),
            relation_type=getattr(row, "relation_type", "ownership"),
            amount=0.0,
            date=pd.NaT,
        )

    # Precompute edge age and relation ids for later use
    for _, _, data in G.edges(data=True):
        data["age_days"] = _safe_age_days(now_ts, data.get("date", pd.NaT))
        data["relation_id"] = 0 if data.get("relation_type") == "transaction" else 1

    return G


def compute_company_features(G: nx.MultiDiGraph) -> Dict[str, Dict[str, float]]:
    company_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "company"]
    transaction_edges = [
        (u, v, d) for u, v, d in G.edges(data=True) if d.get("relation_type") == "transaction"
    ]

    # Degree based on full multigraph
    degrees = dict(G.degree(company_nodes))

    # Transaction counts and avg amounts
    txn_amounts: Dict[str, List[float]] = {n: [] for n in company_nodes}
    txn_counts: Dict[str, int] = {n: 0 for n in company_nodes}
    for u, v, d in transaction_edges:
        amt = float(d.get("amount", 0.0))
        if u in txn_counts:
            txn_counts[u] += 1
            txn_amounts[u].append(amt)
        if v in txn_counts:
            txn_counts[v] += 1
            txn_amounts[v].append(amt)

    avg_amounts = {
        n: (float(sum(vals)) / len(vals) if vals else 0.0) for n, vals in txn_amounts.items()
    }

    # Centrality on undirected projection of company nodes
    company_sub = G.subgraph(company_nodes).to_undirected()
    pagerank = nx.pagerank(company_sub) if company_sub.number_of_nodes() > 0 else {}
    betweenness = (
        nx.betweenness_centrality(company_sub) if company_sub.number_of_nodes() > 0 else {}
    )

    features: Dict[str, Dict[str, float]] = {}
    for n in company_nodes:
        node_data = G.nodes[n]
        features[n] = {
            "degree": float(degrees.get(n, 0)),
            "avg_invoice_amount": float(avg_amounts.get(n, 0.0)),
            "transaction_count": float(txn_counts.get(n, 0)),
            "pagerank": float(pagerank.get(n, 0.0)),
            "betweenness": float(betweenness.get(n, 0.0)),
            "registration_age_days": float(node_data.get("registration_age_days", 0)),
        }
    return features


def to_heterodata(G: nx.MultiDiGraph, company_features: Dict[str, Dict[str, float]]) -> HeteroData:
    data = HeteroData()
    company_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "company"]
    invoice_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "invoice"]

    company_index = {n: i for i, n in enumerate(company_nodes)}
    invoice_index = {n: i for i, n in enumerate(invoice_nodes)}

    company_feat_order = [
        "degree",
        "avg_invoice_amount",
        "transaction_count",
        "pagerank",
        "betweenness",
        "registration_age_days",
    ]
    company_x = [
        [company_features[n][k] for k in company_feat_order]
        for n in company_nodes
    ]
    data["company"].x = torch.tensor(company_x, dtype=torch.float) if company_x else torch.empty((0, 6))

    invoice_x = []
    for n in invoice_nodes:
        nd = G.nodes[n]
        invoice_x.append([float(nd.get("amount", 0.0)), float(nd.get("age_days", 0.0))])
    data["invoice"].x = torch.tensor(invoice_x, dtype=torch.float) if invoice_x else torch.empty((0, 2))

    # Edges: company -> invoice (seller), invoice -> company (buyer), company -> company (relation)
    def edge_block(src_nodes, dst_nodes, src_index, dst_index, edge_filter):
        edge_index = []
        edge_attr = []
        for u, v, edata in edge_filter:
            edge_index.append([src_index[u], dst_index[v]])
            edge_attr.append([
                float(edata.get("amount", 0.0)),
                float(edata.get("age_days", 0.0)),
                float(edata.get("relation_id", 0)),
            ])
        if not edge_index:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 3), dtype=torch.float)
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous(), torch.tensor(edge_attr, dtype=torch.float)

    seller_edges = [
        (u, v, d)
        for u, v, d in G.edges(data=True)
        if G.nodes[u].get("node_type") == "company" and G.nodes[v].get("node_type") == "invoice"
    ]
    buyer_edges = [
        (u, v, d)
        for u, v, d in G.edges(data=True)
        if G.nodes[u].get("node_type") == "invoice" and G.nodes[v].get("node_type") == "company"
    ]
    relation_edges = [
        (u, v, d)
        for u, v, d in G.edges(data=True)
        if G.nodes[u].get("node_type") == "company" and G.nodes[v].get("node_type") == "company"
    ]

    edge_index, edge_attr = edge_block(company_nodes, invoice_nodes, company_index, invoice_index, seller_edges)
    data["company", "transacts", "invoice"].edge_index = edge_index
    data["company", "transacts", "invoice"].edge_attr = edge_attr

    edge_index, edge_attr = edge_block(invoice_nodes, company_nodes, invoice_index, company_index, buyer_edges)
    data["invoice", "billed_to", "company"].edge_index = edge_index
    data["invoice", "billed_to", "company"].edge_attr = edge_attr

    edge_index, edge_attr = edge_block(company_nodes, company_nodes, company_index, company_index, relation_edges)
    data["company", "related", "company"].edge_index = edge_index
    data["company", "related", "company"].edge_attr = edge_attr

    return data


def build_pyg_data(
    companies_path: str,
    invoices_path: str,
    relations_path: str,
    now: datetime | None = None,
) -> HeteroData:
    companies, invoices, relations, now_ts = read_inputs(
        companies_path, invoices_path, relations_path, now=now
    )
    G = build_networkx_graph(companies, invoices, relations, now_ts)
    company_features = compute_company_features(G)
    return to_heterodata(G, company_features)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build hetero graph for GNN training")
    parser.add_argument("companies", help="Path to companies.csv")
    parser.add_argument("invoices", help="Path to invoices.csv")
    parser.add_argument("relations", help="Path to relations.csv")
    args = parser.parse_args()

    hetero = build_pyg_data(args.companies, args.invoices, args.relations)
    print(hetero)
