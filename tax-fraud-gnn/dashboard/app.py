"""
Streamlit Dashboard for Tax Fraud Detection
Interactive visualization and fraud risk analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import torch
import pickle
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn_models.train_gnn import GNNFraudDetector

# Page config
st.set_page_config(page_title="Tax Fraud Detection Dashboard", layout="wide")

@st.cache_resource
def load_data_and_model():
    """Load processed data and trained model"""
    data_path = Path(__file__).parent.parent / "data" / "processed"
    models_path = Path(__file__).parent.parent / "models"
    
    # Load data
    companies = pd.read_csv(data_path / "companies_processed.csv")
    invoices = pd.read_csv(data_path / "invoices_processed.csv")
    
    # Load graph
    graph_data = torch.load(data_path / "graphs" / "graph_data.pt")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNNFraudDetector(in_channels=3, hidden_channels=64, out_channels=2, model_type="gcn").to(device)
    try:
        model.load_state_dict(torch.load(models_path / "best_model.pt", map_location=device))
    except:
        st.warning("Model weights not found. Using untrained model.")
    
    # Load node mappings
    with open(data_path / "graphs" / "node_mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    
    return companies, invoices, graph_data, model, device, mappings


def get_fraud_predictions(model, graph_data, device):
    """Get fraud predictions from model"""
    model.eval()
    with torch.no_grad():
        out = model(graph_data.x.to(device), graph_data.edge_index.to(device))
        predictions = torch.softmax(out, dim=1)
        fraud_proba = predictions[:, 1].cpu().numpy()
    return fraud_proba


# ============================================================================
# MAIN APP
# ============================================================================

st.title("🚨 Tax Fraud Detection Dashboard")
st.markdown("Graph Neural Network-based fraud detection for invoice networks")

# Load data
companies, invoices, graph_data, model, device, mappings = load_data_and_model()
fraud_proba = get_fraud_predictions(model, graph_data, device)

# Add predictions to companies dataframe
companies["fraud_probability"] = fraud_proba
companies["predicted_fraud"] = (fraud_proba > 0.5).astype(int)

# ============================================================================
# SIDEBAR FILTERS
# ============================================================================

with st.sidebar:
    st.header("🎯 Filters")
    
    risk_threshold = st.slider(
        "Fraud Risk Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Companies with fraud probability above this are flagged as high-risk"
    )
    
    location_filter = st.multiselect(
        "Filter by Location",
        options=companies["location"].unique(),
        default=companies["location"].unique()
    )
    
    st.divider()
    st.markdown("**Model Info**")
    st.metric("Total Companies", len(companies))
    st.metric("High-Risk Companies", (fraud_proba > risk_threshold).sum())
    st.metric("Flagged as Fraud", (companies["predicted_fraud"] == 1).sum())

# ============================================================================
# MAIN CONTENT
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Detailed Analysis", "⚠️ Risk Scoring", "📈 Network Insights"])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    
    filtered_companies = companies[companies["location"].isin(location_filter)]
    
    with col1:
        st.metric(
            "Total Companies",
            len(filtered_companies),
            delta=None,
            delta_color="normal"
        )
    
    with col2:
        high_risk_count = (filtered_companies["fraud_probability"] > risk_threshold).sum()
        st.metric(
            "High-Risk Companies",
            high_risk_count,
            delta=f"{100*high_risk_count/len(filtered_companies):.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        fraud_count = (filtered_companies["predicted_fraud"] == 1).sum()
        st.metric(
            "Predicted Fraud",
            fraud_count,
            delta=None,
            delta_color="inverse"
        )
    
    with col4:
        avg_risk = filtered_companies["fraud_probability"].mean()
        st.metric(
            "Avg Risk Score",
            f"{avg_risk:.2%}",
            delta=None
        )
    
    st.divider()
    
    # Fraud distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        fraud_dist = filtered_companies["predicted_fraud"].value_counts()
        fig = go.Figure(data=[
            go.Bar(
                x=["Normal", "Fraud"],
                y=fraud_dist.values,
                marker=dict(color=["green", "red"])
            )
        ])
        fig.update_layout(
            title="Predicted Fraud Distribution",
            xaxis_title="Status",
            yaxis_title="Count",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk score distribution
        fig = go.Figure(data=[
            go.Histogram(
                x=filtered_companies["fraud_probability"],
                nbinsx=30,
                marker=dict(color="blue")
            )
        ])
        fig.update_layout(
            title="Fraud Probability Distribution",
            xaxis_title="Fraud Probability",
            yaxis_title="Count",
            height=400
        )
        fig.add_vline(x=risk_threshold, line_dash="dash", line_color="red",
                     annotation_text=f"Threshold: {risk_threshold:.2f}")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: DETAILED ANALYSIS
# ============================================================================

with tab2:
    st.header("Company-Level Fraud Analysis")
    
    # Company search
    search_company = st.text_input("Search by Company ID", "")
    
    filtered_companies_detail = companies[companies["location"].isin(location_filter)]
    filtered_companies_detail = filtered_companies_detail[
        filtered_companies_detail["fraud_probability"] > risk_threshold
    ]
    
    if search_company:
        try:
            company_id = int(search_company)
            filtered_companies_detail = filtered_companies_detail[
                filtered_companies_detail["company_id"] == company_id
            ]
        except ValueError:
            st.warning("Please enter a valid company ID")
    
    # Sort by fraud probability
    filtered_companies_detail = filtered_companies_detail.sort_values(
        "fraud_probability",
        ascending=False
    )
    
    # Display table
    st.subheader("High-Risk Companies (Sorted by Risk)")
    
    display_cols = ["company_id", "location", "turnover", "fraud_probability", "predicted_fraud"]
    display_df = filtered_companies_detail[display_cols].copy()
    display_df["fraud_probability"] = display_df["fraud_probability"].apply(lambda x: f"{x:.2%}")
    display_df["turnover"] = display_df["turnover"].apply(lambda x: f"₹{x:.2f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Company detail view
    st.subheader("Detailed Company Information")
    
    if len(filtered_companies_detail) > 0:
        selected_company_idx = st.selectbox(
            "Select a company to view details",
            range(len(filtered_companies_detail)),
            format_func=lambda i: f"Company {filtered_companies_detail.iloc[i]['company_id']}"
        )
        
        company = filtered_companies_detail.iloc[selected_company_idx]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Company ID", int(company["company_id"]))
            st.metric("Location", company["location"])
        
        with col2:
            st.metric("Fraud Probability", f"{company['fraud_probability']:.2%}")
            st.metric("Predicted Status", "🚨 FRAUD" if company["predicted_fraud"] == 1 else "✅ Normal")
        
        with col3:
            st.metric("Turnover", f"₹{company['turnover']:.2f}")
            st.metric("Invoice Frequency", int(company["invoice_frequency"]))
        
        # Transaction network
        st.subheader("Transaction Partners")
        company_id = int(company["company_id"])
        
        # Outgoing invoices
        outgoing = invoices[invoices["seller_id"] == company_id]
        # Incoming invoices
        incoming = invoices[invoices["buyer_id"] == company_id]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Invoices Sent", len(outgoing))
            st.metric("Total Amount Sent", f"₹{outgoing['amount'].sum():.2f}")
        
        with col2:
            st.metric("Invoices Received", len(incoming))
            st.metric("Total Amount Received", f"₹{incoming['amount'].sum():.2f}")

# ============================================================================
# TAB 3: RISK SCORING
# ============================================================================

with tab3:
    st.header("Risk Scoring Model")
    
    st.markdown("""
    ### Fraud Risk Calculation
    
    The GNN model computes a fraud probability for each company based on:
    
    **Node Features:**
    - Company Turnover
    - Invoice Sent Count
    - Invoice Received Count
    
    **Graph Structure:**
    - Connected company networks (invoice relationships)
    - Invoice amounts and ITC values
    - Subgraph patterns (shell companies)
    
    **Model Architecture:**
    - Graph Convolutional Networks (GCN)
    - 3-layer architecture for learning patterns
    - Classification into Normal (0) or Fraud (1)
    """)
    
    # Risk distribution by location
    fig = px.box(
        companies,
        x="location",
        y="fraud_probability",
        title="Fraud Probability Distribution by Location",
        labels={"fraud_probability": "Fraud Probability", "location": "Location"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Turnover vs Risk
    fig = px.scatter(
        companies[companies["location"].isin(location_filter)],
        x="turnover",
        y="fraud_probability",
        color="predicted_fraud",
        title="Company Turnover vs Fraud Risk",
        labels={"turnover": "Turnover (₹)", "fraud_probability": "Fraud Probability"},
        hover_data=["company_id", "location"]
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4: NETWORK INSIGHTS
# ============================================================================

with tab4:
    st.header("Network Analysis")
    
    st.markdown("""
    ### Graph-Level Insights
    
    These metrics are calculated from the transaction network structure.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    # Load graph stats if available
    with col1:
        st.metric("Total Nodes", graph_data.num_nodes)
    
    with col2:
        st.metric("Total Edges", graph_data.num_edges)
    
    with col3:
        density = (2 * graph_data.num_edges) / (graph_data.num_nodes * (graph_data.num_nodes - 1))
        st.metric("Network Density", f"{density:.4f}")
    
    st.divider()
    
    # Invoice pattern analysis
    st.subheader("Invoice Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top senders
        top_senders = invoices.groupby("seller_id").size().nlargest(10)
        fig = go.Figure(data=[
            go.Bar(y=top_senders.values, x=top_senders.index, marker=dict(color="steelblue"))
        ])
        fig.update_layout(
            title="Top 10 Invoice Senders",
            xaxis_title="Company ID",
            yaxis_title="Invoice Count",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top recipients
        top_recipients = invoices.groupby("buyer_id").size().nlargest(10)
        fig = go.Figure(data=[
            go.Bar(y=top_recipients.values, x=top_recipients.index, marker=dict(color="coral"))
        ])
        fig.update_layout(
            title="Top 10 Invoice Recipients",
            xaxis_title="Company ID",
            yaxis_title="Invoice Count",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
---
**Tax Fraud Detection System** | Powered by Graph Neural Networks  
*SIH 2024 Project* | For demonstration and research purposes only
""")
