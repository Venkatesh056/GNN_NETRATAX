"""
Flask Web Application for Tax Fraud Detection
Replace Streamlit with professional HTML/CSS interface
"""

from flask import Flask, render_template, request, jsonify
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import sys
import logging
import plotly.graph_objects as go
import plotly.express as px
import json
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from gnn_models.train_gnn import GNNFraudDetector
from db import init_db, record_upload, list_uploads
from crypto import encrypt_file

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Custom JSON encoder for numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

app.json_encoder = NumpyEncoder

# Global variables for model and data
MODEL = None
GRAPH_DATA = None
DEVICE = None
COMPANIES = None
INVOICES = None
MAPPINGS = None
FRAUD_PROBA = None


def load_model_and_data():
    """Load model and data on startup"""
    global MODEL, GRAPH_DATA, DEVICE, COMPANIES, INVOICES, MAPPINGS, FRAUD_PROBA
    
    data_path = Path(__file__).parent / "data" / "processed"
    models_path = Path(__file__).parent / "models"
    
    logger.info("Loading data...")
    COMPANIES = pd.read_csv(data_path / "companies_processed.csv")
    INVOICES = pd.read_csv(data_path / "invoices_processed.csv")
    
    logger.info("Loading graph...")
    # Handle PyTorch 2.6+ safe_globals for torch_geometric
    try:
        GRAPH_DATA = torch.load(data_path / "graphs" / "graph_data.pt", weights_only=False)
    except Exception as e:
        logger.warning(f"Could not load graph with weights_only=False: {e}")
        try:
            import torch.serialization
            from torch_geometric.data import Data as PyGData
            torch.serialization.add_safe_globals([PyGData])
            GRAPH_DATA = torch.load(data_path / "graphs" / "graph_data.pt", weights_only=False)
        except Exception as e2:
            logger.error(f"Failed to load graph: {e2}")
            raise
    
    logger.info("Loading model...")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = GNNFraudDetector(in_channels=3, hidden_channels=64, out_channels=2, model_type="gcn").to(DEVICE)
    
    try:
        MODEL.load_state_dict(torch.load(models_path / "best_model.pt", map_location=DEVICE))
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load model weights: {e}")
    
    with open(data_path / "graphs" / "node_mappings.pkl", "rb") as f:
        MAPPINGS = pickle.load(f)
    
    # Get fraud predictions
    MODEL.eval()
    with torch.no_grad():
        out = MODEL(GRAPH_DATA.x.to(DEVICE), GRAPH_DATA.edge_index.to(DEVICE))
        predictions = torch.softmax(out, dim=1)
        FRAUD_PROBA = predictions[:, 1].cpu().numpy()
    
    COMPANIES["fraud_probability"] = FRAUD_PROBA
    COMPANIES["predicted_fraud"] = (FRAUD_PROBA > 0.5).astype(int)
    
    logger.info("Model and data loaded successfully!")


@app.route('/upload', methods=['GET'])
def upload_page():
    """Render upload page"""
    return render_template('upload.html')


@app.route('/api/upload_data', methods=['POST'])
def upload_data():
    """Accept CSV uploads (companies or invoices), save and record metadata"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'no file part'}), 400
        f = request.files['file']
        if f.filename == '':
            return jsonify({'error': 'no selected file'}), 400

        fname = f.filename
        # Save under data/uploads/<timestamp>/fname
        uploads_dir = Path(__file__).parent / 'data' / 'uploads' / time.strftime('%Y%m%d')
        uploads_dir.mkdir(parents=True, exist_ok=True)
        save_path = uploads_dir / fname
        f.save(str(save_path))

        # Quick validation: try reading head with pandas
        import pandas as pd
        df = pd.read_csv(save_path)
        rows, cols = df.shape

        # Optional encryption: form field 'encrypt' can be 'on'/'true'/'1'
        encrypt_flag = str(request.form.get('encrypt', '')).lower() in ('1', 'true', 'on', 'yes')
        stored_path = save_path
        encrypted = 0
        if encrypt_flag:
            try:
                enc_path = encrypt_file(save_path)
                # remove plaintext copy
                try:
                    import os
                    os.remove(save_path)
                except Exception:
                    pass
                stored_path = enc_path
                encrypted = 1
            except Exception as ee:
                logger.error(f"Encryption failed: {ee}", exc_info=True)
                return jsonify({'error': 'encryption_failed', 'detail': str(ee)}), 500

        # Record to DB (include encrypted flag)
        record_upload(fname, stored_path, uploader=request.form.get('uploader', 'anonymous'), filetype='csv', rows=int(rows), columns=int(cols), encrypted=encrypted)

        return jsonify({'status': 'ok', 'filename': fname, 'rows': int(rows), 'columns': int(cols), 'encrypted': bool(encrypted)})

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/uploads')
def uploads_list():
    try:
        init_db()
        items = list_uploads(limit=100)
        return jsonify(items)
    except Exception as e:
        logger.error(f"Error listing uploads: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ROUTES - Pages
# ============================================================================

@app.route("/")
def home():
    """Home/Dashboard page"""
    high_risk_count = (FRAUD_PROBA > 0.5).sum()
    avg_risk = FRAUD_PROBA.mean()
    fraud_count = (COMPANIES["predicted_fraud"] == 1).sum()
    
    return render_template("index.html",
                         total_companies=len(COMPANIES),
                         high_risk_count=int(high_risk_count),
                         fraud_count=int(fraud_count),
                         avg_risk=f"{avg_risk:.2%}")


@app.route("/dashboard")
def dashboard():
    """Dashboard with analytics"""
    fraud_dist = COMPANIES["predicted_fraud"].value_counts()
    
    return render_template("dashboard.html",
                         total_companies=len(COMPANIES),
                         fraud_count=int(fraud_dist.get(1, 0)))


@app.route("/companies")
def companies():
    """Companies listing page"""
    return render_template("companies.html")


@app.route("/analytics")
def analytics():
    """Analytics page"""
    return render_template("analytics.html")


@app.route("/api/companies")
def get_companies():
    """API: Get all companies with filters"""
    try:
        # Get filters from query parameters
        risk_threshold = float(request.args.get("risk_threshold", 0.5))
        location_filter = request.args.get("location", "").split(",")
        location_filter = [l.strip() for l in location_filter if l.strip()]
        search_id = request.args.get("search", "").strip()
        
        # Filter data
        filtered_df = COMPANIES.copy()
        
        # Apply location filter
        if location_filter and location_filter != [""]:
            filtered_df = filtered_df[filtered_df["location"].isin(location_filter)]
        
        # Apply risk threshold
        filtered_df = filtered_df[filtered_df["fraud_probability"] >= risk_threshold]
        
        # Apply search (company_id is a string like "GST000123")
        if search_id:
            filtered_df = filtered_df[filtered_df["company_id"].astype(str).str.contains(search_id, case=False)]
        
        # Sort by fraud probability
        filtered_df = filtered_df.sort_values("fraud_probability", ascending=False)
        
        # Format for JSON
        companies_list = []
        for _, row in filtered_df.iterrows():
            companies_list.append({
                "company_id": str(row["company_id"]),
                "location": str(row["location"]),
                "turnover": f"₹{float(row['turnover']):.2f}",
                "fraud_probability": f"{float(row['fraud_probability']):.2%}",
                "risk_level": "🔴 HIGH" if float(row["fraud_probability"]) > 0.7 else "🟡 MEDIUM" if float(row["fraud_probability"]) > 0.3 else "🟢 LOW",
                "status": "🚨 FRAUD" if int(row["predicted_fraud"]) == 1 else "✅ NORMAL"
            })
        
        return jsonify(companies_list)
    
    except Exception as e:
        logger.error(f"Error in get_companies: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/company/<company_id>")
def get_company_detail(company_id):
    """API: Get company details"""
    try:
        company_id_str = str(company_id).strip()
        company = COMPANIES[COMPANIES["company_id"].astype(str) == company_id_str]
        
        if len(company) == 0:
            return jsonify({"error": f"Company {company_id_str} not found"}), 404
        
        row = company.iloc[0]
        
        # Get transaction info
        outgoing = INVOICES[INVOICES["seller_id"].astype(str) == company_id_str]
        incoming = INVOICES[INVOICES["buyer_id"].astype(str) == company_id_str]
        
        return jsonify({
            "company_id": company_id_str,
            "location": str(row["location"]),
            "turnover": float(row["turnover"]),
            "fraud_probability": float(row["fraud_probability"]),
            "predicted_fraud": int(row["predicted_fraud"]),
            "risk_level": "HIGH" if float(row["fraud_probability"]) > 0.7 else "MEDIUM" if float(row["fraud_probability"]) > 0.3 else "LOW",
            "sent_invoice_count": int(row.get("sent_invoices", 0)),
            "received_invoice_count": int(row.get("received_invoices", 0)),
            "total_sent_amount": float(row.get("total_sent_amount", 0)),
            "total_received_amount": float(row.get("total_received_amount", 0)),
            "outgoing_invoices": len(outgoing),
            "incoming_invoices": len(incoming)
        })
    
    except Exception as e:
        logger.error(f"Error in get_company_detail: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/statistics")
def get_statistics():
    """API: Get overall statistics"""
    try:
        high_risk = (FRAUD_PROBA > 0.7).sum()
        medium_risk = ((FRAUD_PROBA > 0.3) & (FRAUD_PROBA <= 0.7)).sum()
        low_risk = (FRAUD_PROBA <= 0.3).sum()
        
        return jsonify({
            "total_companies": len(COMPANIES),
            "total_edges": GRAPH_DATA.num_edges,
            "high_risk_count": int(high_risk),
            "medium_risk_count": int(medium_risk),
            "low_risk_count": int(low_risk),
            "fraud_count": int((COMPANIES["predicted_fraud"] == 1).sum()),
            "average_fraud_probability": float(np.mean(FRAUD_PROBA))
        })
    
    except Exception as e:
        logger.error(f"Error in get_statistics: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart/fraud_distribution")
def chart_fraud_distribution():
    """API: Fraud distribution chart - returns Plotly JSON"""
    try:
        fraud_dist = COMPANIES["predicted_fraud"].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(x=["Normal", "Fraud"],
                   y=[fraud_dist.get(0, 0), fraud_dist.get(1, 0)],
                   marker=dict(color=["green", "red"]))
        ])
        fig.update_layout(
            title="Fraud Distribution",
            xaxis_title="Status",
            yaxis_title="Count",
            height=400,
            showlegend=False
        )
        
        return jsonify(fig.to_dict())
    
    except Exception as e:
        logger.error(f"Error in chart_fraud_distribution: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart/risk_distribution")
def chart_risk_distribution():
    """API: Risk score distribution chart - returns Plotly JSON"""
    try:
        fig = go.Figure(data=[
            go.Histogram(
                x=FRAUD_PROBA.tolist(),  # Convert numpy array to list
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
        fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                     annotation_text="Threshold: 0.50")
        
        return jsonify(fig.to_dict())
    
    except Exception as e:
        logger.error(f"Error in chart_risk_distribution: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart/risk_by_location")
def chart_risk_by_location():
    """API: Risk by location chart - returns Plotly JSON"""
    try:
        fig = px.box(
            COMPANIES,
            x="location",
            y="fraud_probability",
            title="Fraud Probability Distribution by Location",
            labels={"fraud_probability": "Fraud Probability", "location": "Location"}
        )
        fig.update_layout(height=400)
        
        return jsonify(fig.to_dict())
    
    except Exception as e:
        logger.error(f"Error in chart_risk_by_location: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart/turnover_vs_risk")
def chart_turnover_vs_risk():
    """API: Turnover vs Risk scatter plot - returns Plotly JSON"""
    try:
        fig = px.scatter(
            COMPANIES,
            x="turnover",
            y="fraud_probability",
            color="predicted_fraud",
            title="Company Turnover vs Fraud Risk",
            labels={"turnover": "Turnover (₹)", "fraud_probability": "Fraud Probability"},
            hover_data=["company_id", "location"]
        )
        fig.update_layout(height=400)
        
        return jsonify(fig.to_dict())
    
    except Exception as e:
        logger.error(f"Error in chart_turnover_vs_risk: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/top_senders")
def get_top_senders():
    """API: Top invoice senders"""
    try:
        top_senders = INVOICES.groupby("seller_id").size().nlargest(10)
        
        data = []
        for seller_id, count in top_senders.items():
            seller_id_str = str(seller_id).strip()
            company = COMPANIES[COMPANIES["company_id"].astype(str) == seller_id_str]
            if len(company) > 0:
                data.append({
                    "company_id": seller_id_str,
                    "invoice_count": int(count),
                    "fraud_probability": f"{float(company.iloc[0]['fraud_probability']):.2%}"
                })
        
        return jsonify(data)
    
    except Exception as e:
        logger.error(f"Error in get_top_senders: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/top_receivers")
def get_top_receivers():
    """API: Top invoice receivers"""
    try:
        top_receivers = INVOICES.groupby("buyer_id").size().nlargest(10)
        
        data = []
        for buyer_id, count in top_receivers.items():
            buyer_id_str = str(buyer_id).strip()
            company = COMPANIES[COMPANIES["company_id"].astype(str) == buyer_id_str]
            if len(company) > 0:
                data.append({
                    "company_id": buyer_id_str,
                    "invoice_count": int(count),
                    "fraud_probability": f"{float(company.iloc[0]['fraud_probability']):.2%}"
                })
        
        return jsonify(data)
    
    except Exception as e:
        logger.error(f"Error in get_top_receivers: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/locations")
def get_locations():
    """API: Get all unique locations for filtering"""
    try:
        locations = sorted(COMPANIES["location"].unique().tolist())
        return jsonify(locations)
    except Exception as e:
        logger.error(f"Error in get_locations: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """API: Single company prediction"""
    try:
        data = request.get_json()
        company_id = data.get("company_id")
        
        if not company_id:
            return jsonify({"error": "company_id is required"}), 400
        
        node_list = MAPPINGS["node_list"]
        if company_id not in node_list:
            return jsonify({"error": f"Company ID {company_id} not found"}), 404
        
        node_idx = node_list.index(company_id)
        fraud_proba = float(FRAUD_PROBA[node_idx])
        
        company_row = COMPANIES[COMPANIES["company_id"] == company_id].iloc[0]
        
        return jsonify({
            "company_id": company_id,
            "fraud_probability": fraud_proba,
            "is_fraud": float(fraud_proba > 0.5),
            "risk_level": "HIGH" if fraud_proba > 0.7 else "MEDIUM" if fraud_proba > 0.3 else "LOW",
            "location": company_row["location"],
            "turnover": float(company_row["turnover"])
        })
    
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return render_template("500.html"), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    logger.info("Loading model and data...")
    load_model_and_data()
    
    logger.info("Starting Flask application...")
    app.run(debug=True, host="0.0.0.0", port=5000)
