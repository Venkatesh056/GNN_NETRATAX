from flask import Flask, render_template, request, jsonify, send_from_directory, Response
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
import os
import networkx as nx
from collections import deque
import torch.nn as nn
from flask_cors import CORS
from torch_geometric.data import Data  # Add this import

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules with error handling
GNNFraudDetector = None
init_db = None
record_upload = None
list_uploads = None
encrypt_file = None

try:
    from src.gnn_models.train_gnn import GNNFraudDetector
    from src.db import init_db, record_upload, list_uploads
    from src.crypto import encrypt_file
except ImportError as e:
    logger.error(f"Import error: {e}")

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['JSON_SORT_KEYS'] = False
CORS(app)  # Enable CORS for React dev server

# Custom JSON encoder for numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.bool_):
            return bool(o)
        return super(NumpyEncoder, self).default(o)

# For Flask 2.0+, use json.default instead of json_encoder
app.json.default = lambda o: NumpyEncoder().default(o)

# Global variables for model and data
MODEL = None
GRAPH_DATA = None
DEVICE = None
COMPANIES = None
INVOICES = None
MAPPINGS = None
FRAUD_PROBA = None
NETWORKX_GRAPH = None  # Add NetworkX graph for easier manipulation

# Accumulation tracking
ACCUMULATED_DATA_PATH = Path(__file__).parent / "data" / "accumulated"
UPLOAD_HISTORY = []  # Track all uploads for audit trail
TOTAL_UPLOADS = 0
LAST_RETRAIN_TIME = None


def _first_bad_param(model: nn.Module):
    """Return the first parameter name that contains NaN/Inf, else None."""
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            return name
    return None


def load_model_and_data():
    """Load model and data on startup - prioritizes accumulated data"""
    global MODEL, GRAPH_DATA, DEVICE, COMPANIES, INVOICES, MAPPINGS, FRAUD_PROBA, NETWORKX_GRAPH, ACCUMULATED_DATA_PATH
    
    data_path = Path(__file__).parent / "data" / "processed"
    models_path = Path(__file__).parent / "models"
    uploads_path = Path(__file__).parent / "data" / "uploads"
    
    # Create accumulated data directory if it doesn't exist
    ACCUMULATED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading data...")
    
    # PRIORITY 1: Try to load accumulated data first (most recent state)
    accumulated_companies = ACCUMULATED_DATA_PATH / "companies_accumulated.csv"
    accumulated_invoices = ACCUMULATED_DATA_PATH / "invoices_accumulated.csv"
    
    companies_file = None
    invoices_file = None
    
    if accumulated_companies.exists():
        logger.info(f"Loading ACCUMULATED companies data: {accumulated_companies}")
        companies_file = accumulated_companies
    
    if accumulated_invoices.exists():
        logger.info(f"Loading ACCUMULATED invoices data: {accumulated_invoices}")
        invoices_file = accumulated_invoices
    
    # PRIORITY 2: Try uploads folder if not found in accumulated
    if companies_file is None or invoices_file is None:
        if uploads_path.exists():
            upload_folders = sorted([d for d in uploads_path.iterdir() if d.is_dir()], reverse=True)
            for upload_folder in upload_folders:
                companies_csv = upload_folder / "companies.csv"
                invoices_csv = upload_folder / "invoices.csv"
                if companies_csv.exists() and companies_file is None:
                    companies_file = companies_csv
                    logger.info(f"Found companies.csv in uploads: {companies_csv}")
                if invoices_csv.exists() and invoices_file is None:
                    invoices_file = invoices_csv
                    logger.info(f"Found invoices.csv in uploads: {invoices_csv}")
                if companies_file and invoices_file:
                    break
    
    # PRIORITY 3: Fallback to processed data
    if companies_file is None:
        companies_file = data_path / "companies_processed.csv"
        if not companies_file.exists():
            companies_file = data_path.parent / "raw" / "companies.csv"
        logger.info(f"Loading companies from: {companies_file}")
    
    if invoices_file is None:
        invoices_file = data_path / "invoices_processed.csv"
        if not invoices_file.exists():
            invoices_file = data_path.parent / "raw" / "invoices.csv"
        logger.info(f"Loading invoices from: {invoices_file}")
    
    COMPANIES = pd.read_csv(companies_file)
    INVOICES = pd.read_csv(invoices_file) if invoices_file.exists() else pd.DataFrame()
    logger.info(f"Loaded {len(COMPANIES)} companies and {len(INVOICES)} invoices")
    
    logger.info("Loading graph...")
    # Handle PyTorch 2.6+ safe_globals for torch_geometric
    graph_needs_rebuild = False
    try:
        GRAPH_DATA = torch.load(data_path / "graphs" / "graph_data.pt", weights_only=False)
        # Check if graph has correct number of features
        if GRAPH_DATA.x.shape[1] != NUM_NODE_FEATURES:
            logger.warning(f"Graph has {GRAPH_DATA.x.shape[1]} features but model expects {NUM_NODE_FEATURES}")
            graph_needs_rebuild = True
    except Exception as e:
        logger.warning(f"Could not load graph: {e}")
        graph_needs_rebuild = True
    
    # Load NetworkX graph for easier manipulation
    try:
        with open(data_path / "graphs" / "networkx_graph.gpickle", "rb") as f:
            NETWORKX_GRAPH = pickle.load(f)
        logger.info("Loaded NetworkX graph")
    except Exception as e:
        logger.warning(f"Could not load NetworkX graph: {e}")
        NETWORKX_GRAPH = nx.DiGraph()
    
    logger.info("Loading model...")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use 12 features: turnover, sent_invoices, received_invoices, total_sent_amount, total_received_amount,
    # unique_buyers, unique_sellers, circular_trading_score, gst_compliance_rate, late_filing_count, round_amount_ratio, buyer_concentration
    MODEL = GNNFraudDetector(in_channels=NUM_NODE_FEATURES, hidden_channels=64, out_channels=2, model_type="gcn").to(DEVICE)
    
    # If graph needs rebuild OR model weights don't match, rebuild everything
    model_needs_training = False
    try:
        MODEL.load_state_dict(torch.load(models_path / "best_model.pt", map_location=DEVICE))
        logger.info("Model weights loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load model weights (will train fresh): {e}")
        model_needs_training = True

    # Detect corrupted weights (NaN/Inf) and reinitialize
    bad_param = _first_bad_param(MODEL)
    if bad_param:
        logger.warning(f"Detected NaN/Inf in model parameter '{bad_param}'. Reinitializing weights and retraining.")

        def _reset(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

        MODEL.apply(_reset)
        model_needs_training = True
    
    # If graph needs rebuild, do it now with proper features
    if graph_needs_rebuild or model_needs_training:
        logger.info("=" * 60)
        logger.info("REBUILDING GRAPH AND TRAINING MODEL WITH NEW FEATURES")
        logger.info("=" * 60)
        
        # Rebuild graph with extended features
        NETWORKX_GRAPH, MAPPINGS, total_nodes, total_edges = rebuild_full_graph(COMPANIES, INVOICES)
        logger.info(f"Graph rebuilt: {total_nodes} nodes, {total_edges} edges")
        
        # Convert to PyG format with extended features
        GRAPH_DATA = rebuild_pyg_graph(NETWORKX_GRAPH, COMPANIES)
        logger.info(f"PyG graph: {GRAPH_DATA.x.shape[0]} nodes, {GRAPH_DATA.x.shape[1]} features")

        # Retrain if weights were missing/corrupted or graph changed
        training_stats = retrain_full_model(GRAPH_DATA, epochs=80, lr=0.001)
        
        # Save the updated graph and model
        save_updated_graph_and_model()
        save_accumulated_data()
    else:
        with open(data_path / "graphs" / "node_mappings.pkl", "rb") as f:
            MAPPINGS = pickle.load(f)
    
    # Get fraud predictions
    MODEL.eval()
    with torch.no_grad():
        out = MODEL(GRAPH_DATA.x.to(DEVICE), GRAPH_DATA.edge_index.to(DEVICE))
        predictions = torch.softmax(out, dim=1)
        predictions = torch.nan_to_num(predictions, nan=0.5, posinf=1.0, neginf=0.0)
        FRAUD_PROBA = predictions[:, 1].clamp(0, 1).cpu().numpy()
    
    # Handle mismatch between companies data and graph nodes
    # This can happen when companies data has been updated but graph hasn't been rebuilt
    if len(COMPANIES) != len(FRAUD_PROBA):
        logger.warning(f"Length mismatch: Companies ({len(COMPANIES)}) vs Fraud probabilities ({len(FRAUD_PROBA)})")
        if len(COMPANIES) > len(FRAUD_PROBA):
            # More companies than graph nodes - truncate companies to match
            logger.info(f"Truncating companies data from {len(COMPANIES)} to {len(FRAUD_PROBA)} rows")
            COMPANIES = COMPANIES.iloc[:len(FRAUD_PROBA)]
        else:
            # More graph nodes than companies - pad fraud probabilities with zeros
            logger.info(f"Padding fraud probabilities from {len(FRAUD_PROBA)} to {len(COMPANIES)} elements")
            padded_fraud_proba = np.zeros(len(COMPANIES))
            padded_fraud_proba[:len(FRAUD_PROBA)] = FRAUD_PROBA
            FRAUD_PROBA = padded_fraud_proba
    
    COMPANIES["fraud_probability"] = FRAUD_PROBA
    # Use actual is_fraud labels if available, otherwise use model predictions
    if "is_fraud" in COMPANIES.columns:
        COMPANIES["predicted_fraud"] = COMPANIES["is_fraud"].astype(int)
        logger.info(f"Using ground truth labels: {(COMPANIES['is_fraud']==1).sum()} fraud, {(COMPANIES['is_fraud']==0).sum()} non-fraud")
    else:
        COMPANIES["predicted_fraud"] = (FRAUD_PROBA > 0.5).astype(int)
    
    logger.info("Model and data loaded successfully!")


@app.route('/upload', methods=['GET'])
def upload_page():
    """Render upload page"""
    return render_template('upload.html')


@app.route('/api/upload_data', methods=['POST'])
def upload_data():
    """Accept CSV uploads - both companies (nodes) and invoices (edges) files"""
    try:
        # Check for dual file upload (new format)
        companies_file = request.files.get('companies_file')
        invoices_file = request.files.get('invoices_file')
        
        # Also support legacy single file upload
        legacy_file = request.files.get('file')
        
        if not companies_file and not invoices_file and not legacy_file:
            return jsonify({'error': 'No files uploaded. Please provide both companies and invoices CSV files.'}), 400
        
        # Setup upload directory
        uploads_dir = Path(__file__).parent / 'data' / 'uploads' / time.strftime('%Y%m%d')
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        companies_path = None
        invoices_path = None
        total_rows = 0
        total_cols = 0
        
        # Handle companies file
        if companies_file and companies_file.filename:
            companies_path = uploads_dir / companies_file.filename
            companies_file.save(str(companies_path))
            df_companies = pd.read_csv(companies_path)
            total_rows += df_companies.shape[0]
            total_cols = max(total_cols, df_companies.shape[1])
            logger.info(f"Saved companies file: {companies_file.filename} ({df_companies.shape[0]} rows)")
            record_upload(companies_file.filename, companies_path, uploader=request.form.get('uploader', 'anonymous'), filetype='csv', rows=int(df_companies.shape[0]), columns=int(df_companies.shape[1]), encrypted=0)
        
        # Handle invoices file
        if invoices_file and invoices_file.filename:
            invoices_path = uploads_dir / invoices_file.filename
            invoices_file.save(str(invoices_path))
            df_invoices = pd.read_csv(invoices_path)
            total_rows += df_invoices.shape[0]
            total_cols = max(total_cols, df_invoices.shape[1])
            logger.info(f"Saved invoices file: {invoices_file.filename} ({df_invoices.shape[0]} rows)")
            record_upload(invoices_file.filename, invoices_path, uploader=request.form.get('uploader', 'anonymous'), filetype='csv', rows=int(df_invoices.shape[0]), columns=int(df_invoices.shape[1]), encrypted=0)
        
        # Handle legacy single file upload
        if legacy_file and legacy_file.filename and not companies_path and not invoices_path:
            legacy_path = uploads_dir / legacy_file.filename
            legacy_file.save(str(legacy_path))
            df = pd.read_csv(legacy_path)
            total_rows = df.shape[0]
            total_cols = df.shape[1]
            record_upload(legacy_file.filename, legacy_path, uploader=request.form.get('uploader', 'anonymous'), filetype='csv', rows=int(total_rows), columns=int(total_cols), encrypted=0)
            
            # Process single file for incremental learning
            try:
                logger.info(f"Starting incremental learning for file: {legacy_file.filename}")
                incremental_results = process_incremental_learning(legacy_path, legacy_file.filename)
                logger.info(f"Incremental learning completed for file: {legacy_file.filename}")
                return jsonify({
                    'status': 'ok',
                    'message': f'Successfully processed {legacy_file.filename}',
                    'filename': legacy_file.filename,
                    'rows': int(total_rows),
                    'columns': int(total_cols),
                    'incremental_learning': incremental_results
                })
            except Exception as e:
                logger.error(f"Incremental learning failed: {e}", exc_info=True)
                return jsonify({
                    'status': 'ok',
                    'message': f'File uploaded but incremental learning failed: {str(e)}',
                    'filename': legacy_file.filename,
                    'rows': int(total_rows),
                    'columns': int(total_cols)
                })
        
        # Process dual file upload for incremental learning
        incremental_results = None
        try:
            logger.info(f"Starting incremental learning with both companies and invoices files")
            incremental_results = process_dual_file_upload(companies_path, invoices_path)
            logger.info(f"Incremental learning completed")
        except Exception as e:
            logger.error(f"Incremental learning failed: {e}", exc_info=True)
            return jsonify({
                'status': 'partial',
                'message': f'Files uploaded but incremental learning failed: {str(e)}',
                'error': str(e)
            }), 200
        
        # Build response
        response_data = {
            'status': 'ok',
            'message': 'Successfully processed both datasets!',
            'total_rows': int(total_rows),
            'total_columns': int(total_cols)
        }
        
        if incremental_results:
            response_data.update(incremental_results)
            
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def process_incremental_learning(file_path, filename):
    """
    Process uploaded CSV for TRUE incremental learning with data ACCUMULATION.
    
    This function:
    1. Loads new data from uploaded CSV
    2. MERGES with existing accumulated data (deduplicates by company_id)
    3. Rebuilds the FULL graph with accumulated data
    4. Retrains model on full accumulated graph
    5. Updates fraud scores for ALL companies
    6. Persists accumulated state for next session
    7. Dashboard shows ALL accumulated data
    """
    global COMPANIES, INVOICES, NETWORKX_GRAPH, GRAPH_DATA, MAPPINGS, FRAUD_PROBA, MODEL
    global UPLOAD_HISTORY, TOTAL_UPLOADS, LAST_RETRAIN_TIME, ACCUMULATED_DATA_PATH
    
    import time as time_module
    start_time = time_module.time()
    
    logger.info(f"=" * 60)
    logger.info(f"INCREMENTAL LEARNING: Processing {filename}")
    logger.info(f"=" * 60)
    
    # Track upload
    TOTAL_UPLOADS += 1
    upload_record = {
        "upload_id": TOTAL_UPLOADS,
        "filename": filename,
        "timestamp": time_module.strftime("%Y-%m-%d %H:%M:%S"),
        "file_path": str(file_path)
    }
    UPLOAD_HISTORY.append(upload_record)
    
    # Record counts BEFORE processing
    companies_before = len(COMPANIES) if COMPANIES is not None else 0
    invoices_before = len(INVOICES) if INVOICES is not None else 0
    
    logger.info(f"State BEFORE: {companies_before} companies, {invoices_before} invoices")
    
    # Load the uploaded data
    df = pd.read_csv(file_path)
    
    # Determine if it's companies or invoices data
    new_companies_df = pd.DataFrame()
    new_invoices_df = pd.DataFrame()
    
    if "company_id" in df.columns:
        # Companies data
        logger.info("Processing companies data...")
        new_companies_df = df.copy()
        # Apply basic cleaning (similar to clean_data.py)
        if "turnover" not in new_companies_df.columns:
            # Try to use avg_monthly_turnover if available
            if "avg_monthly_turnover" in new_companies_df.columns:
                new_companies_df["turnover"] = new_companies_df["avg_monthly_turnover"]
            else:
                new_companies_df["turnover"] = 0
        if "location" not in new_companies_df.columns:
            new_companies_df["location"] = "Unknown"
        if "is_fraud" not in new_companies_df.columns:
            new_companies_df["is_fraud"] = 0
            
        # Ensure correct data types
        new_companies_df["company_id"] = new_companies_df["company_id"].astype(str).str.strip()
        new_companies_df["turnover"] = pd.to_numeric(new_companies_df["turnover"], errors='coerce').fillna(0)
        new_companies_df["is_fraud"] = pd.to_numeric(new_companies_df["is_fraud"], errors='coerce').fillna(0).astype(int)
    elif "seller_id" in df.columns and "buyer_id" in df.columns:
        # Invoices data
        logger.info("Processing invoices data...")
        new_invoices_df = df.copy()
        # Apply basic cleaning
        if "amount" not in new_invoices_df.columns:
            new_invoices_df["amount"] = 0
        if "itc_claimed" not in new_invoices_df.columns:
            new_invoices_df["itc_claimed"] = 0
            
        # Ensure correct data types
        new_invoices_df["seller_id"] = new_invoices_df["seller_id"].astype(str).str.strip()
        new_invoices_df["buyer_id"] = new_invoices_df["buyer_id"].astype(str).str.strip()
        new_invoices_df["amount"] = pd.to_numeric(new_invoices_df["amount"], errors='coerce').fillna(0)
        new_invoices_df["itc_claimed"] = pd.to_numeric(new_invoices_df["itc_claimed"], errors='coerce').fillna(0)
    else:
        logger.warning(f"Unknown data format in {filename}")
        return
    
    # If we have invoices but no companies data, we need to extract company info
    if not new_companies_df.empty and new_invoices_df.empty:
        # Companies data only - need to engineer features
        logger.info("Engineering features for new companies...")
        # Simple feature engineering for new companies
        new_companies_df["sent_invoice_count"] = 0
        new_companies_df["received_invoice_count"] = 0
        new_companies_df["total_sent_amount"] = 0
        new_companies_df["total_received_amount"] = 0
    elif new_companies_df.empty and not new_invoices_df.empty:
        # Invoices data only - extract company info
        logger.info("Extracting company info from invoices...")
        seller_info = new_invoices_df.groupby("seller_id").agg({
            "amount": "sum",
            "invoice_id": "count"
        }).reset_index()
        seller_info.columns = ["company_id", "total_sent_amount", "sent_invoice_count"]
        seller_info["company_id"] = seller_info["company_id"].astype(str)
        
        buyer_info = new_invoices_df.groupby("buyer_id").agg({
            "amount": "sum",
            "invoice_id": "count"
        }).reset_index()
        buyer_info.columns = ["company_id", "total_received_amount", "received_invoice_count"]
        buyer_info["company_id"] = buyer_info["company_id"].astype(str)
        
        # Merge seller and buyer info
        company_info = pd.merge(seller_info, buyer_info, on="company_id", how="outer").fillna(0)
        company_info["turnover"] = company_info["total_sent_amount"] + company_info["total_received_amount"]
        company_info["is_fraud"] = 0  # Default to non-fraud
        
        new_companies_df = company_info
    elif not new_companies_df.empty and not new_invoices_df.empty:
        # Both companies and invoices - engineer features
        logger.info("Engineering features for companies with invoices...")
        # Count invoices sent (seller perspective)
        seller_counts = new_invoices_df.groupby("seller_id").agg({
            "amount": "sum",
            "invoice_id": "count"
        }).reset_index()
        seller_counts.columns = ["company_id", "total_sent_amount", "sent_invoice_count"]
        seller_counts["company_id"] = seller_counts["company_id"].astype(str)
        
        # Count invoices received (buyer perspective)
        buyer_counts = new_invoices_df.groupby("buyer_id").agg({
            "amount": "sum",
            "invoice_id": "count"
        }).reset_index()
        buyer_counts.columns = ["company_id", "total_received_amount", "received_invoice_count"]
        buyer_counts["company_id"] = buyer_counts["company_id"].astype(str)
        
        # Merge features back to companies
        new_companies_df = new_companies_df.merge(seller_counts, on="company_id", how="left")
        new_companies_df = new_companies_df.merge(buyer_counts, on="company_id", how="left")
        
        # Fill NaN with 0
        new_companies_df.fillna(0, inplace=True)
    
    # =========================================================================
    # STEP 1: ACCUMULATE DATA (Merge new with existing)
    # =========================================================================
    logger.info("STEP 1: Accumulating data...")
    
    # Accumulate companies (keep latest version if duplicate company_id)
    if COMPANIES is not None and len(COMPANIES) > 0:
        # Ensure company_id is string in both DataFrames
        COMPANIES["company_id"] = COMPANIES["company_id"].astype(str).str.strip()
        new_companies_df["company_id"] = new_companies_df["company_id"].astype(str).str.strip()
        
        # Combine old and new, keeping latest version of duplicates
        combined_companies = pd.concat([COMPANIES, new_companies_df], ignore_index=True)
        COMPANIES = combined_companies.drop_duplicates(subset=["company_id"], keep="last")
        logger.info(f"Accumulated companies: {len(COMPANIES)} (added {len(new_companies_df)} new records)")
    else:
        COMPANIES = new_companies_df.copy()
        logger.info(f"Initialized companies with {len(COMPANIES)} records")
    
    # Accumulate invoices (append all - invoices are unique transactions)
    if not new_invoices_df.empty:
        if INVOICES is not None and len(INVOICES) > 0:
            # Ensure invoice_id column exists for deduplication
            if "invoice_id" in new_invoices_df.columns and "invoice_id" in INVOICES.columns:
                combined_invoices = pd.concat([INVOICES, new_invoices_df], ignore_index=True)
                INVOICES = combined_invoices.drop_duplicates(subset=["invoice_id"], keep="last")
            else:
                INVOICES = pd.concat([INVOICES, new_invoices_df], ignore_index=True)
            logger.info(f"Accumulated invoices: {len(INVOICES)} records")
        else:
            INVOICES = new_invoices_df.copy()
            logger.info(f"Initialized invoices with {len(INVOICES)} records")
    
    # Record counts AFTER accumulation
    companies_after = len(COMPANIES)
    invoices_after = len(INVOICES) if INVOICES is not None else 0
    new_companies_count = companies_after - companies_before
    new_invoices_count = invoices_after - invoices_before
    
    logger.info(f"State AFTER: {companies_after} companies (+{new_companies_count}), {invoices_after} invoices (+{new_invoices_count})")
    
    # =========================================================================
    # STEP 2: REBUILD FULL GRAPH from accumulated data
    # =========================================================================
    logger.info("STEP 2: Rebuilding full graph from accumulated data...")
    
    NETWORKX_GRAPH, MAPPINGS, total_nodes, total_edges = rebuild_full_graph(COMPANIES, INVOICES)
    
    logger.info(f"Full graph rebuilt: {total_nodes} nodes, {total_edges} edges")
    
    # =========================================================================
    # STEP 3: RETRAIN MODEL on full accumulated graph
    # =========================================================================
    logger.info("STEP 3: Retraining model on full accumulated graph...")
    
    # Rebuild PyG graph from accumulated data
    GRAPH_DATA = rebuild_pyg_graph(NETWORKX_GRAPH, COMPANIES)
    
    # Retrain model on full graph
    training_stats = retrain_full_model(GRAPH_DATA, epochs=100, lr=0.001)
    
    # =========================================================================
    # STEP 4: UPDATE FRAUD SCORES for ALL companies
    # =========================================================================
    logger.info("STEP 4: Updating fraud scores for ALL companies...")
    
    update_global_embeddings()
    
    # =========================================================================
    # STEP 5: PERSIST accumulated state
    # =========================================================================
    logger.info("STEP 5: Persisting accumulated state...")
    
    save_accumulated_data()
    save_updated_graph_and_model()
    
    # Update last retrain time
    LAST_RETRAIN_TIME = time_module.strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate processing time
    processing_time = time_module.time() - start_time
    
    # Count high-risk companies after retraining
    high_risk_count = int((FRAUD_PROBA > 0.7).sum()) if FRAUD_PROBA is not None else 0
    fraud_count = int((COMPANIES["predicted_fraud"] == 1).sum()) if "predicted_fraud" in COMPANIES.columns else 0
    
    logger.info(f"=" * 60)
    logger.info(f"INCREMENTAL LEARNING COMPLETE")
    logger.info(f"Total companies: {companies_after}, High-risk: {high_risk_count}, Fraud: {fraud_count}")
    logger.info(f"Processing time: {processing_time:.2f}s")
    logger.info(f"=" * 60)
    
    return {
        "status": "success",
        "companies_before": companies_before,
        "companies_after": companies_after,
        "new_companies": new_companies_count,
        "invoices_before": invoices_before,
        "invoices_after": invoices_after,
        "new_invoices": new_invoices_count,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "high_risk_count": high_risk_count,
        "fraud_count": fraud_count,
        "training_loss": training_stats.get("final_loss", 0),
        "training_accuracy": training_stats.get("accuracy", 0),
        "processing_time_seconds": round(processing_time, 2),
        "upload_number": TOTAL_UPLOADS
    }


def process_dual_file_upload(companies_path, invoices_path):
    """
    Process dual file upload (companies + invoices) for TRUE incremental learning.
    
    This function:
    1. Loads both companies (nodes) and invoices (edges) data
    2. MERGES with existing accumulated data
    3. Rebuilds the FULL graph with accumulated data
    4. Retrains model on full accumulated graph
    5. Returns comprehensive stats
    """
    global COMPANIES, INVOICES, NETWORKX_GRAPH, GRAPH_DATA, MAPPINGS, FRAUD_PROBA, MODEL
    global UPLOAD_HISTORY, TOTAL_UPLOADS, LAST_RETRAIN_TIME
    
    import time as time_module
    start_time = time_module.time()
    
    logger.info(f"=" * 60)
    logger.info(f"DUAL FILE INCREMENTAL LEARNING: Processing companies and invoices")
    logger.info(f"=" * 60)
    
    # Track upload
    TOTAL_UPLOADS += 1
    upload_record = {
        "upload_id": TOTAL_UPLOADS,
        "companies_file": str(companies_path) if companies_path else None,
        "invoices_file": str(invoices_path) if invoices_path else None,
        "timestamp": time_module.strftime("%Y-%m-%d %H:%M:%S")
    }
    UPLOAD_HISTORY.append(upload_record)
    
    # Record counts BEFORE processing
    companies_before = len(COMPANIES) if COMPANIES is not None else 0
    invoices_before = len(INVOICES) if INVOICES is not None else 0
    
    logger.info(f"State BEFORE: {companies_before} companies, {invoices_before} invoices")
    
    # Load companies data
    new_companies_df = pd.DataFrame()
    if companies_path and Path(companies_path).exists():
        new_companies_df = pd.read_csv(companies_path)
        logger.info(f"Loaded {len(new_companies_df)} companies from {companies_path}")
        
        # Clean and standardize company data
        if "company_id" in new_companies_df.columns:
            new_companies_df["company_id"] = new_companies_df["company_id"].astype(str).str.strip()
        elif "gstin" in new_companies_df.columns:
            new_companies_df["company_id"] = new_companies_df["gstin"].astype(str).str.strip()
        
        # Ensure turnover column
        if "turnover" not in new_companies_df.columns:
            if "avg_monthly_turnover" in new_companies_df.columns:
                new_companies_df["turnover"] = new_companies_df["avg_monthly_turnover"]
            else:
                new_companies_df["turnover"] = 0
        
        # Ensure is_fraud column
        if "is_fraud" not in new_companies_df.columns:
            new_companies_df["is_fraud"] = 0
        
        new_companies_df["turnover"] = pd.to_numeric(new_companies_df["turnover"], errors='coerce').fillna(0)
        new_companies_df["is_fraud"] = pd.to_numeric(new_companies_df["is_fraud"], errors='coerce').fillna(0).astype(int)
    
    # Load invoices data
    new_invoices_df = pd.DataFrame()
    if invoices_path and Path(invoices_path).exists():
        new_invoices_df = pd.read_csv(invoices_path)
        logger.info(f"Loaded {len(new_invoices_df)} invoices from {invoices_path}")
        
        # Clean and standardize invoice data
        new_invoices_df["seller_id"] = new_invoices_df["seller_id"].astype(str).str.strip()
        new_invoices_df["buyer_id"] = new_invoices_df["buyer_id"].astype(str).str.strip()
        
        if "amount" not in new_invoices_df.columns:
            new_invoices_df["amount"] = 0
        if "itc_claimed" not in new_invoices_df.columns:
            new_invoices_df["itc_claimed"] = 0
        
        new_invoices_df["amount"] = pd.to_numeric(new_invoices_df["amount"], errors='coerce').fillna(0)
        new_invoices_df["itc_claimed"] = pd.to_numeric(new_invoices_df["itc_claimed"], errors='coerce').fillna(0)
    
    # Engineer features from invoices for companies
    if not new_companies_df.empty and not new_invoices_df.empty:
        logger.info("Engineering transaction features from invoices...")
        
        # Seller statistics
        seller_stats = new_invoices_df.groupby("seller_id").agg({
            "amount": ["sum", "count"],
            "buyer_id": "nunique"
        }).reset_index()
        seller_stats.columns = ["company_id", "total_sent_amount", "sent_invoice_count", "unique_buyers"]
        seller_stats["company_id"] = seller_stats["company_id"].astype(str)
        
        # Buyer statistics
        buyer_stats = new_invoices_df.groupby("buyer_id").agg({
            "amount": ["sum", "count"],
            "seller_id": "nunique"
        }).reset_index()
        buyer_stats.columns = ["company_id", "total_received_amount", "received_invoice_count", "unique_sellers"]
        buyer_stats["company_id"] = buyer_stats["company_id"].astype(str)
        
        # Merge with companies
        new_companies_df = new_companies_df.merge(seller_stats, on="company_id", how="left")
        new_companies_df = new_companies_df.merge(buyer_stats, on="company_id", how="left")
        new_companies_df.fillna(0, inplace=True)
    
    # =========================================================================
    # ACCUMULATE DATA
    # =========================================================================
    logger.info("Accumulating data...")
    
    # Accumulate companies
    if not new_companies_df.empty:
        if COMPANIES is not None and len(COMPANIES) > 0:
            COMPANIES["company_id"] = COMPANIES["company_id"].astype(str).str.strip()
            combined = pd.concat([COMPANIES, new_companies_df], ignore_index=True)
            COMPANIES = combined.drop_duplicates(subset=["company_id"], keep="last")
        else:
            COMPANIES = new_companies_df.copy()
    
    # Accumulate invoices
    if not new_invoices_df.empty:
        if INVOICES is not None and len(INVOICES) > 0:
            if "invoice_id" in new_invoices_df.columns and "invoice_id" in INVOICES.columns:
                combined = pd.concat([INVOICES, new_invoices_df], ignore_index=True)
                INVOICES = combined.drop_duplicates(subset=["invoice_id"], keep="last")
            else:
                INVOICES = pd.concat([INVOICES, new_invoices_df], ignore_index=True)
        else:
            INVOICES = new_invoices_df.copy()
    
    # Record counts AFTER
    companies_after = len(COMPANIES) if COMPANIES is not None else 0
    invoices_after = len(INVOICES) if INVOICES is not None else 0
    new_companies_count = companies_after - companies_before
    new_invoices_count = invoices_after - invoices_before
    
    logger.info(f"State AFTER: {companies_after} companies (+{new_companies_count}), {invoices_after} invoices (+{new_invoices_count})")
    
    # =========================================================================
    # REBUILD GRAPH
    # =========================================================================
    logger.info("Rebuilding full graph...")
    NETWORKX_GRAPH, MAPPINGS, total_nodes, total_edges = rebuild_full_graph(COMPANIES, INVOICES)
    
    # =========================================================================
    # REBUILD PyG DATA
    # =========================================================================
    logger.info("Rebuilding PyG graph data...")
    GRAPH_DATA = rebuild_pyg_graph(NETWORKX_GRAPH, COMPANIES)
    
    # =========================================================================
    # RETRAIN MODEL
    # =========================================================================
    logger.info("Retraining model on full graph...")
    training_stats = retrain_full_model(GRAPH_DATA, epochs=100, lr=0.001)
    
    # =========================================================================
    # UPDATE FRAUD SCORES
    # =========================================================================
    logger.info("Updating fraud scores...")
    update_global_embeddings()
    
    # =========================================================================
    # PERSIST STATE
    # =========================================================================
    logger.info("Persisting state...")
    save_accumulated_data()
    save_updated_graph_and_model()
    
    LAST_RETRAIN_TIME = time_module.strftime("%Y-%m-%d %H:%M:%S")
    processing_time = time_module.time() - start_time
    
    high_risk_count = int((FRAUD_PROBA > 0.7).sum()) if FRAUD_PROBA is not None else 0
    fraud_count = int((COMPANIES["predicted_fraud"] == 1).sum()) if COMPANIES is not None and "predicted_fraud" in COMPANIES.columns else 0
    
    logger.info(f"=" * 60)
    logger.info(f"DUAL FILE PROCESSING COMPLETE")
    logger.info(f"Total: {companies_after} companies, {invoices_after} invoices")
    logger.info(f"Processing time: {processing_time:.2f}s")
    logger.info(f"=" * 60)
    
    return {
        "status": "success",
        "message": f"Successfully processed {new_companies_count} companies and {new_invoices_count} invoices",
        "new_nodes": new_companies_count,
        "new_edges": new_invoices_count,
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "datasets_uploaded": TOTAL_UPLOADS,
        "high_risk_count": high_risk_count,
        "fraud_count": fraud_count,
        "processing_time": round(processing_time, 2)
    }


# Number of features used in the GNN model (must match in_channels)
NUM_NODE_FEATURES = 12  # Updated feature count

def rebuild_full_graph(companies_df, invoices_df):
    """
    Rebuild the complete NetworkX graph from accumulated data.
    This ensures graph always represents the FULL accumulated state.
    Now with EXTENDED FEATURES for better fraud detection.
    """
    global NETWORKX_GRAPH, MAPPINGS
    
    logger.info("Rebuilding NetworkX graph from accumulated data with extended features...")
    
    # Create fresh graph
    G = nx.DiGraph()
    
    # Add all company nodes with EXTENDED FEATURES
    for _, row in companies_df.iterrows():
        try:
            company_id = str(row["company_id"]).strip()
            
            # Basic features
            turnover = float(row.get("turnover", row.get("avg_monthly_turnover", 0)))
            sent_invoices = float(row.get("sent_invoice_count", row.get("total_invoices_sent", 0)))
            received_invoices = float(row.get("received_invoice_count", row.get("total_invoices_received", 0)))
            total_sent_amount = float(row.get("total_sent_amount", row.get("total_amount_sent", 0)))
            total_received_amount = float(row.get("total_received_amount", row.get("total_amount_received", 0)))
            
            # Extended features from dataset (risk indicators)
            unique_buyers = float(row.get("unique_buyers", 0))
            unique_sellers = float(row.get("unique_sellers", 0))
            circular_trading_score = float(row.get("circular_trading_score", 0))
            gst_compliance_rate = float(row.get("gst_compliance_rate", 1.0))
            late_filing_count = float(row.get("late_filing_count", 0))
            round_amount_ratio = float(row.get("round_amount_ratio", 0))
            buyer_concentration = float(row.get("buyer_concentration", 0))
            
            G.add_node(
                company_id,
                # Basic features
                turnover=turnover,
                sent_invoices=sent_invoices,
                received_invoices=received_invoices,
                total_sent_amount=total_sent_amount,
                total_received_amount=total_received_amount,
                # Extended features
                unique_buyers=unique_buyers,
                unique_sellers=unique_sellers,
                circular_trading_score=circular_trading_score,
                gst_compliance_rate=gst_compliance_rate,
                late_filing_count=late_filing_count,
                round_amount_ratio=round_amount_ratio,
                buyer_concentration=buyer_concentration,
                # Labels
                location=str(row.get("location", row.get("city", "Unknown"))),
                is_fraud=int(row.get("is_fraud", 0))
            )
        except Exception as e:
            logger.warning(f"Error adding node {row.get('company_id')}: {e}")
            continue
    
    # Add all edges from invoices
    edges_added = 0
    if invoices_df is not None and len(invoices_df) > 0:
        for _, row in invoices_df.iterrows():
            try:
                seller = str(row["seller_id"]).strip()
                buyer = str(row["buyer_id"]).strip()
                
                # Add edge if both nodes exist
                if seller in G.nodes() and buyer in G.nodes():
                    # For multi-edges, we aggregate or add new edge
                    if G.has_edge(seller, buyer):
                        # Update existing edge with aggregated amount
                        G[seller][buyer]["amount"] += float(row.get("amount", 0))
                        G[seller][buyer]["itc_claimed"] += float(row.get("itc_claimed", 0))
                    else:
                        G.add_edge(
                            seller,
                            buyer,
                            amount=float(row.get("amount", 0)),
                            itc_claimed=float(row.get("itc_claimed", 0))
                        )
                        edges_added += 1
            except Exception as e:
                continue
    
    # Update mappings
    node_list = sorted(list(G.nodes()))
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    MAPPINGS = {
        "node_list": node_list,
        "node_to_idx": node_to_idx
    }
    
    NETWORKX_GRAPH = G
    
    logger.info(f"Graph rebuilt: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G, MAPPINGS, G.number_of_nodes(), G.number_of_edges()


def rebuild_pyg_graph(G, companies_df):
    """
    Convert NetworkX graph to PyTorch Geometric Data object.
    Now with EXTENDED FEATURES (12 features) and NORMALIZATION.
    """
    global GRAPH_DATA, NUM_NODE_FEATURES
    
    logger.info("Converting NetworkX to PyTorch Geometric format with extended features...")
    
    node_list = sorted(list(G.nodes()))
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    
    # Build feature matrix with EXTENDED FEATURES (12 features)
    x_list = []
    y_list = []
    
    for node in node_list:
        node_data = G.nodes[node]
        features = [
            # Basic features (5)
            float(node_data.get("turnover", 0)),
            float(node_data.get("sent_invoices", 0)),
            float(node_data.get("received_invoices", 0)),
            float(node_data.get("total_sent_amount", 0)),
            float(node_data.get("total_received_amount", 0)),
            # Extended features (7) - risk indicators
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
    # Replace NaN/Inf with zeros to avoid corrupting normalization and model inputs
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.tensor(y_list, dtype=torch.long)
    
    # Normalize features (important for GNN training)
    # Use min-max normalization to [0, 1] range
    x_min = x.min(dim=0, keepdim=True)[0]
    x_max = x.max(dim=0, keepdim=True)[0]
    x_range = x_max - x_min
    x_range[x_range == 0] = 1  # Avoid division by zero
    x = (x - x_min) / x_range
    
    logger.info(f"Feature matrix shape: {x.shape}, normalized to [0,1] range")
    
    # Build edge index
    edge_index = []
    edge_attr = []
    
    for u, v, data in G.edges(data=True):
        if u in node_to_idx and v in node_to_idx:
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            edge_index.append([u_idx, v_idx])
            edge_attr.append([float(data.get("amount", 0))])
    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )
    
    GRAPH_DATA = data
    
    logger.info(f"PyG graph: {data.num_nodes} nodes, {data.num_edges} edges, features shape {x.shape}")
    return data


def retrain_full_model(graph_data, epochs=200, lr=0.001):
    """
    Retrain model on the FULL accumulated graph.
    Uses class-weighted loss to handle imbalanced fraud/non-fraud classes.
    """
    global MODEL, DEVICE
    
    logger.info(f"Retraining model on full graph ({graph_data.num_nodes} nodes)...")
    
    # Move data to device
    graph_data = graph_data.to(DEVICE)
    
    # Compute class weights to address imbalance (fraud is minority class)
    y = graph_data.y.cpu().numpy()
    num_class_0 = (y == 0).sum()
    num_class_1 = (y == 1).sum()
    total = len(y)
    logger.info(f"Class distribution: Non-fraud={num_class_0} ({100*num_class_0/total:.1f}%), Fraud={num_class_1} ({100*num_class_1/total:.1f}%)")

    # Weight fraud class higher so model learns to detect it
    if num_class_1 > 0:
        weight_0 = 1.0
        weight_1 = min(num_class_0 / num_class_1, 10.0)  # Cap at 10x to avoid instability
    else:
        weight_0 = 1.0
        weight_1 = 1.0
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32, device=DEVICE)
    logger.info(f"Using class weights: non-fraud={weight_0:.2f}, fraud={weight_1:.2f}")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    
    # Create optimizer
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    # Training loop
    MODEL.train()
    best_loss = float('inf')
    patience = 30
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = MODEL(graph_data.x, graph_data.edge_index)
        loss = criterion(out, graph_data.y)
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss at epoch {epoch+1}")
            patience_counter += 1
            if patience_counter > patience:
                logger.info(f"Early stopping at epoch {epoch+1} due to NaN losses")
                break
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(MODEL.parameters(), max_norm=1.0)
        optimizer.step()
        
        scheduler.step(loss.item())
        
        # Track best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = MODEL.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 30 == 0:
            MODEL.eval()
            with torch.no_grad():
                pred = out.argmax(dim=1)
                correct = (pred == graph_data.y).sum().item()
                accuracy = correct / graph_data.num_nodes
            MODEL.train()
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Acc: {accuracy:.3f}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state is not None:
        MODEL.load_state_dict(best_model_state)
    
    MODEL.eval()
    
    # Final metrics
    with torch.no_grad():
        out = MODEL(graph_data.x, graph_data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred == graph_data.y).sum().item()
        final_accuracy = correct / graph_data.num_nodes
    
    logger.info(f"Training complete. Best loss: {best_loss:.4f}, Final Acc: {final_accuracy:.3f}")
    
    return {
        "final_loss": best_loss,
        "accuracy": final_accuracy,
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "epochs": epochs,
        "class_weights": None
    }


def save_accumulated_data():
    """
    Save accumulated companies and invoices data to disk.
    This ensures data persists across server restarts.
    """
    global COMPANIES, INVOICES, ACCUMULATED_DATA_PATH
    
    logger.info("Saving accumulated data...")
    
    # Ensure directory exists
    ACCUMULATED_DATA_PATH.mkdir(parents=True, exist_ok=True)
    
    # Save accumulated companies
    if COMPANIES is not None and len(COMPANIES) > 0:
        companies_path = ACCUMULATED_DATA_PATH / "companies_accumulated.csv"
        COMPANIES.to_csv(companies_path, index=False)
        logger.info(f"✓ Saved {len(COMPANIES)} companies to {companies_path}")
    
    # Save accumulated invoices
    if INVOICES is not None and len(INVOICES) > 0:
        invoices_path = ACCUMULATED_DATA_PATH / "invoices_accumulated.csv"
        INVOICES.to_csv(invoices_path, index=False)
        logger.info(f"✓ Saved {len(INVOICES)} invoices to {invoices_path}")
    
    # Save upload history
    history_path = ACCUMULATED_DATA_PATH / "upload_history.json"
    with open(history_path, "w") as f:
        json.dump(UPLOAD_HISTORY, f, indent=2)
    logger.info(f"✓ Saved upload history ({len(UPLOAD_HISTORY)} records)")


# Keep original update_graph for backward compatibility but mark as deprecated
def update_graph_deprecated(new_companies_df, new_invoices_df):
    """
    DEPRECATED: Use rebuild_full_graph instead.
    This function is kept for backward compatibility.
    """
    return rebuild_full_graph(new_companies_df, new_invoices_df if not new_invoices_df.empty else pd.DataFrame())


def identify_affected_nodes(new_companies_df, new_invoices_df, k_hop=2):
    """
    Identify nodes that are affected by new data (new nodes + neighbors within k-hop)
    Returns list of affected node IDs
    """
    global NETWORKX_GRAPH
    
    logger.info("Identifying affected nodes...")
    
    # Start with new nodes
    new_node_ids = set()
    for _, row in new_companies_df.iterrows():
        try:
            new_node_ids.add(row["company_id"])  # Keep as string
        except:
            continue
    
    # Add nodes connected through new edges
    for _, row in new_invoices_df.iterrows():
        try:
            seller = row["seller_id"]  # Keep as string
            buyer = row["buyer_id"]    # Keep as string
            new_node_ids.add(seller)
            new_node_ids.add(buyer)
        except:
            continue
    
    # Find k-hop neighbors using BFS
    affected_nodes = set(new_node_ids)
    queue = deque([(node, 0) for node in new_node_ids])  # (node, distance)
    visited = set(new_node_ids)
    
    while queue:
        node, distance = queue.popleft()
        
        if distance < k_hop:
            # Get neighbors (both incoming and outgoing)
            neighbors = set(NETWORKX_GRAPH.successors(node)) | set(NETWORKX_GRAPH.predecessors(node))
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    affected_nodes.add(neighbor)
                    queue.append((neighbor, distance + 1))
    
    logger.info(f"Identified {len(affected_nodes)} affected nodes within {k_hop}-hop neighborhood")
    return list(affected_nodes)


def extract_subgraph(affected_nodes, k_hop=2):
    """
    Extract k-hop subgraph around affected nodes
    Returns NetworkX subgraph
    """
    global NETWORKX_GRAPH
    
    logger.info(f"Extracting {k_hop}-hop subgraph for {len(affected_nodes)} nodes...")
    
    # Get k-hop neighborhood using BFS
    subgraph_nodes = set(affected_nodes)
    queue = deque([(node, 0) for node in affected_nodes])  # (node, distance)
    visited = set(affected_nodes)
    
    while queue:
        node, distance = queue.popleft()
        
        if distance < k_hop:
            # Get neighbors (both incoming and outgoing)
            neighbors = set(NETWORKX_GRAPH.successors(node)) | set(NETWORKX_GRAPH.predecessors(node))
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph_nodes.add(neighbor)
                    queue.append((neighbor, distance + 1))
    
    # Extract subgraph
    subgraph = NETWORKX_GRAPH.subgraph(subgraph_nodes).copy()
    logger.info(f"Extracted subgraph with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")
    
    return subgraph


def networkx_to_pytorch_geometric_subgraph(G, full_node_to_idx):
    """
    Convert NetworkX subgraph to PyTorch Geometric Data object
    """
    logger.info("Converting subgraph to PyTorch Geometric format...")
    
    # Create node list and feature matrix for subgraph
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
    
    logger.info(f"Subgraph node feature matrix shape: {x.shape}")
    
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
    
    logger.info(f"Subgraph edge index shape: {edge_index.shape}")
    
    # Create PyG Data object (without node_ids since they're strings)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )
    
    return data, node_list, node_to_idx


def incremental_retrain(subgraph_data, epochs=50, lr=0.001):
    """
    Retrain model on subgraph data
    Returns updated model state dict
    """
    global MODEL, DEVICE
    
    logger.info("Starting incremental retraining on subgraph...")
    
    # Move data to device
    subgraph_data = subgraph_data.to(DEVICE)
    
    # Create optimizer (use same parameters as original training)
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Set model to training mode
    MODEL.train()
    
    # Train for specified epochs
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = MODEL(subgraph_data.x, subgraph_data.edge_index)
        loss = criterion(out, subgraph_data.y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Incremental retraining epoch {epoch+1}/{epochs}, loss: {loss.item():.4f}")
    
    # Set model back to eval mode
    MODEL.eval()
    
    logger.info("Incremental retraining completed")
    return MODEL.state_dict()


def update_global_embeddings():
    """
    Update global fraud probabilities for ALL nodes after training.
    This ensures all companies (old and new) get updated fraud scores.
    """
    global MODEL, GRAPH_DATA, DEVICE, FRAUD_PROBA, COMPANIES, MAPPINGS
    
    logger.info("Updating fraud probabilities for ALL companies...")
    
    # Get updated predictions for all nodes
    MODEL.eval()
    with torch.no_grad():
        out = MODEL(GRAPH_DATA.x.to(DEVICE), GRAPH_DATA.edge_index.to(DEVICE))
        predictions = torch.softmax(out, dim=1)
        new_fraud_proba = predictions[:, 1].cpu().numpy()
    
    # Ensure FRAUD_PROBA matches current company count
    num_companies = len(COMPANIES)
    num_proba = len(new_fraud_proba)
    
    if num_proba != num_companies:
        logger.warning(f"Mismatch: {num_companies} companies vs {num_proba} predictions")
        # Align by padding or truncating
        if num_proba < num_companies:
            # Pad with zeros for new companies not in graph yet
            padded = np.zeros(num_companies)
            padded[:num_proba] = new_fraud_proba
            FRAUD_PROBA = padded
        else:
            # Truncate to match companies
            FRAUD_PROBA = new_fraud_proba[:num_companies]
    else:
        FRAUD_PROBA = new_fraud_proba
    
    # Map predictions back to companies using node mappings
    if MAPPINGS is not None and "node_list" in MAPPINGS:
        node_list = MAPPINGS["node_list"]
        # Create a mapping from company_id to fraud probability
        fraud_proba_dict = {}
        for i, node_id in enumerate(node_list):
            if i < len(new_fraud_proba):
                fraud_proba_dict[str(node_id)] = float(new_fraud_proba[i])
        
        # Update COMPANIES dataframe using the mapping
        COMPANIES["company_id"] = COMPANIES["company_id"].astype(str).str.strip()
        COMPANIES["fraud_probability"] = COMPANIES["company_id"].map(fraud_proba_dict).fillna(0.0)
    else:
        # Fallback: direct assignment (assumes same order)
        COMPANIES["fraud_probability"] = FRAUD_PROBA
    
    # Use actual is_fraud labels if available, otherwise use model predictions
    if "is_fraud" in COMPANIES.columns:
        COMPANIES["predicted_fraud"] = COMPANIES["is_fraud"].astype(int)
    else:
        COMPANIES["predicted_fraud"] = (COMPANIES["fraud_probability"] > 0.5).astype(int)
    
    # Ensure location column exists for dashboard
    if "location" not in COMPANIES.columns:
        if "city" in COMPANIES.columns:
            COMPANIES["location"] = COMPANIES["city"]
        elif "state" in COMPANIES.columns:
            COMPANIES["location"] = COMPANIES["state"]
        else:
            COMPANIES["location"] = "Unknown"
    # Fill missing/blank locations to keep dashboard endpoints stable and avoid nulls in Plotly
    def _clean_location(val):
        if pd.isna(val):
            return "Unknown"
        s = str(val).strip()
        if s.lower() in {"", "nan", "none"}:
            return "Unknown"
        return s
    COMPANIES["location"] = COMPANIES["location"].apply(_clean_location)

    # Persist cleaned locations so subsequent loads don't reintroduce NaNs
    save_accumulated_data()
    
    high_risk = (COMPANIES["fraud_probability"] > 0.7).sum()
    fraud_count = (COMPANIES["predicted_fraud"] == 1).sum()
    
    logger.info(f"Updated {len(COMPANIES)} companies: {high_risk} high-risk, {fraud_count} predicted fraud")


def save_updated_graph_and_model():
    """
    Save updated graph, mappings, and model weights
    """
    global GRAPH_DATA, MAPPINGS, MODEL, NETWORKX_GRAPH
    
    data_path = Path(__file__).parent / "data" / "processed"
    models_path = Path(__file__).parent / "models"
    graph_path = data_path / "graphs"
    
    # Ensure the graphs directory exists
    graph_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Saving updated graph and model...")
    
    # Save NetworkX graph using pickle (NetworkX 3.x compatibility)
    with open(graph_path / "networkx_graph.gpickle", "wb") as f:
        pickle.dump(NETWORKX_GRAPH, f)
    logger.info("✓ NetworkX graph saved")
    
    # Convert NetworkX graph to PyTorch Geometric and save
    try:
        from src.graph_construction.build_graph import GraphBuilder
        builder = GraphBuilder(str(data_path))  # Pass the correct path
        pyg_data, node_list, node_to_idx = builder.networkx_to_pytorch_geometric(NETWORKX_GRAPH, COMPANIES)
        torch.save(pyg_data, graph_path / "graph_data.pt")
        logger.info("✓ PyTorch Geometric graph saved")
        
        # Update global GRAPH_DATA
        GRAPH_DATA = pyg_data
        
        # Save mappings
        mappings = {
            "node_list": node_list,
            "node_to_idx": node_to_idx
        }
        with open(graph_path / "node_mappings.pkl", "wb") as f:
            pickle.dump(mappings, f)
        logger.info("✓ Node mappings saved")
        
        MAPPINGS = mappings
    except Exception as e:
        logger.error(f"Error converting/saving PyTorch Geometric graph: {e}")
    
    # Save updated model
    try:
        torch.save(MODEL.state_dict(), models_path / "best_model.pt")
        logger.info("✓ Model weights saved")
    except Exception as e:
        logger.error(f"Error saving model weights: {e}")


@app.route('/uploads')
def uploads_list():
    try:
        init_db()
        items = list_uploads(limit=100)
        return jsonify(items)
    except Exception as e:
        logger.error(f"Error listing uploads: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/accumulation_status')
def accumulation_status():
    """
    API: Get current accumulation status - shows how data has been accumulated over uploads.
    This is the key endpoint that shows the dashboard's accumulated state.
    """
    try:
        status = {
            "total_uploads": TOTAL_UPLOADS,
            "total_companies": len(COMPANIES) if COMPANIES is not None else 0,
            "total_invoices": len(INVOICES) if INVOICES is not None else 0,
            "graph_nodes": NETWORKX_GRAPH.number_of_nodes() if NETWORKX_GRAPH else 0,
            "graph_edges": NETWORKX_GRAPH.number_of_edges() if NETWORKX_GRAPH else 0,
            "last_retrain": LAST_RETRAIN_TIME,
            "upload_history": UPLOAD_HISTORY[-10:],  # Last 10 uploads
            "high_risk_count": int((FRAUD_PROBA > 0.7).sum()) if FRAUD_PROBA is not None else 0,
            "fraud_count": int((COMPANIES["predicted_fraud"] == 1).sum()) if COMPANIES is not None and "predicted_fraud" in COMPANIES.columns else 0,
            "accumulated_data_path": str(ACCUMULATED_DATA_PATH),
            "has_accumulated_companies": (ACCUMULATED_DATA_PATH / "companies_accumulated.csv").exists(),
            "has_accumulated_invoices": (ACCUMULATED_DATA_PATH / "invoices_accumulated.csv").exists()
        }
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting accumulation status: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/reset_accumulation', methods=['POST'])
def reset_accumulation():
    """
    API: Reset accumulated data to start fresh.
    Use with caution - this clears all accumulated state!
    """
    global COMPANIES, INVOICES, NETWORKX_GRAPH, GRAPH_DATA, MAPPINGS, FRAUD_PROBA
    global UPLOAD_HISTORY, TOTAL_UPLOADS, LAST_RETRAIN_TIME
    
    try:
        # Clear accumulated files
        import shutil
        if ACCUMULATED_DATA_PATH.exists():
            shutil.rmtree(ACCUMULATED_DATA_PATH)
            ACCUMULATED_DATA_PATH.mkdir(parents=True, exist_ok=True)
        
        # Reset global state
        UPLOAD_HISTORY = []
        TOTAL_UPLOADS = 0
        LAST_RETRAIN_TIME = None
        
        # Reload original data
        load_model_and_data()
        
        logger.info("Accumulation reset successfully")
        return jsonify({
            "status": "success",
            "message": "Accumulated data cleared. System reset to initial state.",
            "companies_count": len(COMPANIES) if COMPANIES is not None else 0
        })
    except Exception as e:
        logger.error(f"Error resetting accumulation: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/high_risk_companies')
def get_high_risk_companies():
    """
    API: Get all high-risk companies from accumulated data.
    Returns companies with fraud_probability > 0.7
    """
    try:
        if COMPANIES is None or len(COMPANIES) == 0:
            return jsonify([])
        
        high_risk = COMPANIES[COMPANIES["fraud_probability"] > 0.7].copy()
        high_risk = high_risk.sort_values("fraud_probability", ascending=False)
        
        result = []
        for _, row in high_risk.iterrows():
            result.append({
                "company_id": str(row["company_id"]),
                "fraud_probability": float(row["fraud_probability"]),
                "risk_level": "CRITICAL" if row["fraud_probability"] > 0.9 else "HIGH",
                "turnover": float(row.get("turnover", 0)),
                "location": str(row.get("location", "Unknown")),
                "is_fraud_labeled": int(row.get("is_fraud", 0))
            })
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error getting high-risk companies: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ROUTES - Pages
# ============================================================================

@app.route("/")
def home():
    """Serve landing page as default"""
    return render_template('landing.html')


# Serve static files (CSS, JS, images) from static folder
@app.route("/static/<path:path>")
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

# Serve React assets if React build exists
@app.route("/assets/<path:path>")
def serve_react_assets(path):
    """Serve React static assets (JS, CSS, etc.)"""
    react_static_path = Path(__file__).parent / "static" / "react" / "assets"
    if (react_static_path / path).exists():
        return send_from_directory(str(react_static_path), path)
    return "Not Found", 404


@app.route("/dashboard")
def dashboard():
    """Dashboard route - serves React or template"""
    react_build_path = Path(__file__).parent / "static" / "react" / "index.html"
    if react_build_path.exists():
        return send_from_directory(str(react_build_path.parent), "index.html")
    # Fallback to template
    if FRAUD_PROBA is None or COMPANIES is None or len(FRAUD_PROBA) == 0:
        logger.error(f"Dashboard: FRAUD_PROBA is {FRAUD_PROBA}, COMPANIES len={len(COMPANIES) if COMPANIES is not None else 0}")
        return render_template("index.html",
                             total_companies=0,
                             high_risk_count=0,
                             fraud_count=0,
                             avg_risk="0%")
    
    high_risk_count = (FRAUD_PROBA > 0.5).sum()
    avg_risk = FRAUD_PROBA.mean()
    fraud_count = (COMPANIES["predicted_fraud"] == 1).sum()
    
    logger.info(f"Dashboard: {len(COMPANIES)} companies, {int(high_risk_count)} high-risk, {int(fraud_count)} fraud, avg_risk={avg_risk:.2%}")
    return render_template("index.html",
                         total_companies=len(COMPANIES),
                         high_risk_count=int(high_risk_count),
                         fraud_count=int(fraud_count),
                         avg_risk=f"{avg_risk:.2%}")


@app.route("/companies")
def companies():
    """Companies route - serves React or template"""
    react_build_path = Path(__file__).parent / "static" / "react" / "index.html"
    if react_build_path.exists():
        return send_from_directory(str(react_build_path.parent), "index.html")
    return render_template("companies.html")


@app.route('/chatbot')
def chatbot_page():
    """Render chatbot page"""
    return render_template('chatbot.html')

@app.route('/landing')
def landing_page():
    """Render modern landing page"""
    return render_template('landing.html')

# ============================================================================
# ROUTES - API
# ============================================================================

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
        # Check if data is loaded
        if FRAUD_PROBA is None or len(FRAUD_PROBA) == 0:
            logger.error("FRAUD_PROBA is not initialized")
            return jsonify({"error": "Model not loaded", "total_companies": 0}), 500
        
        if GRAPH_DATA is None:
            logger.error("GRAPH_DATA is not initialized")
            return jsonify({"error": "Graph data not loaded", "total_companies": 0}), 500
        
        if COMPANIES is None or len(COMPANIES) == 0:
            logger.error("COMPANIES is not initialized")
            return jsonify({"error": "Companies data not loaded", "total_companies": 0}), 500
        
        high_risk = (FRAUD_PROBA > 0.7).sum()
        medium_risk = ((FRAUD_PROBA > 0.3) & (FRAUD_PROBA <= 0.7)).sum()
        low_risk = (FRAUD_PROBA <= 0.3).sum()
        
        stats = {
            "total_companies": len(COMPANIES),
            "total_edges": int(GRAPH_DATA.num_edges),
            "high_risk_count": int(high_risk),
            "medium_risk_count": int(medium_risk),
            "low_risk_count": int(low_risk),
            "fraud_count": int((COMPANIES["predicted_fraud"] == 1).sum()),
            "average_fraud_probability": float(np.mean(FRAUD_PROBA))
        }
        
        logger.info(f"Returning statistics: total_companies={stats['total_companies']}, total_edges={stats['total_edges']}")
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error in get_statistics: {e}", exc_info=True)
        return jsonify({"error": str(e), "total_companies": 0}), 500


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
        df = COMPANIES.copy()
        df["location"] = df["location"].fillna("Unknown")

        fig = px.box(
            df,
            x="location",
            y="fraud_probability",
            title="Fraud Probability Distribution by Location",
            labels={"fraud_probability": "Fraud Probability", "location": "Location"}
        )
        fig.update_layout(height=400)
        
        # Serialize with custom encoder
        fig_dict = fig.to_dict()
        return Response(json.dumps(fig_dict, cls=NumpyEncoder), mimetype='application/json')
    
    except Exception as e:
        logger.error(f"Error in chart_risk_by_location: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/chart/turnover_vs_risk")
def chart_turnover_vs_risk():
    """API: Turnover vs Risk scatter plot - returns Plotly JSON"""
    try:
        df = COMPANIES.copy()
        df["location"] = df["location"].fillna("Unknown")

        fig = px.scatter(
            df,
            x="turnover",
            y="fraud_probability",
            color="predicted_fraud",
            title="Company Turnover vs Fraud Risk",
            labels={"turnover": "Turnover (₹)", "fraud_probability": "Fraud Probability"},
            hover_data=["company_id", "location"]
        )
        fig.update_layout(height=400)
        
        # Serialize with custom encoder
        fig_dict = fig.to_dict()
        return Response(json.dumps(fig_dict, cls=NumpyEncoder), mimetype='application/json')
    
    except Exception as e:
        logger.error(f"Error in chart_turnover_vs_risk: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/top_senders_table")
def get_top_senders_table():
    """API: Top invoice senders as table data (JSON)"""
    try:
        # Check if seller_id column exists, if not use first column as sender
        sender_col = "seller_id" if "seller_id" in INVOICES.columns else INVOICES.columns[0]
        logger.info(f"Using column '{sender_col}' for sellers")
        
        top_senders = INVOICES.groupby(sender_col).size().nlargest(10)
        logger.info(f"Found {len(top_senders)} top senders")
        
        data = []
        for sender_id, count in top_senders.items():
            sender_id_str = str(sender_id).strip()
            company = COMPANIES[COMPANIES["company_id"].astype(str) == sender_id_str]
            if len(company) > 0:
                fraud_prob = float(company.iloc[0].get('fraud_probability', 0))
                data.append({
                    "company_id": sender_id_str,
                    "invoice_count": int(count),
                    "fraud_probability": f"{fraud_prob:.2%}"
                })
        
        logger.info(f"Returning {len(data)} senders with matching companies")
        return jsonify(data)
    
    except Exception as e:
        logger.error(f"Error in get_top_senders_table: {e}", exc_info=True)
        return jsonify({"error": str(e), "invoices_cols": list(INVOICES.columns)}), 500


@app.route("/api/top_receivers_table")
def get_top_receivers_table():
    """API: Top invoice receivers as table data (JSON)"""
    try:
        # Check if buyer_id column exists, if not use second column as receiver
        receiver_col = "buyer_id" if "buyer_id" in INVOICES.columns else (INVOICES.columns[1] if len(INVOICES.columns) > 1 else INVOICES.columns[0])
        logger.info(f"Using column '{receiver_col}' for receivers")
        
        top_receivers = INVOICES.groupby(receiver_col).size().nlargest(10)
        logger.info(f"Found {len(top_receivers)} top receivers")
        
        data = []
        for receiver_id, count in top_receivers.items():
            receiver_id_str = str(receiver_id).strip()
            company = COMPANIES[COMPANIES["company_id"].astype(str) == receiver_id_str]
            if len(company) > 0:
                fraud_prob = float(company.iloc[0].get('fraud_probability', 0))
                data.append({
                    "company_id": receiver_id_str,
                    "invoice_count": int(count),
                    "fraud_probability": f"{fraud_prob:.2%}"
                })
        
        logger.info(f"Returning {len(data)} receivers with matching companies")
        return jsonify(data)
    
    except Exception as e:
        logger.error(f"Error in get_top_receivers_table: {e}", exc_info=True)
        return jsonify({"error": str(e), "invoices_cols": list(INVOICES.columns)}), 500


@app.route("/api/top_senders")
def get_top_senders():
    """API: Top invoice senders - returns Plotly chart data"""
    try:
        if INVOICES is None or len(INVOICES) == 0:
            logger.warning("INVOICES is empty, attempting to reload data...")
            load_model_and_data()
            if INVOICES is None or len(INVOICES) == 0:
                logger.warning("INVOICES is still empty after reload")
                return jsonify({"error": "No invoice data available"}), 404
        
        if "seller_id" not in INVOICES.columns:
            logger.error(f"Missing seller_id column. Available columns: {list(INVOICES.columns)}")
            return jsonify({"error": f"Missing seller_id column. Available: {list(INVOICES.columns)}"}), 400
        
        top_senders = INVOICES.groupby("seller_id").size().nlargest(10)
        
        if len(top_senders) == 0:
            logger.warning("No top senders found after grouping")
            return jsonify({"error": "No sender data available"}), 404
        
        company_ids = []
        counts = []
        colors = []
        
        for seller_id, count in top_senders.items():
            seller_id_str = str(seller_id).strip()
            company_ids.append(seller_id_str)
            counts.append(int(count))
            
            # Try to find matching company for color coding
            if COMPANIES is not None and len(COMPANIES) > 0 and "company_id" in COMPANIES.columns:
                company = COMPANIES[COMPANIES["company_id"].astype(str).str.strip() == seller_id_str]
                if len(company) > 0 and 'fraud_probability' in company.columns:
                    fraud_prob = float(company.iloc[0]['fraud_probability'])
                    if fraud_prob > 0.7:
                        colors.append('#FF4444')  # Red for high risk
                    elif fraud_prob > 0.3:
                        colors.append('#FF9932')  # Orange for medium risk
                    else:
                        colors.append('#114C5A')  # Blue for low risk
                else:
                    colors.append('#6C757D')  # Gray for unknown
            else:
                colors.append('#114C5A')  # Default color
        
        # Ensure we have data
        if len(company_ids) == 0:
            logger.error("No company IDs collected")
            return jsonify({"error": "No data to display"}), 404
        
        fig = go.Figure(data=[
            go.Bar(
                x=company_ids,
                y=counts,
                marker=dict(color=colors if len(colors) == len(counts) else '#114C5A'),
                text=counts,
                textposition='outside',
                textfont=dict(size=11)
            )
        ])
        fig.update_layout(
            title="Top 10 Invoice Senders",
            xaxis_title="Company ID",
            yaxis_title="Invoice Count",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#172B36', family='Inter, sans-serif'),
            xaxis=dict(tickangle=-45),
            yaxis=dict(range=[0, max(counts) * 1.15] if counts else [0, 10]),
            margin=dict(b=100, l=60, r=20, t=60)
        )
        
        return jsonify(fig.to_dict())
    
    except Exception as e:
        logger.error(f"Error in get_top_senders: {e}", exc_info=True)
        return jsonify({"error": str(e), "traceback": str(e.__traceback__)}), 500


@app.route("/api/top_receivers")
def get_top_receivers():
    """API: Top invoice receivers - returns Plotly chart data"""
    try:
        if INVOICES is None or len(INVOICES) == 0:
            logger.warning("INVOICES is empty, attempting to reload data...")
            load_model_and_data()
            if INVOICES is None or len(INVOICES) == 0:
                logger.warning("INVOICES is still empty after reload")
                return jsonify({"error": "No invoice data available"}), 404
        
        if "buyer_id" not in INVOICES.columns:
            logger.error(f"Missing buyer_id column. Available columns: {list(INVOICES.columns)}")
            return jsonify({"error": f"Missing buyer_id column. Available: {list(INVOICES.columns)}"}), 400
        
        top_receivers = INVOICES.groupby("buyer_id").size().nlargest(10)
        
        if len(top_receivers) == 0:
            logger.warning("No top receivers found after grouping")
            return jsonify({"error": "No receiver data available"}), 404
        
        company_ids = []
        counts = []
        colors = []
        
        for buyer_id, count in top_receivers.items():
            buyer_id_str = str(buyer_id).strip()
            company_ids.append(buyer_id_str)
            counts.append(int(count))
            
            # Try to find matching company for color coding
            if COMPANIES is not None and len(COMPANIES) > 0 and "company_id" in COMPANIES.columns:
                company = COMPANIES[COMPANIES["company_id"].astype(str).str.strip() == buyer_id_str]
                if len(company) > 0 and 'fraud_probability' in company.columns:
                    fraud_prob = float(company.iloc[0]['fraud_probability'])
                    if fraud_prob > 0.7:
                        colors.append('#FF4444')  # Red for high risk
                    elif fraud_prob > 0.3:
                        colors.append('#FF9932')  # Orange for medium risk
                    else:
                        colors.append('#114C5A')  # Blue for low risk
                else:
                    colors.append('#6C757D')  # Gray for unknown
            else:
                colors.append('#FF6B6B')  # Default coral color
        
        # Ensure we have data
        if len(company_ids) == 0:
            logger.error("No company IDs collected")
            return jsonify({"error": "No data to display"}), 404
        
        fig = go.Figure(data=[
            go.Bar(
                x=company_ids,
                y=counts,
                marker=dict(color=colors if len(colors) == len(counts) else '#FF6B6B'),
                text=counts,
                textposition='outside',
                textfont=dict(size=11)
            )
        ])
        fig.update_layout(
            title="Top 10 Invoice Receivers",
            xaxis_title="Company ID",
            yaxis_title="Invoice Count",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#172B36', family='Inter, sans-serif'),
            xaxis=dict(tickangle=-45),
            yaxis=dict(range=[0, max(counts) * 1.15] if counts else [0, 10]),
            margin=dict(b=100, l=60, r=20, t=60)
        )
        
        return jsonify(fig.to_dict())
    
    except Exception as e:
        logger.error(f"Error in get_top_receivers: {e}", exc_info=True)
        return jsonify({"error": str(e), "traceback": str(e.__traceback__)}), 500


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


@app.route("/api/chatbot", methods=['POST'])
def chatbot_api():
    """API endpoint for chatbot queries"""
    try:
        # Get the user message
        data = request.get_json()
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Import Groq client
        from groq import Groq
        
        # Initialize Groq client - use environment variable for API key
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
        if not GROQ_API_KEY:
            return jsonify({'error': 'GROQ_API_KEY environment variable not set'}), 500
        client = Groq(api_key=GROQ_API_KEY)
        
        # Get data statistics for context
        def get_data_statistics():
            stats = []
            
            if COMPANIES is not None:
                stats.append(f"Companies Dataset: {len(COMPANIES)} records")
                if "is_fraud" in COMPANIES.columns:
                    fraud_count = COMPANIES["is_fraud"].sum()
                    stats.append(f"Fraud Companies: {fraud_count} ({fraud_count/len(COMPANIES)*100:.2f}%)")
                if "turnover" in COMPANIES.columns:
                    stats.append(f"Total Turnover: ₹{COMPANIES['turnover'].sum():,.0f}")
                    stats.append(f"Average Turnover: ₹{COMPANIES['turnover'].mean():,.0f}")
                if "location" in COMPANIES.columns:
                    stats.append(f"Locations Covered: {COMPANIES['location'].nunique()}")
                    top_locations = COMPANIES["location"].value_counts().head(3)
                    stats.append(f"Top 3 Locations: {dict(top_locations)}")
            
            if INVOICES is not None:
                stats.append(f"Invoices Dataset: {len(INVOICES)} records")
                if "amount" in INVOICES.columns:
                    stats.append(f"Total Invoice Value: ₹{INVOICES['amount'].sum():,.0f}")
                    stats.append(f"Average Invoice Value: ₹{INVOICES['amount'].mean():,.0f}")
                if "itc_claimed" in INVOICES.columns:
                    stats.append(f"Total ITC Claims: ₹{INVOICES['itc_claimed'].sum():,.0f}")
                    stats.append(f"Average ITC per Invoice: ₹{INVOICES['itc_claimed'].mean():,.0f}")
            
            return "\n".join(stats)
        
        # Enhanced context for the LLM
        system_context = (
            "You are a GST tax compliance and fraud detection expert assistant. "
            "You have access to a dataset of companies and their invoices. "
            "Provide accurate, data-driven responses based on the following information:\n\n"
            f"=== DATASET STATISTICS ===\n{get_data_statistics()}\n\n"
            "Answer the user's question accurately and concisely."
        )
        
        # Prepare messages with context
        messages = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_message}
        ]
        
        # Call Groq API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        ai_response = response.choices[0].message.content
        
        return jsonify({'response': ai_response})
        
    except Exception as e:
        logger.error(f"Chatbot error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


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
    # Set debug=False to avoid auto-restart issues during testing
    app.run(debug=False, host="0.0.0.0", port=5000)