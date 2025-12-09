# 📊 NETRA TAX - COMPLETE PROJECT WORKFLOW & FILE DOCUMENTATION

## 📋 TABLE OF CONTENTS

1. [Project Overview](#project-overview)
2. [Complete Directory Structure](#complete-directory-structure)
3. [Data Flow Architecture](#data-flow-architecture)
4. [File-by-File Documentation](#file-by-file-documentation)
5. [Processing Workflows](#processing-workflows)
6. [API Endpoints Workflow](#api-endpoints-workflow)
7. [Frontend-Backend Integration](#frontend-backend-integration)
8. [Machine Learning Pipeline](#machine-learning-pipeline)
9. [Deployment Workflow](#deployment-workflow)

---

## 1. PROJECT OVERVIEW

### What is NETRA TAX?

**NETRA TAX** is an AI-powered Tax Fraud Detection Platform that uses Graph Neural Networks (GNN) to detect fraudulent patterns in GST/tax transactions. The system analyzes company networks as graphs to identify:

- Circular trading patterns
- Fraud rings and collusion
- Transaction anomalies
- Suspicious invoice patterns
- Tax evasion schemes

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML/AI** | PyTorch + PyTorch Geometric | Graph Neural Network model |
| **Backend** | FastAPI (Python 3.9+) | REST API server |
| **Frontend** | HTML5 + CSS3 + Vanilla JS | User interface |
| **Visualization** | D3.js v7 | Network graph visualization |
| **Data** | CSV + PyTorch tensors | Data storage |
| **Alternative Backend** | Flask | Legacy backend (tax-fraud-gnn) |
| **Authentication** | JWT tokens | Security |

### Project Statistics

- **Total Files**: 100+
- **Lines of Code**: 13,500+
- **API Endpoints**: 25+
- **Frontend Pages**: 8
- **Python Modules**: 30+
- **Documentation Files**: 10+
- **Data Files**: 4 CSV files
- **ML Models**: GNN with 6 fraud detection algorithms

---

## 2. COMPLETE DIRECTORY STRUCTURE

```
GNN_NETRATAX/
│
├── 📁 NETRA_TAX/                          ← PRIMARY APPLICATION (PRODUCTION-READY)
│   │
│   ├── 📁 backend/                        ← FastAPI Backend Application
│   │   ├── main.py                        (500+ lines - Complete API server)
│   │   │   ├─ FastAPI app initialization
│   │   │   ├─ CORS middleware
│   │   │   ├─ 25+ REST API endpoints
│   │   │   ├─ GNN model loading & inference
│   │   │   ├─ Fraud detection algorithms (6 patterns)
│   │   │   ├─ Network analysis functions
│   │   │   ├─ User authentication (JWT)
│   │   │   ├─ File upload handling
│   │   │   └─ Error handling & validation
│   │   │
│   │   ├── requirements.txt               (Backend dependencies)
│   │   │   ├─ fastapi==0.104.1
│   │   │   ├─ uvicorn==0.24.0
│   │   │   ├─ torch==2.1.0
│   │   │   ├─ torch-geometric==2.4.0
│   │   │   ├─ pandas, numpy, networkx
│   │   │   └─ pydantic, python-multipart
│   │   │
│   │   ├── 📁 app/                        (Modular app structure - optional)
│   │   ├── 📁 routers/                    (API route modules)
│   │   ├── 📁 services/                   (Business logic)
│   │   └── 📁 utils/                      (Helper functions)
│   │
│   ├── 📁 frontend/                       ← User Interface (HTML/CSS/JS)
│   │   │
│   │   ├── index.html                     (Main Dashboard - 500+ lines)
│   │   │   ├─ KPI metric cards (total entities, high risk, fraud rings)
│   │   │   ├─ Risk distribution pie chart
│   │   │   ├─ Fraud trend line chart
│   │   │   ├─ Score distribution bar chart
│   │   │   ├─ High-risk companies table
│   │   │   ├─ Auto-refresh (30 seconds)
│   │   │   └─ System health indicator
│   │   │
│   │   ├── company-explorer.html          (Company Search - 400+ lines)
│   │   │   ├─ Search by GSTIN or company name
│   │   │   ├─ Fraud score display (0-100)
│   │   │   ├─ Risk level indicator
│   │   │   ├─ Detected fraud patterns list
│   │   │   ├─ Connected entities count
│   │   │   ├─ Transaction volume metrics
│   │   │   └─ Link to network visualization
│   │   │
│   │   ├── invoice-explorer.html          (Invoice Search - 400+ lines)
│   │   │   ├─ Search by invoice ID, GSTIN, date range
│   │   │   ├─ Advanced filters (amount, risk level)
│   │   │   ├─ Fraud probability per invoice
│   │   │   ├─ Red flags identification
│   │   │   ├─ Supplier/buyer details
│   │   │   └─ Invoice details modal
│   │   │
│   │   ├── graph-visualizer.html          (Network Graph - 600+ lines)
│   │   │   ├─ D3.js force-directed graph
│   │   │   ├─ Interactive node dragging
│   │   │   ├─ Zoom controls (0.5x - 3x)
│   │   │   ├─ Pan functionality
│   │   │   ├─ Node coloring by risk (RED/ORANGE/GREEN)
│   │   │   ├─ Fraud ring highlighting (dark red)
│   │   │   ├─ Network statistics panel
│   │   │   ├─ Center on GSTIN feature
│   │   │   └─ Export to PNG
│   │   │
│   │   ├── reports.html                   (Report Generator - 400+ lines)
│   │   │   ├─ Report type selection (Comprehensive/Executive/Network)
│   │   │   ├─ GSTIN input for reports
│   │   │   ├─ PDF generation
│   │   │   ├─ Report management (list, download, delete)
│   │   │   └─ Report preview
│   │   │
│   │   ├── admin.html                     (Admin Panel - 500+ lines)
│   │   │   ├─ System health monitoring
│   │   │   ├─ User management (CRUD operations)
│   │   │   ├─ Role assignment
│   │   │   ├─ Logs viewer with filtering
│   │   │   ├─ System configuration
│   │   │   └─ Performance metrics
│   │   │
│   │   ├── login.html                     (Authentication - 300+ lines)
│   │   │   ├─ Login form
│   │   │   ├─ JWT token handling
│   │   │   ├─ Role-based redirects
│   │   │   ├─ Remember me functionality
│   │   │   └─ Error handling
│   │   │
│   │   ├── upload.html                    (File Upload - 400+ lines)
│   │   │   ├─ Drag-and-drop CSV upload
│   │   │   ├─ File validation (type, size, columns)
│   │   │   ├─ Processing progress bar
│   │   │   ├─ Validation results display
│   │   │   ├─ Data quality report
│   │   │   └─ Error reporting by line number
│   │   │
│   │   ├── 📁 js/                         (JavaScript files)
│   │   │   ├── api.js                     (API Client - 400+ lines)
│   │   │   │   ├─ HTTP request wrapper
│   │   │   │   ├─ JWT token management
│   │   │   │   ├─ Error handling
│   │   │   │   ├─ Base URL configuration
│   │   │   │   └─ Response formatting
│   │   │   │
│   │   │   └── dashboard.js               (Dashboard Logic - 300+ lines)
│   │   │       ├─ Chart initialization
│   │   │       ├─ Data fetching
│   │   │       ├─ Auto-refresh logic
│   │   │       └─ UI updates
│   │   │
│   │   ├── 📁 css/                        (Stylesheets)
│   │   │   └── style.css                  (Main Stylesheet - 1000+ lines)
│   │   │       ├─ Color scheme (Arctic Powder theme)
│   │   │       ├─ Layout (flexbox, grid)
│   │   │       ├─ Components (cards, tables, charts)
│   │   │       ├─ Responsive design
│   │   │       └─ Animations & transitions
│   │   │
│   │   ├── README.md                      (Frontend documentation)
│   │   └── QUICK_START.md                 (Frontend quick reference)
│   │
│   ├── 📁 docs/                           (Technical Documentation)
│   │   ├── ARCHITECTURE.md                (System architecture design)
│   │   └── QUICKSTART.md                  (Deployment guide)
│   │
│   ├── FEATURE_CHECKLIST.md               (All features documented)
│   └── INTEGRATION_GUIDE.md               (Setup and integration guide)
│
├── 📁 tax-fraud-gnn/                      ← MACHINE LEARNING CORE & DATA
│   │
│   ├── 📁 data/                           (Dataset storage)
│   │   ├── 📁 raw/                        (Original data)
│   │   │   ├─ invoices_raw.csv
│   │   │   └─ companies_raw.csv
│   │   │
│   │   ├── 📁 processed/                  (Processed data)
│   │   │   ├── companies_processed.csv    (Clean company data)
│   │   │   ├── invoices_processed.csv     (Clean invoice data)
│   │   │   │
│   │   │   └── 📁 graphs/                 (Graph data)
│   │   │       ├── graph_data.pt          (PyTorch Geometric graph)
│   │   │       │   ├─ Node features (company attributes)
│   │   │       │   ├─ Edge indices (transaction links)
│   │   │       │   ├─ Edge attributes (amounts, dates)
│   │   │       │   └─ Labels (fraud/normal)
│   │   │       │
│   │   │       └── node_mappings.pkl      (GSTIN ↔ Node ID mapping)
│   │   │
│   │   └── 📁 uploads/                    (User-uploaded files)
│   │       └── 📁 upload_TIMESTAMP/
│   │           ├── original.csv
│   │           ├── companies.csv
│   │           └── invoices.csv
│   │
│   ├── 📁 models/                         (Trained ML models)
│   │   └── best_model.pt                  (Trained GNN model - PyTorch)
│   │       ├─ Model architecture (GNNFraudDetector)
│   │       ├─ Trained weights
│   │       ├─ Node embeddings
│   │       └─ Classification layers
│   │
│   ├── 📁 src/                            (Source code modules)
│   │   │
│   │   ├── 📁 gnn_models/                 (GNN model definitions)
│   │   │   ├── __init__.py
│   │   │   └── train_gnn.py               (GNN Model Class - 300+ lines)
│   │   │       ├─ GNNFraudDetector class
│   │   │       ├─ GCN layers
│   │   │       ├─ GraphSAGE layers
│   │   │       ├─ Attention mechanism
│   │   │       ├─ Forward pass logic
│   │   │       └─ Training/evaluation methods
│   │   │
│   │   ├── 📁 data_processing/            (Data preprocessing)
│   │   │   ├── __init__.py
│   │   │   ├── clean_data.py              (Data cleaning)
│   │   │   ├── feature_engineering.py     (Feature creation)
│   │   │   └── validation.py              (Data validation)
│   │   │
│   │   ├── 📁 graph_construction/         (Graph building)
│   │   │   ├── __init__.py
│   │   │   └── build_graph.py             (Graph construction logic)
│   │   │       ├─ Create nodes from companies
│   │   │       ├─ Create edges from invoices
│   │   │       ├─ Add node features
│   │   │       ├─ Add edge attributes
│   │   │       └─ Save PyTorch Geometric Data
│   │   │
│   │   ├── 📁 api/                        (API module)
│   │   │   ├── __init__.py
│   │   │   └── app.py                     (Flask API - alternative backend)
│   │   │
│   │   ├── db.py                          (Database operations)
│   │   │   ├─ SQLite database setup
│   │   │   ├─ Upload tracking
│   │   │   └─ Query functions
│   │   │
│   │   └── crypto.py                      (File encryption/decryption)
│   │
│   ├── 📁 frontend/                       (React frontend - alternative)
│   │   ├── index.html
│   │   ├── vite.config.js                 (Vite configuration)
│   │   ├── package.json                   (Node dependencies)
│   │   └── 📁 src/
│   │       ├── App.jsx                    (Main React app)
│   │       ├── 📁 pages/                  (Page components)
│   │       └── 📁 components/             (Reusable components)
│   │
│   ├── 📁 templates/                      (Flask HTML templates)
│   │   ├── index.html                     (Dashboard template)
│   │   ├── companies.html                 (Companies page)
│   │   ├── analytics.html                 (Analytics page)
│   │   ├── upload.html                    (Upload page)
│   │   ├── landing.html                   (Landing page)
│   │   ├── chatbot.html                   (Chatbot interface)
│   │   ├── 404.html                       (Error page)
│   │   └── 500.html                       (Server error page)
│   │
│   ├── 📁 static/                         (Static assets for Flask)
│   │   ├── 📁 css/
│   │   │   ├── style.css                  (Main styles)
│   │   │   └── landing.css                (Landing page styles)
│   │   ├── 📁 js/
│   │   │   ├── dashboard.js               (Dashboard logic)
│   │   │   ├── companies.js               (Companies page logic)
│   │   │   ├── analytics.js               (Analytics logic)
│   │   │   ├── theme.js                   (Theme switcher)
│   │   │   └── landing.js                 (Landing page logic)
│   │   └── 📁 images/                     (Image assets)
│   │
│   ├── app.py                             (Flask Application - 800+ lines)
│   │   ├─ Flask app initialization
│   │   ├─ Model loading
│   │   ├─ Route definitions
│   │   ├─ API endpoints
│   │   ├─ File upload handling
│   │   └─ Error handling
│   │
│   ├── train_gnn_model.py                 (Model training script)
│   │   ├─ Load graph data
│   │   ├─ Initialize GNN model
│   │   ├─ Training loop
│   │   ├─ Validation
│   │   ├─ Save best model
│   │   └─ Performance metrics
│   │
│   ├── pipeline.py                        (Complete processing pipeline)
│   │   ├─ Data loading
│   │   ├─ Data cleaning
│   │   ├─ Graph construction
│   │   ├─ Model training
│   │   └─ Evaluation
│   │
│   ├── accuracy_model.py                  (Model evaluation)
│   ├── prepare_real_data.py               (Data preparation)
│   ├── test_backend.py                    (Backend tests)
│   ├── verify_setup.py                    (Setup verification)
│   ├── config.py                          (Configuration settings)
│   ├── requirements.txt                   (Python dependencies)
│   ├── setup.py                           (Setup script)
│   ├── setup.sh                           (Linux setup)
│   ├── setup.bat                          (Windows setup)
│   └── README.md                          (Module documentation)
│
├── 📁 chatbot/                            ← CHATBOT MODULE (SEPARATE)
│   ├── chatbot.py                         (Streamlit chatbot app)
│   ├── requirements.txt                   (Chatbot dependencies)
│   ├── README.md                          (Chatbot documentation)
│   └── 📁 .streamlit/                     (Streamlit config)
│
├── 📁 Data Files (Root)                   ← CSV DATASETS
│   ├── companies.csv                      (Company data - 49KB)
│   │   ├─ Columns: gstin, company_name, registration_date, address
│   │   └─ ~1,000 companies
│   │
│   ├── company_nodes.csv                  (Node features - 488KB)
│   │   ├─ Columns: node_id, gstin, features, labels
│   │   └─ Preprocessed for GNN
│   │
│   ├── invoices.csv                       (Invoice data - 317KB)
│   │   ├─ Columns: invoice_id, supplier_gstin, buyer_gstin,
│   │   │           amount, date, cgst, sgst, igst, itc_claimed
│   │   └─ ~5,000 invoices
│   │
│   └── invoice_edges.csv                  (Graph edges - 3.7MB)
│       ├─ Columns: source_node, target_node, weight, attributes
│       └─ Transaction relationships
│
├── 📁 Documentation Files (Root)          ← PROJECT DOCUMENTATION
│   ├── README.md                          (Main project README - 660 lines)
│   ├── 00_START_HERE.txt                  (Getting started guide)
│   ├── INDEX.md                           (Documentation index)
│   ├── QUICK_START.md                     (5-minute setup - 320 lines)
│   ├── INTEGRATION_GUIDE.md               (Full integration guide)
│   ├── SOLUTION_SUMMARY.md                (What was built)
│   ├── SYSTEM_STATUS.md                   (Current status)
│   ├── CHECKLIST.md                       (Implementation checklist)
│   └── ANALYSIS_COMPLETE.txt              (Analysis report)
│
├── 📁 Startup & Utility Scripts           ← HELPER SCRIPTS
│   ├── start_backend.bat                  (Windows backend startup)
│   │   ├─ Activate virtual environment
│   │   ├─ Install dependencies
│   │   ├─ Start FastAPI server
│   │   └─ Open API docs
│   │
│   ├── start_backend.sh                   (Linux/Mac backend startup)
│   ├── verify_system.py                   (System verification)
│   │   ├─ Check directory structure
│   │   ├─ Check required files
│   │   ├─ Check Python packages
│   │   ├─ Check port availability
│   │   └─ Generate diagnostic report
│   │
│   ├── startup_check.py                   (Post-startup verification)
│   │   ├─ Test API endpoints
│   │   ├─ Check model loading
│   │   ├─ Verify data access
│   │   └─ Health check
│   │
│   └── remove_chatbot.py                  (Cleanup script)
│
└── .gitignore                             (Git ignore rules)
    ├─ __pycache__/
    ├─ *.pyc
    ├─ venv/
    ├─ node_modules/
    └─ .env
```

---

## 3. DATA FLOW ARCHITECTURE

### Complete System Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER INTERACTION                            │
│  Browser → http://localhost:8080/index.html (Frontend)              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND LAYER                               │
│  (HTML/CSS/JavaScript - Port 8080)                                  │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   Dashboard  │  │   Company    │  │   Invoice    │             │
│  │  (index.html)│  │   Explorer   │  │   Explorer   │             │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘             │
│         │                  │                  │                      │
│         │  ┌───────────────┴──────────────────┘                     │
│         │  │                                                        │
│         ▼  ▼                                                        │
│  ┌─────────────────────────────────────┐                           │
│  │      api.js (API Client)             │                           │
│  │  - HTTP GET/POST wrapper             │                           │
│  │  - JWT token management              │                           │
│  │  - Error handling                    │                           │
│  └─────────────────┬────────────────────┘                           │
└────────────────────┼─────────────────────────────────────────────────┘
                     │
                     │ HTTP REST API
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         BACKEND LAYER                                │
│  (FastAPI - Port 8000)                                              │
│                                                                      │
│  NETRA_TAX/backend/main.py                                          │
│                                                                      │
│  ┌──────────────────────────────────────────────────────┐          │
│  │              FastAPI Application                      │          │
│  │  - CORS Middleware                                    │          │
│  │  - Request validation (Pydantic)                      │          │
│  │  - Response formatting                                │          │
│  └──────────────────────────────────────────────────────┘          │
│           │                                                          │
│           │                                                          │
│  ┌────────▼─────────────────────────────────────────┐              │
│  │           API ROUTERS (25+ endpoints)            │              │
│  ├──────────────────────────────────────────────────┤              │
│  │                                                   │              │
│  │  /api/auth/*          (Authentication)           │              │
│  │    ├─ POST /login     → Validate credentials     │              │
│  │    ├─ POST /signup    → Create user              │              │
│  │    └─ GET /user       → Get user info            │              │
│  │                                                   │              │
│  │  /api/fraud/*         (Fraud Detection) ⭐       │              │
│  │    ├─ GET /summary    → Dashboard metrics        │              │
│  │    ├─ GET /company/risk?gstin=XXX                │              │
│  │    │   └─ Calculate fraud score                  │              │
│  │    ├─ GET /invoice/risk?id=XXX                   │              │
│  │    │   └─ Calculate invoice risk                 │              │
│  │    ├─ GET /network/analysis?gstin=XXX            │              │
│  │    │   └─ Network analysis + fraud rings         │              │
│  │    ├─ GET /search/companies?query=XXX            │              │
│  │    └─ GET /search/invoices?query=XXX             │              │
│  │                                                   │              │
│  │  /api/graph/*         (Graph Analysis)           │              │
│  │    ├─ GET /network?gstin=XXX → D3.js graph data  │              │
│  │    ├─ GET /patterns?gstin=XXX → Fraud patterns   │              │
│  │    └─ GET /rings?gstin=XXX → Fraud rings         │              │
│  │                                                   │              │
│  │  /api/files/*         (File Upload)              │              │
│  │    ├─ POST /upload   → Upload CSV                │              │
│  │    ├─ POST /process  → Process & validate        │              │
│  │    └─ GET /list      → List uploads              │              │
│  │                                                   │              │
│  │  /api/reports/*       (PDF Reports)              │              │
│  │    ├─ POST /generate → Generate PDF report       │              │
│  │    └─ GET /download?id=XXX → Download PDF        │              │
│  │                                                   │              │
│  │  /api/system/*        (System Health)            │              │
│  │    ├─ GET /health    → Health check              │              │
│  │    ├─ GET /stats     → System statistics         │              │
│  │    └─ GET /logs      → View logs                 │              │
│  └───────────────────────────────────────────────────┘              │
│                                                                      │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MACHINE LEARNING CORE                            │
│  (GNN Model + Fraud Detection Algorithms)                           │
│                                                                      │
│  ┌───────────────────────────────────────────────────┐             │
│  │       STEP 1: Load Model & Data (on startup)      │             │
│  ├───────────────────────────────────────────────────┤             │
│  │  tax-fraud-gnn/models/best_model.pt               │             │
│  │    └─ GNNFraudDetector (PyTorch model)            │             │
│  │       ├─ GCN layers                                │             │
│  │       ├─ GraphSAGE layers                          │             │
│  │       ├─ Attention mechanism                       │             │
│  │       └─ Classification head                       │             │
│  │                                                    │             │
│  │  tax-fraud-gnn/data/processed/graphs/             │             │
│  │    ├─ graph_data.pt (PyTorch Geometric Data)      │             │
│  │    │   ├─ x: Node features [num_nodes, features]  │             │
│  │    │   ├─ edge_index: [2, num_edges]              │             │
│  │    │   ├─ edge_attr: Edge attributes              │             │
│  │    │   └─ y: Labels (fraud/normal)                │             │
│  │    │                                               │             │
│  │    └─ node_mappings.pkl                           │             │
│  │        └─ {gstin: node_id, ...}                   │             │
│  └────────────────────────────────────────────────────┘             │
│                                                                      │
│  ┌───────────────────────────────────────────────────┐             │
│  │       STEP 2: GNN Inference (per request)         │             │
│  ├───────────────────────────────────────────────────┤             │
│  │  Input: GSTIN or Invoice ID                       │             │
│  │     ↓                                              │             │
│  │  1. Map GSTIN → Node ID (using mappings.pkl)      │             │
│  │     ↓                                              │             │
│  │  2. Run GNN forward pass                          │             │
│  │     model(graph_data.x, graph_data.edge_index)    │             │
│  │     ↓                                              │             │
│  │  3. Get node embedding & fraud probability        │             │
│  │     output = model(x, edge_index)                 │             │
│  │     fraud_prob = sigmoid(output[node_id])         │             │
│  │     ↓                                              │             │
│  │  4. Return fraud score (0-1)                      │             │
│  └────────────────────────────────────────────────────┘             │
│                                                                      │
│  ┌───────────────────────────────────────────────────┐             │
│  │    STEP 3: Fraud Pattern Detection (6 algorithms) │             │
│  ├───────────────────────────────────────────────────┤             │
│  │                                                    │             │
│  │  1️⃣ Circular Trading Detection                    │             │
│  │     - Use NetworkX to detect cycles               │             │
│  │     - Find simple cycles in transaction graph     │             │
│  │     - Flag if company in any cycle                │             │
│  │                                                    │             │
│  │  2️⃣ High-Degree Node Detection                    │             │
│  │     - Count incoming/outgoing edges               │             │
│  │     - Compare to average degree                   │             │
│  │     - Flag if degree > threshold                  │             │
│  │                                                    │             │
│  │  3️⃣ Fraud Ring Detection                          │             │
│  │     - Community detection algorithm               │             │
│  │     - Identify tightly connected groups           │             │
│  │     - Check if fraud scores clustered high        │             │
│  │                                                    │             │
│  │  4️⃣ Chain Depth Analysis                          │             │
│  │     - BFS/DFS to trace invoice chains             │             │
│  │     - Measure chain length                        │             │
│  │     - Flag if chain too long                      │             │
│  │                                                    │             │
│  │  5️⃣ Transaction Spike Detection                   │             │
│  │     - Group transactions by time period           │             │
│  │     - Calculate moving average                    │             │
│  │     - Detect sudden spikes (> 2σ)                 │             │
│  │                                                    │             │
│  │  6️⃣ Clustering Coefficient Analysis               │             │
│  │     - Calculate local clustering coefficient      │             │
│  │     - Detect unusual grouping patterns            │             │
│  │     - Flag abnormal coefficients                  │             │
│  └────────────────────────────────────────────────────┘             │
│                                                                      │
│  ┌───────────────────────────────────────────────────┐             │
│  │       STEP 4: Combine Scores & Return             │             │
│  ├───────────────────────────────────────────────────┤             │
│  │  fraud_score = (                                   │             │
│  │      0.5 * gnn_score +                             │             │
│  │      0.2 * circular_trading_weight +               │             │
│  │      0.15 * high_degree_weight +                   │             │
│  │      0.15 * other_patterns_weight                  │             │
│  │  )                                                 │             │
│  │                                                    │             │
│  │  risk_level = {                                    │             │
│  │      fraud_score >= 0.7: "HIGH",                   │             │
│  │      fraud_score >= 0.4: "MEDIUM",                 │             │
│  │      fraud_score < 0.4: "LOW"                      │             │
│  │  }                                                 │             │
│  │                                                    │             │
│  │  Return JSON {                                     │             │
│  │      "fraud_score": 0.78,                          │             │
│  │      "risk_level": "HIGH",                         │             │
│  │      "fraud_factors": [...],                       │             │
│  │      "connected_entities": 23,                     │             │
│  │      "red_flags": [...]                            │             │
│  │  }                                                 │             │
│  └────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                       │
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
│                                                                      │
│  CSV Files (Root & tax-fraud-gnn/data/)                             │
│    ├─ companies.csv                                                 │
│    ├─ company_nodes.csv                                             │
│    ├─ invoices.csv                                                  │
│    └─ invoice_edges.csv                                             │
│                                                                      │
│  PyTorch Files (tax-fraud-gnn/data/processed/graphs/)               │
│    ├─ graph_data.pt (tensor format)                                │
│    └─ node_mappings.pkl (pickle format)                             │
│                                                                      │
│  Model Files (tax-fraud-gnn/models/)                                │
│    └─ best_model.pt (trained weights)                               │
└─────────────────────────────────────────────────────────────────────┘
```


### Workflow Sequence Diagram

```
User Request → Frontend (index.html)
                    ↓
           JavaScript (api.js) makes HTTP call
                    ↓
           http://localhost:8000/api/fraud/summary
                    ↓
           FastAPI Backend (main.py)
                    ↓
           Load GNN model & graph data
                    ↓
           Run GNN inference
                    ↓
           Run 6 fraud detection algorithms
                    ↓
           Combine scores
                    ↓
           Format JSON response
                    ↓
           Send back to frontend
                    ↓
           JavaScript updates UI (charts, tables)
                    ↓
           User sees results
```

---

## 4. FILE-BY-FILE DOCUMENTATION

### NETRA_TAX/backend/main.py (500+ lines)

**Purpose**: Production-ready FastAPI backend with complete fraud detection API

**Key Components**:

1. **Imports & Setup** (Lines 1-50)
   - FastAPI framework
   - PyTorch & PyTorch Geometric for GNN
   - Pandas, NumPy for data processing
   - NetworkX for graph analysis
   - Pydantic for data validation

2. **Pydantic Models** (Lines 52-150)
   ```python
   class CompanyRiskResponse(BaseModel):
       gstin: str
       fraud_score: float
       risk_level: str
       fraud_factors: List[str]
       connected_entities: int
   ```

3. **Global Variables** (Lines 152-160)
   - MODEL: Loaded GNN model
   - GRAPH_DATA: PyTorch Geometric graph
   - COMPANIES_DF: Company dataframe
   - INVOICES_DF: Invoice dataframe
   - NODE_MAPPINGS: GSTIN to node ID mapping

4. **Model Loading** (Lines 162-250)
   ```python
   def load_model_and_data():
       """Load GNN model and graph data on startup"""
       # Load best_model.pt
       # Load graph_data.pt
       # Load CSV files
       # Create mappings
   ```

5. **Fraud Detection Functions** (Lines 252-400)
   - `calculate_fraud_score(gstin)` - Main scoring function
   - `detect_circular_trading(gstin)` - Cycle detection
   - `detect_high_degree_nodes(gstin)` - Hub detection
   - `detect_fraud_rings(gstin)` - Community detection
   - `detect_spikes(gstin)` - Anomaly detection
   - `calculate_clustering_coefficient(gstin)` - Pattern detection

6. **API Endpoints** (Lines 402-500)

   **Authentication Endpoints**:
   - `POST /api/auth/login` - User login
   - `POST /api/auth/signup` - User registration
   - `GET /api/auth/user` - Get current user

   **Fraud Detection Endpoints**:
   - `GET /api/fraud/summary` - Dashboard summary
   - `GET /api/fraud/company/risk` - Company fraud score
   - `GET /api/fraud/invoice/risk` - Invoice risk score
   - `GET /api/fraud/network/analysis` - Network analysis
   - `GET /api/fraud/search/companies` - Search companies
   - `GET /api/fraud/search/invoices` - Search invoices

   **Graph Endpoints**:
   - `GET /api/graph/network` - D3.js graph data
   - `GET /api/graph/patterns` - Fraud patterns
   - `GET /api/graph/rings` - Fraud rings

   **File Endpoints**:
   - `POST /api/files/upload` - Upload CSV
   - `POST /api/files/process` - Process CSV
   - `GET /api/files/list` - List uploads

   **Report Endpoints**:
   - `POST /api/reports/generate` - Generate PDF
   - `GET /api/reports/download` - Download PDF

   **System Endpoints**:
   - `GET /api/health` - Health check
   - `GET /api/system/stats` - System stats

**Flow**: Startup → Load model → Listen for requests → Process → Return JSON

---

### NETRA_TAX/frontend/index.html (500+ lines)

**Purpose**: Main dashboard showing fraud metrics and charts

**Sections**:

1. **HTML Structure** (Lines 1-100)
   - Header with navigation
   - Metric cards (4 KPIs)
   - Chart containers (3 charts)
   - High-risk table
   - Footer

2. **CSS Styling** (Inline & external)
   - Flexbox layout
   - Grid system
   - Responsive design
   - Color scheme (Arctic Powder)

3. **JavaScript Logic** (Lines 200-500)
   ```javascript
   // On page load
   async function loadDashboard() {
       // Fetch summary data from API
       const data = await fetch('/api/fraud/summary');
       
       // Update metric cards
       updateMetricCards(data);
       
       // Render charts
       renderRiskDistribution(data);
       renderFraudTrend(data);
       renderScoreDistribution(data);
       
       // Populate table
       populateHighRiskTable(data);
   }
   
   // Auto-refresh every 30 seconds
   setInterval(loadDashboard, 30000);
   ```

4. **API Calls**:
   - GET `/api/fraud/summary` - Main dashboard data
   - GET `/api/system/stats` - System health

5. **Charts**:
   - Risk Distribution (Pie Chart) - Canvas API
   - Fraud Trend (Line Chart) - Canvas API
   - Score Distribution (Bar Chart) - Canvas API

**Flow**: Load → Fetch API → Render UI → Auto-refresh

---

### NETRA_TAX/frontend/graph-visualizer.html (600+ lines)

**Purpose**: Interactive D3.js network visualization with fraud highlighting

**Key Features**:

1. **D3.js Force-Directed Graph** (Lines 100-400)
   ```javascript
   // Create SVG
   const svg = d3.select("#graph")
       .append("svg")
       .attr("width", width)
       .attr("height", height);
   
   // Load graph data
   const graphData = await fetch(`/api/graph/network?gstin=${gstin}`);
   
   // Create force simulation
   const simulation = d3.forceSimulation(nodes)
       .force("link", d3.forceLink(links))
       .force("charge", d3.forceManyBody())
       .force("center", d3.forceCenter(width/2, height/2));
   
   // Draw nodes with color by fraud score
   const node = svg.selectAll(".node")
       .data(nodes)
       .enter().append("circle")
       .attr("r", 8)
       .attr("fill", d => getColorByScore(d.fraud_score))
       .call(drag);
   
   // Draw edges
   const link = svg.selectAll(".link")
       .data(links)
       .enter().append("line")
       .attr("stroke", "#999");
   
   // Update positions on tick
   simulation.on("tick", () => {
       node.attr("cx", d => d.x).attr("cy", d => d.y);
       link.attr("x1", d => d.source.x)
           .attr("y1", d => d.source.y)
           .attr("x2", d => d.target.x)
           .attr("y2", d => d.target.y);
   });
   ```

2. **Node Coloring** (Lines 450-480)
   ```javascript
   function getColorByScore(score) {
       if (score >= 0.7) return "#DC3545";  // RED (HIGH)
       if (score >= 0.4) return "#FF9932";  // ORANGE (MEDIUM)
       return "#28A745";  // GREEN (LOW)
   }
   ```

3. **Fraud Ring Highlighting** (Lines 500-550)
   - Detect cycles in graph
   - Highlight edges in dark red
   - Add visual indicators

4. **Interactive Features**:
   - Drag nodes to rearrange
   - Zoom slider (0.5x - 3x)
   - Pan with mouse
   - Click node for details
   - Export to PNG

5. **Statistics Panel** (Lines 560-600)
   - Total nodes count
   - Total edges count
   - Network density
   - Anomaly score
   - Fraud rings detected

**API Call**: GET `/api/graph/network?gstin=XXX`

**Flow**: Input GSTIN → Fetch graph → Render D3.js → User interaction

---

### NETRA_TAX/frontend/js/api.js (400+ lines)

**Purpose**: Centralized API client for all HTTP requests

**Structure**:

1. **Configuration** (Lines 1-20)
   ```javascript
   const API_BASE_URL = "http://localhost:8000";
   const AUTH_TOKEN_KEY = "netra_tax_token";
   
   class APIClient {
       constructor() {
           this.baseURL = API_BASE_URL;
           this.token = localStorage.getItem(AUTH_TOKEN_KEY);
       }
   }
   ```

2. **HTTP Methods** (Lines 22-150)
   ```javascript
   async request(method, endpoint, data = null) {
       const headers = {
           "Content-Type": "application/json"
       };
       
       if (this.token) {
           headers["Authorization"] = `Bearer ${this.token}`;
       }
       
       const options = {
           method,
           headers
       };
       
       if (data && method !== "GET") {
           options.body = JSON.stringify(data);
       }
       
       const response = await fetch(this.baseURL + endpoint, options);
       
       if (!response.ok) {
           throw new Error(`API Error: ${response.status}`);
       }
       
       return await response.json();
   }
   
   async get(endpoint) {
       return this.request("GET", endpoint);
   }
   
   async post(endpoint, data) {
       return this.request("POST", endpoint, data);
   }
   ```

3. **API Methods** (Lines 152-400)
   ```javascript
   // Authentication
   async login(username, password) {
       const data = await this.post("/api/auth/login", {username, password});
       this.token = data.access_token;
       localStorage.setItem(AUTH_TOKEN_KEY, this.token);
       return data;
   }
   
   // Fraud Detection
   async getFraudSummary() {
       return this.get("/api/fraud/summary");
   }
   
   async getCompanyRisk(gstin) {
       return this.get(`/api/fraud/company/risk?gstin=${gstin}`);
   }
   
   async getInvoiceRisk(invoiceId) {
       return this.get(`/api/fraud/invoice/risk?id=${invoiceId}`);
   }
   
   async getNetworkAnalysis(gstin) {
       return this.get(`/api/fraud/network/analysis?gstin=${gstin}`);
   }
   
   // Graph
   async getNetworkGraph(gstin) {
       return this.get(`/api/graph/network?gstin=${gstin}`);
   }
   
   // File Upload
   async uploadFile(file) {
       const formData = new FormData();
       formData.append("file", file);
       
       const response = await fetch(this.baseURL + "/api/files/upload", {
           method: "POST",
           headers: {
               "Authorization": `Bearer ${this.token}`
           },
           body: formData
       });
       
       return await response.json();
   }
   
   // System
   async getSystemHealth() {
       return this.get("/api/health");
   }
   ```

4. **Error Handling** (Lines 402-450)
   - Network errors
   - Authentication errors
   - Validation errors
   - Server errors

**Usage in Frontend**:
```javascript
const api = new APIClient();

// In dashboard
const summary = await api.getFraudSummary();

// In company explorer
const risk = await api.getCompanyRisk("1234567890GST");

// In graph visualizer
const graph = await api.getNetworkGraph("1234567890GST");
```

---

### tax-fraud-gnn/src/gnn_models/train_gnn.py (300+ lines)

**Purpose**: GNN model definition and training logic

**GNNFraudDetector Class** (Lines 1-150):

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool

class GNNFraudDetector(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=2):
        super(GNNFraudDetector, self).__init__()
        
        # Graph Convolutional Layers
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # Classification layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index, batch=None):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 3 (GraphSAGE)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        
        # Attention (optional)
        # x = self.attention(x, x, x)[0]
        
        # Classification
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x  # Returns logits [num_nodes, num_classes]
```

**Training Function** (Lines 152-250):

```python
def train_model(model, data, optimizer, criterion, epochs=100):
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index)
        
        # Calculate loss (only on labeled nodes)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation
        if epoch % 10 == 0:
            val_acc = evaluate(model, data, data.val_mask)
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Val Acc={val_acc:.4f}")
    
    return model

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum()
        acc = correct / mask.sum()
    return acc.item()
```

**Inference Function** (Lines 252-300):

```python
def predict_fraud_probability(model, data, node_id):
    """Get fraud probability for a specific node"""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probabilities = torch.softmax(out[node_id], dim=0)
        fraud_prob = probabilities[1].item()  # Probability of class 1 (fraud)
    return fraud_prob
```

**Usage in Backend**:
```python
# Load model
model = GNNFraudDetector(num_features=10)
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Get fraud score
fraud_score = predict_fraud_probability(model, graph_data, node_id)
```

---

### tax-fraud-gnn/src/graph_construction/build_graph.py (200+ lines)

**Purpose**: Build PyTorch Geometric graph from CSV data

**Main Function** (Lines 1-200):

```python
import pandas as pd
import torch
from torch_geometric.data import Data
import pickle

def build_graph_from_csv(companies_csv, invoices_csv, output_path):
    """
    Build PyTorch Geometric graph from company and invoice CSVs
    
    Args:
        companies_csv: Path to companies.csv
        invoices_csv: Path to invoices.csv
        output_path: Path to save graph_data.pt
    
    Returns:
        Data object, node_mappings dict
    """
    
    # STEP 1: Load data
    companies = pd.read_csv(companies_csv)
    invoices = pd.read_csv(invoices_csv)
    
    # STEP 2: Create node mappings
    # Map each unique GSTIN to a node ID (0, 1, 2, ...)
    unique_gstins = pd.concat([
        companies['gstin'],
        invoices['supplier_gstin'],
        invoices['buyer_gstin']
    ]).unique()
    
    node_mappings = {gstin: idx for idx, gstin in enumerate(unique_gstins)}
    num_nodes = len(node_mappings)
    
    # STEP 3: Create node features
    # For each company, create feature vector
    node_features = []
    for gstin in unique_gstins:
        company_data = companies[companies['gstin'] == gstin]
        
        if len(company_data) > 0:
            # Extract features from company data
            features = [
                # Registration age (days since registration)
                (pd.Timestamp.now() - pd.to_datetime(company_data.iloc[0]['registration_date'])).days,
                # Total invoices as supplier
                len(invoices[invoices['supplier_gstin'] == gstin]),
                # Total invoices as buyer
                len(invoices[invoices['buyer_gstin'] == gstin]),
                # Total amount as supplier
                invoices[invoices['supplier_gstin'] == gstin]['amount'].sum(),
                # Total amount as buyer
                invoices[invoices['buyer_gstin'] == gstin]['amount'].sum(),
                # Average invoice amount
                invoices[(invoices['supplier_gstin'] == gstin) | (invoices['buyer_gstin'] == gstin)]['amount'].mean(),
                # ITC claimed total
                invoices[invoices['buyer_gstin'] == gstin]['itc_claimed'].sum(),
                # Number of unique trading partners
                len(set(invoices[invoices['supplier_gstin'] == gstin]['buyer_gstin'].tolist() + 
                        invoices[invoices['buyer_gstin'] == gstin]['supplier_gstin'].tolist()))
            ]
        else:
            # Default features for unknown companies
            features = [0] * 8
        
        node_features.append(features)
    
    # Convert to tensor
    x = torch.tensor(node_features, dtype=torch.float)
    
    # STEP 4: Create edges (from invoices)
    edge_list = []
    edge_attrs = []
    
    for _, invoice in invoices.iterrows():
        supplier_id = node_mappings[invoice['supplier_gstin']]
        buyer_id = node_mappings[invoice['buyer_gstin']]
        
        # Add edge (directed: supplier → buyer)
        edge_list.append([supplier_id, buyer_id])
        
        # Edge attributes (amount, tax, date)
        edge_attrs.append([
            invoice['amount'],
            invoice['cgst'] + invoice['sgst'] + invoice['igst'],
            pd.to_datetime(invoice['date']).timestamp()
        ])
    
    # Convert to tensor
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    # STEP 5: Create labels (if available)
    # For now, use fraud column if exists, else None
    if 'fraud' in companies.columns:
        y = torch.zeros(num_nodes, dtype=torch.long)
        for gstin, node_id in node_mappings.items():
            company_data = companies[companies['gstin'] == gstin]
            if len(company_data) > 0:
                y[node_id] = int(company_data.iloc[0]['fraud'])
    else:
        y = None
    
    # STEP 6: Create PyTorch Geometric Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y
    )
    
    # STEP 7: Save
    torch.save(data, output_path + "/graph_data.pt")
    with open(output_path + "/node_mappings.pkl", 'wb') as f:
        pickle.dump(node_mappings, f)
    
    print(f"Graph created:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Node features: {x.shape[1]}")
    print(f"  Saved to {output_path}")
    
    return data, node_mappings
```

**Usage**:
```python
# Build graph from CSVs
data, mappings = build_graph_from_csv(
    "data/companies.csv",
    "data/invoices.csv",
    "data/processed/graphs"
)
```

---


## 5. PROCESSING WORKFLOWS

### Workflow 1: Initial System Startup

**Sequence**:

```
Step 1: User runs verify_system.py
   ├─ Check directory structure
   ├─ Check required files exist
   ├─ Check Python packages installed
   ├─ Check ports 8000 and 8080 available
   └─ Print diagnostic report

Step 2: User runs start_backend.bat (Windows) or start_backend.sh (Linux/Mac)
   ├─ Activate virtual environment (if exists)
   ├─ Install/update dependencies from requirements.txt
   ├─ Change directory to NETRA_TAX/backend
   ├─ Run: uvicorn main:app --host 0.0.0.0 --port 8000
   └─ Backend server starts listening

Step 3: Backend Initialization (main.py startup)
   ├─ Import dependencies (FastAPI, PyTorch, etc.)
   ├─ Initialize FastAPI app
   ├─ Add CORS middleware
   ├─ Define Pydantic models
   ├─ Call load_model_and_data() function
   │   ├─ Load tax-fraud-gnn/models/best_model.pt
   │   ├─ Load tax-fraud-gnn/data/processed/graphs/graph_data.pt
   │   ├─ Load tax-fraud-gnn/data/processed/graphs/node_mappings.pkl
   │   ├─ Load companies.csv into pandas DataFrame
   │   ├─ Load invoices.csv into pandas DataFrame
   │   ├─ Create NetworkX graph for pattern detection
   │   └─ Log "Model and data loaded successfully"
   ├─ Register all API routes
   └─ Server ready - Print "Uvicorn running on http://localhost:8000"

Step 4: User starts frontend (new terminal)
   ├─ Open new terminal/PowerShell
   ├─ cd NETRA_TAX/frontend
   ├─ Run: python -m http.server 8080
   └─ Frontend server starts on port 8080

Step 5: User opens browser
   ├─ Navigate to http://localhost:8080/index.html
   ├─ Browser loads HTML, CSS, JavaScript
   ├─ JavaScript makes API call to /api/fraud/summary
   ├─ Backend processes request and returns JSON
   ├─ Frontend renders dashboard with real data
   └─ System fully operational
```

**Time**: ~2-3 minutes total

---

### Workflow 2: Processing a Company Risk Request

**User Action**: Search for company "1234567890GST" in Company Explorer

**Backend Processing**:

```
1. User inputs GSTIN in company-explorer.html
   └─ JavaScript: api.getCompanyRisk("1234567890GST")

2. API Call: GET /api/fraud/company/risk?gstin=1234567890GST
   └─ HTTP request to http://localhost:8000

3. FastAPI receives request (main.py)
   └─ Route: @app.get("/api/fraud/company/risk")
   └─ Extract gstin parameter from query string

4. Look up node ID from GSTIN
   ├─ Check if GSTIN exists in node_mappings
   ├─ If exists: node_id = node_mappings["1234567890GST"]
   └─ If not: Return error "GSTIN not found"

5. Run GNN Inference
   ├─ Get node features: x = GRAPH_DATA.x[node_id]
   ├─ Get edge connections: edges = GRAPH_DATA.edge_index
   ├─ Run forward pass: out = MODEL(GRAPH_DATA.x, GRAPH_DATA.edge_index)
   ├─ Extract logits for this node: logits = out[node_id]
   ├─ Apply softmax: probs = torch.softmax(logits, dim=0)
   ├─ Get fraud probability: fraud_prob = probs[1].item()
   └─ GNN score = fraud_prob (e.g., 0.65)

6. Run Pattern Detection Algorithms
   
   6a. Circular Trading Detection
       ├─ Get neighbors of node in NetworkX graph
       ├─ Use nx.simple_cycles() to find all cycles
       ├─ Check if node_id is in any cycle
       └─ If yes: circular_trading_flag = True, weight = 0.15
   
   6b. High-Degree Node Detection
       ├─ Count in_degree = number of incoming edges
       ├─ Count out_degree = number of outgoing edges
       ├─ Calculate total_degree = in_degree + out_degree
       ├─ Calculate average_degree for all nodes
       ├─ If total_degree > 2 * average_degree:
       └─ high_degree_flag = True, weight = 0.10
   
   6c. Fraud Ring Detection
       ├─ Run community detection (Louvain algorithm)
       ├─ Get community_id for this node
       ├─ Calculate avg fraud score in this community
       ├─ If avg > 0.6:
       └─ fraud_ring_flag = True, weight = 0.10
   
   6d. Transaction Spike Detection
       ├─ Get all invoices for this GSTIN from last 90 days
       ├─ Group by week
       ├─ Calculate mean and std deviation
       ├─ Check if any week > mean + 2*std
       └─ If yes: spike_flag = True, weight = 0.05
   
   6e. Chain Depth Analysis
       ├─ Run BFS from node
       ├─ Measure max chain depth
       ├─ If depth > threshold (e.g., 5):
       └─ long_chain_flag = True, weight = 0.05
   
   6f. Clustering Coefficient
       ├─ Calculate local clustering coefficient
       ├─ If coefficient > threshold:
       └─ clustering_flag = True, weight = 0.05

7. Combine Scores
   ├─ base_score = gnn_score (0.65)
   ├─ pattern_boost = sum of all flag weights (e.g., 0.15 + 0.10 = 0.25)
   ├─ final_score = min(1.0, base_score + pattern_boost)
   └─ final_score = 0.90

8. Determine Risk Level
   ├─ if final_score >= 0.7: risk_level = "HIGH"
   ├─ elif final_score >= 0.4: risk_level = "MEDIUM"
   └─ else: risk_level = "LOW"

9. Generate Fraud Factors List
   ├─ fraud_factors = []
   ├─ If circular_trading_flag: append "Circular trading detected"
   ├─ If high_degree_flag: append "Unusually high number of connections"
   ├─ If fraud_ring_flag: append "Part of identified fraud ring"
   └─ etc.

10. Get Connected Entities
    ├─ Query NetworkX graph: neighbors = list(nx.neighbors(graph, node_id))
    └─ connected_entities = len(neighbors)

11. Generate Red Flags
    ├─ red_flags = []
    ├─ Check for: high ITC claims, weekend invoices, round amounts
    └─ Add to list

12. Format JSON Response
    {
        "gstin": "1234567890GST",
        "company_name": "ABC Corp",
        "fraud_score": 0.90,
        "risk_level": "HIGH",
        "confidence": 0.92,
        "fraud_factors": [
            "Circular trading detected",
            "Unusually high number of connections",
            "Part of identified fraud ring"
        ],
        "connected_entities": 23,
        "red_flags": [
            "High number of zero-rated invoices",
            "Sudden increase in ITC claims"
        ]
    }

13. Send Response
    └─ Return JSON with HTTP 200 OK

14. Frontend Receives Response
    ├─ api.js parses JSON
    ├─ company-explorer.html updates UI
    ├─ Display fraud score with color (RED for HIGH)
    ├─ Show fraud factors list
    ├─ Show connected entities count
    └─ Enable "View Network" button
```

**Total Time**: ~50-200ms

---

### Workflow 3: Generating Network Visualization

**User Action**: Click "View Network" for GSTIN "1234567890GST"

**Backend Processing**:

```
1. API Call: GET /api/graph/network?gstin=1234567890GST

2. Backend Processing (main.py)
   ├─ Get node_id from GSTIN
   ├─ Extract subgraph (node + neighbors)
   │   ├─ Get 1-hop neighbors
   │   ├─ Get 2-hop neighbors (optional, configurable)
   │   └─ Create node list: [node_id, neighbor1, neighbor2, ...]
   │
   ├─ Get edges for subgraph
   │   ├─ Filter GRAPH_DATA.edge_index
   │   └─ Only keep edges between nodes in subgraph
   │
   ├─ Get fraud scores for all nodes
   │   ├─ Run GNN inference for each node
   │   └─ scores = {node_id: fraud_score, ...}
   │
   ├─ Detect fraud rings (cycles)
   │   ├─ Find cycles in subgraph
   │   └─ Mark edges that are part of cycles
   │
   └─ Format for D3.js
       {
           "nodes": [
               {"id": "0", "gstin": "1234567890GST", "fraud_score": 0.90},
               {"id": "1", "gstin": "9876543210GST", "fraud_score": 0.45},
               ...
           ],
           "links": [
               {"source": "0", "target": "1", "amount": 100000, "in_cycle": false},
               {"source": "1", "target": "2", "amount": 50000, "in_cycle": true},
               ...
           ],
           "stats": {
               "total_nodes": 45,
               "total_edges": 123,
               "network_density": 0.12,
               "fraud_rings": 3,
               "anomaly_score": 0.68
           }
       }

3. Frontend Receives Data (graph-visualizer.html)
   ├─ Parse JSON
   ├─ Initialize D3.js force simulation
   ├─ Create SVG canvas
   ├─ Draw nodes with color by fraud_score
   │   ├─ RED (>= 0.7)
   │   ├─ ORANGE (>= 0.4)
   │   └─ GREEN (< 0.4)
   ├─ Draw edges
   │   ├─ Normal: gray
   │   └─ In cycle: dark red
   ├─ Enable interactions
   │   ├─ Drag nodes
   │   ├─ Zoom slider
   │   └─ Click for details
   └─ Display stats panel
```

**Total Time**: ~200-500ms (depends on subgraph size)

---

### Workflow 4: File Upload and Processing

**User Action**: Upload new invoice CSV file

**Backend Processing**:

```
1. User selects CSV file in upload.html
   └─ Drag-and-drop or file picker

2. JavaScript Validation
   ├─ Check file type (must be .csv)
   ├─ Check file size (< 100MB)
   └─ If valid, proceed

3. API Call: POST /api/files/upload
   ├─ Create FormData with file
   ├─ Send multipart/form-data request
   └─ Show progress bar

4. Backend Receives File (main.py)
   ├─ Validate file extension
   ├─ Generate unique filename with timestamp
   ├─ Save to uploads folder
   └─ Return upload_id

5. API Call: POST /api/files/process (with upload_id)

6. Backend Processing Pipeline
   
   6a. Load CSV
       ├─ Read CSV file with pandas
       ├─ Check encoding (UTF-8, latin1, etc.)
       └─ Load into DataFrame

   6b. Validate Columns
       ├─ Required columns: supplier_gstin, buyer_gstin, invoice_no,
       │                    date, amount, cgst, sgst, igst
       ├─ Check all required columns present
       └─ If missing, return error with column names

   6c. Validate Data Types
       ├─ GSTIN: string, 15 characters
       ├─ Amount: numeric, > 0
       ├─ Date: valid date format
       ├─ Tax: numeric, >= 0
       └─ Record errors by row number

   6d. Data Quality Checks
       ├─ Check for null values
       ├─ Check for duplicates
       ├─ Validate GSTIN format (regex)
       ├─ Validate date range (not future)
       ├─ Check amount ranges (realistic)
       └─ Generate quality score (0-100%)

   6e. Clean Data
       ├─ Remove null rows
       ├─ Remove duplicates
       ├─ Standardize formats
       ├─ Convert data types
       └─ Save cleaned CSV

   6f. Extract Companies
       ├─ Get unique GSTINs from supplier_gstin and buyer_gstin
       ├─ Create companies DataFrame
       ├─ Save companies.csv
       └─ Update master companies list

   6g. Build Graph (optional, for incremental learning)
       ├─ Load existing graph_data.pt
       ├─ Add new nodes for new GSTINs
       ├─ Add new edges from invoices
       ├─ Update node features
       ├─ Save updated graph_data.pt
       └─ Update node_mappings.pkl

   6h. Retrain Model (optional, for incremental learning)
       ├─ Load existing model
       ├─ Fine-tune on new data
       ├─ Validate performance
       └─ Save updated model if better

7. Generate Processing Report
   {
       "status": "success",
       "records_processed": 1500,
       "records_valid": 1450,
       "records_invalid": 50,
       "errors": [
           {"line": 15, "error": "Invalid GSTIN format"},
           {"line": 234, "error": "Amount must be positive"}
       ],
       "warnings": [
           "5 duplicate invoices removed",
           "2 future dates corrected"
       ],
       "quality_score": 96,
       "new_companies": 45,
       "new_invoices": 1450
   }

8. Frontend Displays Results
   ├─ Show success message
   ├─ Display quality score
   ├─ Show error list (if any)
   ├─ Show warnings (if any)
   └─ Enable "View Analysis" button
```

**Total Time**: 5-30 seconds (depends on file size)

---

### Workflow 5: Model Training (Offline)

**Manual Process** (Not exposed in UI, done by data scientist):

```
1. Prepare Data
   ├─ Run: python tax-fraud-gnn/prepare_real_data.py
   ├─ Clean companies.csv and invoices.csv
   └─ Output: processed CSVs in data/processed/

2. Build Graph
   ├─ Run: python tax-fraud-gnn/src/graph_construction/build_graph.py
   ├─ Create node features from company data
   ├─ Create edges from invoice data
   ├─ Save graph_data.pt and node_mappings.pkl
   └─ Print graph statistics

3. Train GNN Model
   ├─ Run: python tax-fraud-gnn/train_gnn_model.py
   ├─ Load graph_data.pt
   ├─ Initialize GNNFraudDetector model
   ├─ Define optimizer (Adam) and loss function (CrossEntropy)
   ├─ Training loop (100 epochs):
   │   ├─ Forward pass
   │   ├─ Calculate loss
   │   ├─ Backward pass
   │   ├─ Update weights
   │   └─ Validate every 10 epochs
   ├─ Save best model to models/best_model.pt
   └─ Print final metrics (accuracy, F1, precision, recall)

4. Evaluate Model
   ├─ Run: python tax-fraud-gnn/accuracy_model.py
   ├─ Load best_model.pt
   ├─ Test on holdout set
   ├─ Calculate metrics:
   │   ├─ Accuracy
   │   ├─ Precision
   │   ├─ Recall
   │   ├─ F1 Score
   │   ├─ ROC-AUC
   │   └─ Confusion Matrix
   └─ Print evaluation report

5. Deploy Model
   ├─ Copy best_model.pt to production models folder
   ├─ Restart backend (loads new model)
   └─ Verify with test API calls
```

**Total Time**: 1-4 hours (depends on data size and hardware)

---

