# Tax Fraud Detection GNN - Implementation Plan & Timeline
**Status**: MVP Production Ready | **Duration**: 8 Sprints (8 weeks) | **Team Size**: 3-5 engineers

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Sprint Roadmap](#sprint-roadmap)
6. [Infrastructure & Costs](#infrastructure--costs)
7. [Success Criteria](#success-criteria)

---

## Executive Summary

### Problem Statement
Tax fraud detection requires analyzing complex relationships between companies, invoices, and transactions. Traditional rule-based systems miss sophisticated fraud patterns that manifest as network anomalies.

### Solution
A Graph Neural Network (GNN) system that:
- Models companies as nodes and transactions as edges
- Learns fraud patterns from historical data
- Scores fraud risk in real-time
- Provides explainable results via interactive visualizations

### Key Metrics (MVP Target)
- **Detection Accuracy**: 85-90% precision
- **Latency**: <500ms per company scoring
- **Scalability**: Support 10K+ companies, 100K+ transactions
- **Deployment**: Docker containers, cloud-agnostic
- **Cost**: <$500/month for MVP (hackathon phase)

---

## System Architecture

### 1. High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION                          │
│  CSV/JSON Upload → Validation → Normalization → Data Store     │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                    GRAPH CONSTRUCTION                           │
│  Companies (Nodes) ←→ Invoices/Transactions (Edges)            │
│  Node Features: Turnover, ITC, GST, Risk Indicators            │
│  Edge Features: Amount, Date, Payment Status                    │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                    GNN MODEL INFERENCE                          │
│  Graph Convolution (3 layers) → Node Embeddings → Binary Class  │
│  Output: Fraud Probability [0-1] per company                    │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                    RESULT AGGREGATION                           │
│  Fraud Score + Risk Level + Suspicious Links + Pattern          │
│  Store in Cache/DB → API Ready                                  │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                  PRESENTATION & ALERTS                          │
│  ┌─────────────┬──────────────┬─────────────┐                  │
│  │ Web UI      │ API Endpoints│ PDF Reports │                  │
│  │ (React)     │ (REST/GraphQL)│(PyPDF2)    │                  │
│  │ Interactive │ Real-time    │ Scheduled   │                  │
│  │ Graph Viz   │ Scoring      │ Batch       │                  │
│  └─────────────┴──────────────┴─────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2. System Components

```
                    ┌─────────────────────┐
                    │   Frontend (React)  │
                    │  - Dashboard        │
                    │  - Interactive Graph│
                    │  - Reports          │
                    └──────────┬──────────┘
                              │ HTTP/REST
        ┌─────────────────────┴──────────────────────┐
        │                                             │
    ┌───▼──────────────────────┐      ┌─────────────▼────────┐
    │  Flask/FastAPI Backend   │      │  WebSocket Server    │
    │  - REST API Endpoints    │      │  - Real-time Updates │
    │  - Data Processing       │      │  - Live Alerts       │
    │  - GNN Inference         │      │  - Event Stream      │
    └───┬──────────────────────┘      └─────────────┬────────┘
        │                                             │
        │              ┌────────────────┐            │
        │              │  Cache Layer   │            │
        │              │  (Redis)       │            │
        │              └────────────────┘            │
        │                      ▲                      │
        │                      │                      │
    ┌───▼──────────────────────┴──────────────────────┴────────┐
    │                    Data Layer                            │
    │  ┌──────────────┬──────────────┬──────────────┐           │
    │  │ PostgreSQL   │  Graph Store │ File Storage │           │
    │  │ (Metadata)   │  (Neo4j/     │ (S3/Local)   │           │
    │  │              │   PyTorch)   │              │           │
    │  └──────────────┴──────────────┴──────────────┘           │
    └───────────────────────────────────────────────────────────┘
        │
    ┌───▼─────────────────────┐
    │   ML Training Pipeline  │
    │  - PyTorch Geometric    │
    │  - GNN Model Training   │
    │  - Hyperparameter Tune  │
    │  - Model Versioning     │
    └─────────────────────────┘
```

### 3. Database Schema (ERD)

```
Companies
├── company_id (PK)
├── name
├── gst_number
├── pan
├── turnover
├── industry
├── location
├── risk_score (0-1)
├── fraud_probability
├── risk_level (HIGH/MEDIUM/LOW)
└── created_at

Invoices
├── invoice_id (PK)
├── seller_id (FK → Companies)
├── buyer_id (FK → Companies)
├── amount
├── date
├── itc_claimed
├── payment_status
└── is_flagged

Transactions
├── txn_id (PK)
├── from_company_id (FK)
├── to_company_id (FK)
├── invoice_id (FK)
├── amount
├── timestamp
└── payment_method

ModelPredictions
├── prediction_id (PK)
├── company_id (FK)
├── fraud_score
├── confidence
├── reasoning (JSON)
├── created_at
└── model_version

AlertLogs
├── alert_id (PK)
├── company_id (FK)
├── alert_type (PATTERN/SCORE/NETWORK)
├── severity (CRITICAL/HIGH/MEDIUM)
├── payload (JSON)
└── timestamp
```

---

## Technology Stack

### Backend Architecture

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **API Framework** | FastAPI + Flask | Fast async (FastAPI), lightweight (Flask), ASGI/WSGI support |
| **ML Framework** | PyTorch + PyTorch Geometric | State-of-art GNNs, production-ready, good community |
| **Graph DB** | Neo4j Community (optional) | Graph queries, community free tier, or embedded PyTorch |
| **SQL DB** | PostgreSQL | ACID compliance, JSON support, TimescaleDB extension for alerts |
| **Cache** | Redis | Sub-ms latency, Lua scripting for complex ops |
| **Job Queue** | Celery + RabbitMQ | Async task processing, batch inference, report generation |
| **Monitoring** | Prometheus + Grafana | Open-source, time-series metrics |

### Frontend Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Framework** | React 18 + TypeScript | Type safety, component reusability, large ecosystem |
| **Graph Viz** | Cytoscape.js / Vis.js | Interactive network visualization, HTML5 canvas |
| **Charts** | Plotly.js / Chart.js | Rich analytics, exportable visualizations |
| **UI Framework** | Material-UI / Tailwind | Responsive, accessible, theming support |
| **State Mgmt** | Redux Toolkit / Zustand | Predictable state, devtools, lightweight |
| **API Client** | Axios + React Query | Request management, caching, retry logic |

### ML/Data Pipeline

| Stage | Tools |
|-------|-------|
| **Data Ingestion** | Pandas, Polars, DuckDB |
| **Preprocessing** | Scikit-learn, Feature-engine |
| **Graph Construction** | NetworkX, PyTorch Geometric |
| **Model Training** | PyTorch Lightning, Optuna |
| **Model Serving** | TorchServe, ONNX Runtime, vLLM |
| **Monitoring** | Evidently AI, WhyLabs |

### DevOps & Deployment

| Component | Technology |
|-----------|-----------|
| **Containerization** | Docker, Docker Compose |
| **Orchestration** | Kubernetes (optional), Docker Swarm |
| **CI/CD** | GitHub Actions, GitLab CI |
| **Cloud Platforms** | AWS (EC2, S3, RDS), GCP, Azure |
| **Infrastructure as Code** | Terraform, CloudFormation |
| **Logging** | ELK Stack (Elasticsearch, Logstash, Kibana) |

### Recommended Tech Stack Summary

```yaml
Backend:
  Framework: FastAPI (async) + Flask (traditional)
  Language: Python 3.9+
  ML: PyTorch 2.0+ with torch-geometric
  Database: PostgreSQL + Redis
  Queue: Celery with RabbitMQ
  Deployment: Docker + Kubernetes

Frontend:
  Framework: React 18 + TypeScript
  UI: Tailwind CSS + Shadcn/UI
  Visualization: Cytoscape.js for graphs
  State: Zustand
  Build: Vite

DevOps:
  Containers: Docker
  IaC: Terraform
  CI/CD: GitHub Actions
  Monitoring: Prometheus + Grafana

Data Pipeline:
  Processing: Pandas/Polars
  Graph Ops: PyTorch Geometric
  Training: PyTorch Lightning
  Serving: ONNX Runtime
```

---

## Project Structure

### Recommended Folder Layout

```
tax-fraud-detection-gnn/
│
├── README.md
├── SETUP.md
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── .github/
│   └── workflows/
│       ├── tests.yml
│       ├── build.yml
│       └── deploy.yml
│
├── backend/
│   ├── app.py                    # FastAPI main app
│   ├── config.py                 # Configuration management
│   ├── requirements.txt
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── companies.py      # Company endpoints
│   │   │   ├── fraud_detection.py # Scoring endpoints
│   │   │   ├── reports.py        # Report generation
│   │   │   └── alerts.py         # Alert management
│   │   ├── schemas/              # Pydantic models
│   │   │   ├── company.py
│   │   │   ├── invoice.py
│   │   │   └── prediction.py
│   │   └── middleware/
│   │       ├── auth.py
│   │       ├── logging.py
│   │       └── error_handlers.py
│   │
│   ├── ml/
│   │   ├── model/
│   │   │   ├── gnn_models.py     # GNN architecture
│   │   │   ├── training.py       # Training pipeline
│   │   │   └── inference.py      # Inference engine
│   │   │
│   │   ├── data/
│   │   │   ├── loaders.py        # Data loading
│   │   │   ├── preprocessing.py  # Data cleaning
│   │   │   ├── graph_builder.py  # Graph construction
│   │   │   └── feature_eng.py    # Feature engineering
│   │   │
│   │   ├── utils/
│   │   │   ├── metrics.py        # Evaluation metrics
│   │   │   ├── visualization.py  # Graph viz
│   │   │   └── explainability.py # SHAP/LIME
│   │   │
│   │   └── checkpoints/          # Saved models
│   │       └── best_model.pt
│   │
│   ├── services/
│   │   ├── fraud_detector.py     # Main service
│   │   ├── report_generator.py   # PDF reports
│   │   ├── alert_manager.py      # Alert logic
│   │   └── data_validator.py     # Input validation
│   │
│   ├── database/
│   │   ├── models.py             # SQLAlchemy models
│   │   ├── schemas.py            # DB schemas
│   │   ├── migrations/           # Alembic migrations
│   │   └── connection.py         # DB setup
│   │
│   └── tests/
│       ├── test_api.py
│       ├── test_models.py
│       ├── test_inference.py
│       └── fixtures/
│
├── frontend/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── index.html
│   │
│   ├── public/
│   │   └── assets/
│   │
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   │
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx      # Main dashboard
│   │   │   ├── Companies.tsx      # Company list
│   │   │   ├── FraudAnalysis.tsx  # Detailed analysis
│   │   │   ├── Reports.tsx        # Report viewer
│   │   │   └── Settings.tsx       # Configuration
│   │   │
│   │   ├── components/
│   │   │   ├── Header.tsx
│   │   │   ├── Sidebar.tsx
│   │   │   ├── GraphVisualization.tsx
│   │   │   ├── FraudScoreCard.tsx
│   │   │   ├── AlertPanel.tsx
│   │   │   └── DataUploader.tsx
│   │   │
│   │   ├── services/
│   │   │   ├── api.ts            # API client
│   │   │   ├── auth.ts           # Auth service
│   │   │   └── websocket.ts      # WebSocket client
│   │   │
│   │   ├── store/
│   │   │   ├── companyStore.ts
│   │   │   ├── alertStore.ts
│   │   │   └── uiStore.ts
│   │   │
│   │   ├── hooks/
│   │   │   ├── useFraudScore.ts
│   │   │   ├── useWebSocket.ts
│   │   │   └── useGraphData.ts
│   │   │
│   │   ├── styles/
│   │   │   ├── globals.css
│   │   │   ├── variables.css
│   │   │   └── components.css
│   │   │
│   │   └── types/
│   │       ├── company.ts
│   │       ├── fraud.ts
│   │       └── api.ts
│   │
│   └── tests/
│       └── __tests__/
│
├── data/
│   ├── raw/
│   │   ├── companies.csv         # Sample data
│   │   └── invoices.csv
│   │
│   ├── processed/
│   │   ├── graphs/
│   │   │   └── graph_data.pt
│   │   ├── features/
│   │   └── mappings.pkl
│   │
│   └── uploads/                  # User uploads
│       └── 20240101/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_gnn_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_inference_testing.ipynb
│
├── scripts/
│   ├── setup_db.py               # Database setup
│   ├── train_model.py            # Training script
│   ├── generate_sample_data.py   # Test data generation
│   ├── import_csv.py             # CSV import tool
│   └── run_inference.py          # Batch inference
│
├── config/
│   ├── settings.py               # Configuration
│   ├── logging.yaml              # Logging config
│   └── model.yaml                # Model config
│
├── docs/
│   ├── API.md                    # API documentation
│   ├── ARCHITECTURE.md           # System design
│   ├── DEPLOYMENT.md             # Deployment guide
│   ├── MODEL.md                  # Model documentation
│   └── USAGE.md                  # User guide
│
└── monitoring/
    ├── prometheus.yml            # Prometheus config
    ├── grafana/
    │   └── dashboards/
    └── alerting/
        └── rules.yml
```

### Key Files Description

| File/Directory | Purpose |
|---|---|
| `backend/ml/model/gnn_models.py` | GNN architecture (GCN, GAT, GraphSAGE) |
| `backend/ml/data/graph_builder.py` | Converts CSV to PyTorch Geometric tensors |
| `backend/api/routes/fraud_detection.py` | Main scoring API endpoints |
| `backend/services/fraud_detector.py` | Orchestrates inference pipeline |
| `frontend/src/components/GraphVisualization.tsx` | Interactive network visualization |
| `scripts/train_model.py` | End-to-end training pipeline |
| `notebooks/03_gnn_training.ipynb` | Jupyter notebook for model development |

---

## Sprint Roadmap

### Sprint-by-Sprint Timeline (8 weeks, 1 week per sprint)

#### **Sprint 1: Foundation & Data Pipeline (Week 1)**

**Goals**:
- Set up project scaffolding and CI/CD
- Implement data ingestion and validation
- Create PostgreSQL schema

**Tasks**:
1. Git repo setup, branch strategy (main, develop, feature/*)
2. Docker setup with PostgreSQL + Redis
3. Data loader for CSV (companies, invoices)
4. Pandas preprocessing pipeline
5. Database migration scripts (Alembic)
6. Unit tests for data loaders (pytest)

**Deliverables**:
- ✅ Project structure live
- ✅ Docker Compose file working
- ✅ Data validation passing 100 test cases
- ✅ 5 sample CSV files loaded to DB

**Acceptance Criteria**:
```
- All CSV files parse without errors
- Data validation tests pass (100% coverage)
- Database initialized and seeded
- CI/CD pipeline triggers on push
```

**Tech**: Docker, PostgreSQL, Pandas, SQLAlchemy, Alembic

---

#### **Sprint 2: Graph Construction & GNN Model Development (Week 2)**

**Goals**:
- Build graph from relational data
- Implement GNN model architecture
- Set up training pipeline

**Tasks**:
1. Graph builder: Companies (nodes) → Invoices/Transactions (edges)
2. Node features: Turnover, ITC, invoice count, location
3. Edge features: Amount, payment_status, date
4. PyTorch Geometric dataset class
5. GNN models: GCN (3-layer), attention variant
6. Training loop with validation
7. Model checkpointing

**Deliverables**:
- ✅ Graph with 1000+ nodes, 5000+ edges
- ✅ Trained GNN model (initial)
- ✅ Training notebook
- ✅ Model saved as PyTorch checkpoint

**Acceptance Criteria**:
```
- Graph loads in <2s for 10K companies
- Model trains in <10 min on 5K samples
- Validation accuracy >70% on initial dataset
- Loss curve shows convergence
```

**Tech**: PyTorch, PyTorch Geometric, NetworkX, Lightning

---

#### **Sprint 3: Backend API & Inference Engine (Week 3)**

**Goals**:
- Build REST API for fraud scoring
- Implement real-time inference
- Add authentication & rate limiting

**Tasks**:
1. FastAPI app structure
2. Endpoints:
   - `POST /api/predict` - single company scoring
   - `GET /api/companies/{id}` - company details
   - `POST /api/batch_predict` - batch scoring
3. Model serving (ONNX or TorchServe)
4. Redis caching for predictions
5. JWT authentication
6. API documentation (OpenAPI/Swagger)
7. Rate limiting & throttling

**Deliverables**:
- ✅ FastAPI app running locally
- ✅ 5 working endpoints
- ✅ API docs accessible at `/docs`
- ✅ <500ms response time per prediction

**Acceptance Criteria**:
```
- All endpoints return 200 for valid input
- Invalid input returns 400/422
- Cached predictions hit in <10ms
- API documentation complete
- Rate limit enforced (100 req/min)
```

**Tech**: FastAPI, ONNX Runtime, Pydantic, Redis, JWT

---

#### **Sprint 4: Frontend - Dashboard & Graph Visualization (Week 4)**

**Goals**:
- Build React dashboard
- Interactive graph visualization
- Real-time score display

**Tasks**:
1. React project setup (Vite + TypeScript)
2. Components:
   - Dashboard (KPIs, top companies)
   - Company list with search/filter
   - Interactive graph (Cytoscape.js)
   - Fraud score card (risk level coloring)
3. State management (Zustand)
4. API integration (Axios)
5. Responsive design (Tailwind CSS)
6. Dark mode support

**Deliverables**:
- ✅ React app runs on localhost:3000
- ✅ Dashboard displays 10+ KPIs
- ✅ Graph renders 1000+ nodes interactively
- ✅ Mobile-responsive layout

**Acceptance Criteria**:
```
- Dashboard loads in <3s
- Graph interaction smooth (60 FPS)
- All filters work (location, risk level, industry)
- Mobile view usable on tablets
- No console errors
```

**Tech**: React 18, TypeScript, Tailwind, Cytoscape.js, Zustand

---

#### **Sprint 5: Alert System & Report Generation (Week 5)**

**Goals**:
- Real-time alert system
- PDF report generation
- WebSocket for live updates

**Tasks**:
1. Alert types: HIGH_FRAUD, PATTERN_DETECTED, NETWORK_ANOMALY
2. Alert storage in PostgreSQL
3. WebSocket server for live alerts
4. Report generator (PyPDF2/ReportLab):
   - Company summary
   - Fraud score explanation
   - Suspicious transaction table
   - Graph visualization
5. Email alerts (SMTP)
6. Alert UI panel in frontend

**Deliverables**:
- ✅ Alert system triggering correctly
- ✅ PDF reports generated in <2s
- ✅ WebSocket connection established
- ✅ Email alerts sent successfully

**Acceptance Criteria**:
```
- Alerts trigger within 1s of score update
- PDF file quality acceptable
- Email delivery >95% success
- WebSocket reconnects on failure
- Alert history persisted for 30 days
```

**Tech**: WebSockets, PyPDF2, Celery, SMTP, SQL

---

#### **Sprint 6: Data Upload & Incremental Learning (Week 6)**

**Goals**:
- File upload interface
- Model retraining with new data
- Incremental learning capability

**Tasks**:
1. File upload UI (drag-drop)
2. CSV validation & preview
3. Data merge with existing data
4. Trigger incremental retraining
5. Model versioning (v1, v2, v3...)
6. Fallback to previous model if accuracy drops
7. Progress tracking (Celery task status)

**Deliverables**:
- ✅ Upload interface working
- ✅ 100 new rows processed in <30s
- ✅ Model retrained & deployed
- ✅ Version tracking working

**Acceptance Criteria**:
```
- File size limit enforced (100MB)
- CSV format validation passes
- New data merges without duplicates
- Retraining accuracy ≥ baseline
- Rollback available if model worse
- Upload history tracked
```

**Tech**: FastAPI file upload, background tasks, model versioning

---

#### **Sprint 7: Monitoring, Observability & Deployment (Week 7)**

**Goals**:
- Production monitoring setup
- Containerization & cloud deployment
- Performance optimization

**Tasks**:
1. Prometheus metrics:
   - Inference latency
   - Model accuracy drift
   - API request count
2. Grafana dashboards (3-5 dashboards)
3. ELK stack logging
4. Docker multi-stage build
5. Kubernetes manifests (optional)
6. AWS/GCP deployment scripts
7. Database query optimization
8. Model inference caching

**Deliverables**:
- ✅ Metrics flowing to Prometheus
- ✅ Grafana dashboards visible
- ✅ Docker image <500MB
- ✅ Deployed to cloud (staging)
- ✅ Latency <500ms p99

**Acceptance Criteria**:
```
- Metrics dashboard complete
- Logs centralized in ELK
- Docker image builds in <5min
- Application health check passing
- Database backup automated daily
- Cost estimation <$500/month
```

**Tech**: Prometheus, Grafana, Docker, Kubernetes, ELK, Terraform

---

#### **Sprint 8: Testing, Documentation & Go-Live (Week 8)**

**Goals**:
- Full test coverage
- Production documentation
- Go-live preparation

**Tasks**:
1. Unit tests (Backend: 80%+, Frontend: 70%+)
2. Integration tests (API + DB)
3. Load testing (100 req/s)
4. Security audit (OWASP Top 10)
5. User documentation (PDF guide)
6. API documentation completeness
7. Incident runbook
8. Training slides for users
9. Go-live checklist
10. Post-launch monitoring plan

**Deliverables**:
- ✅ Test coverage report
- ✅ Load test results
- ✅ Security audit report
- ✅ User manual (5-page PDF)
- ✅ Production runbook
- ✅ System live and stable

**Acceptance Criteria**:
```
- Unit test coverage >80%
- Load test: 100 req/s @ <1s response
- Zero CRITICAL security findings
- Documentation reviewed by stakeholders
- 99.5% uptime in staging
- Incident response time <5min
```

**Tech**: pytest, Jest, k6, OWASP, GitHub Pages

---

### Sprint Summary Table

| Sprint | Focus | Input | Output | Team |
|--------|-------|-------|--------|------|
| 1 | Foundation | Requirements | Docker, DB, CI/CD | 2 backend |
| 2 | ML Pipeline | Data | GNN model, checkpoints | 2 ML |
| 3 | Backend API | Model | REST API, docs | 2 backend |
| 4 | Frontend | API spec | React dashboard | 2 frontend |
| 5 | Alerts/Reports | DB schema | Alert system, PDFs | 1 full-stack |
| 6 | Data Pipeline | CSV samples | Upload, retraining | 1 ML, 1 backend |
| 7 | DevOps | App code | Monitoring, deployment | 1 DevOps |
| 8 | Testing | All code | Tests, docs, go-live | 3 engineers |

---

## Infrastructure & Costs

### Minimum Hardware Requirements (MVP/Hackathon)

#### Local Development
```
CPU: 8+ cores (4 for training, 4 for inference)
RAM: 16 GB (8GB Python, 8GB Redis/DB)
GPU: Optional (NVIDIA RTX 3060 12GB for faster training)
Storage: 50 GB SSD (data + models + logs)
Network: 10 Mbps
```

#### Cloud Deployment (AWS Example)

**Recommended Configuration**:
```
Compute:
  - 1x EC2 t3.xlarge (4 vCPU, 16 GB RAM) for API server
  - 1x EC2 g4dn.xlarge (GPU, optional) for batch training
  - 1x RDS PostgreSQL db.t3.small (2 vCPU, 2 GB RAM)
  
Storage:
  - 20 GB EBS for application
  - 100 GB S3 for models & uploads
  - 10 GB RDS for data

Networking:
  - 1x ALB (Application Load Balancer)
  - 1x NAT Gateway
  - VPC with subnets

Cache:
  - ElastiCache Redis (cache.t3.micro)

Monitoring:
  - CloudWatch (included)
  - Optional: Prometheus + Grafana on EC2
```

### Cost Breakdown (Monthly)

#### AWS Estimate (MVP)

| Component | Size | Cost/Month | Notes |
|-----------|------|-----------|-------|
| **Compute** | | | |
| EC2 (API) | t3.xlarge | $120 | 730 hrs/mo |
| EC2 (Training) | g4dn.xlarge | $500 | Used 20 hrs/mo |
| RDS PostgreSQL | db.t3.small | $40 | Multi-AZ available |
| **Storage** | | | |
| EBS Volume | 20 GB | $2 | |
| S3 | 100 GB | $2.30 | Standard storage |
| S3 Transfer | 10 GB/mo | $1 | Out transfer |
| **Networking** | | | |
| ALB | 1 month | $16 | +$0.005 per LCU |
| NAT Gateway | 1 month | $32 | |
| Data Transfer | 100 GB | $9 | Internal + out |
| **Cache** | | | |
| ElastiCache Redis | micro | $15 | 1 GB memory |
| **Monitoring** | | | |
| CloudWatch | Included | - | Within free tier |
| **Backup** | | | |
| RDS Snapshots | 20 GB | $5 | Daily snapshots |
| **Total** | | **~$740/month** | |

**Cost Optimization Tips**:
- Use **Spot Instances** for training (70% discount): $150 instead of $500
- Use **Reserved Instances** if committing 1 year: 30-40% discount
- **Serverless Option** (AWS Lambda): Pay per invocation (~$0.0000002 per request)
- Use **Local Redis** in EC2 instead of ElastiCache: Save $15
- Consolidate to single **t3.2xlarge** instance for MVP: $200

**Optimized MVP Cost: $300-400/month** ✅

#### GCP Alternative
```
Compute Engine: n1-standard-2 (2 vCPU, 7.5 GB) → $50/month
Cloud SQL PostgreSQL: db-f1-micro → $35/month
Memorystore Redis: 1 GB → $10/month
Cloud Storage: 100 GB → $2/month
Load Balancer: ~$18/month
Total: ~$115/month (70% cheaper!)
```

#### Free/Open-Source Stack (Cost: ~$0)
```
Compute: Your laptop + 1 VPS ($5-10/mo DigitalOcean)
Database: PostgreSQL (self-hosted)
Cache: Redis (self-hosted)
Monitoring: Prometheus + Grafana (self-hosted)
Total: ~$5-10/month
```

### Scaling Considerations (Beyond MVP)

| Metric | MVP | Scale 10x | Scale 100x |
|--------|-----|----------|-----------|
| Companies | 10K | 100K | 1M |
| Invoices | 100K | 1M | 10M |
| Compute | t3.xlarge | 3x xlarge | Auto-scaling group |
| Database | db.t3.small | db.m5.xlarge | Multi-region RDS |
| Cache | 1 GB Redis | 10 GB Redis | Redis Cluster |
| Inference | 1s batch | 100ms streaming | Sub-10ms |
| Cost | $300/mo | $1.5K/mo | $10K/mo |

---

## Success Criteria

### MVP Definition (End of Sprint 8)

#### Functional Requirements Met
- ✅ Data ingestion (CSV/JSON) for companies, invoices, transactions
- ✅ Graph construction from relational data
- ✅ GNN-based fraud scoring (binary classification)
- ✅ REST API for scoring and query
- ✅ Interactive React UI with graph visualization
- ✅ Alert system for high-risk companies
- ✅ PDF report generation
- ✅ Model retraining with new data
- ✅ Real-time WebSocket updates

#### Performance Targets
| Metric | Target | Measurement |
|--------|--------|------------|
| **Inference Latency** | <500ms per company | p99 response time |
| **Throughput** | 100 req/s | API load test |
| **Graph Load Time** | <2s for 10K nodes | UI responsiveness |
| **Uptime** | 99.5% | Monitoring 7 days |
| **Model Accuracy** | 85%+ precision | Test set evaluation |
| **Report Generation** | <2s per PDF | Timing measurement |

#### Quality Metrics
| Aspect | Criteria |
|--------|----------|
| **Test Coverage** | Backend >80%, Frontend >70% |
| **Code Quality** | SonarQube grade A/B, no CRITICAL issues |
| **Documentation** | API docs, user guide, architecture diagrams |
| **Security** | OWASP Top 10 audit passed, HTTPS only |
| **Scalability** | Horizontal scaling tested up to 3 instances |
| **User Experience** | <3s dashboard load, smooth interactions |

#### Deployment Checklist
```
□ Docker images built and tested
□ CI/CD pipeline working (GitHub Actions)
□ Database migrations tested
□ Backup strategy defined
□ Monitoring dashboards live
□ Alert escalation configured
□ Incident runbook created
□ Team trained on operations
□ Cost monitoring in place
□ Legal/compliance review done
```

### Post-MVP Enhancements (Future Roadmap)

#### Phase 2 (Months 3-4)
- [ ] Multi-model ensemble (GCN + GAT + GraphSAGE)
- [ ] Explainability (SHAP values, attention visualization)
- [ ] Real-time transaction monitoring
- [ ] Mobile app (React Native)
- [ ] Bank API integration

#### Phase 3 (Months 5-6)
- [ ] Multi-jurisdiction support (India, Singapore, US)
- [ ] Advanced analytics (time-series patterns)
- [ ] Automated policy enforcement
- [ ] Integration with GST portal / Tax authority APIs
- [ ] Federated learning for privacy

#### Phase 4 (Months 7+)
- [ ] Large Language Model (LLM) for report generation
- [ ] Predictive alerting (forecast fraud 30 days ahead)
- [ ] Blockchain audit trail
- [ ] API monetization (SaaS model)

---

## Implementation Checklist

### Pre-Launch (Week 0)
- [ ] Stakeholder kickoff meeting
- [ ] Environment setup (dev, staging, prod)
- [ ] Data access and permissions
- [ ] Security review and approval
- [ ] Team skill assessment and training

### During Development
- [ ] Weekly sprint reviews with stakeholders
- [ ] Bi-weekly code reviews
- [ ] Monthly security patches
- [ ] Continuous monitoring of tech debt

### Launch (Sprint 8)
- [ ] Load testing (200% of expected load)
- [ ] Security penetration testing
- [ ] Backup and recovery drill
- [ ] Incident response simulation
- [ ] User acceptance testing (UAT)
- [ ] Go/No-Go decision

### Post-Launch (Week 1-4)
- [ ] Daily monitoring for first week
- [ ] User feedback collection
- [ ] Performance optimization
- [ ] Bug fixes and patches
- [ ] Documentation updates

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Data quality issues | High | High | Sprint 1 validation, data profiling |
| Model accuracy low | Medium | High | Ensemble models, hyperparameter tuning |
| Scalability bottleneck | Medium | Medium | Load testing, caching, DB optimization |
| Security breach | Low | Critical | Penetration testing, WAF, secrets rotation |
| Team skill gap | Medium | Medium | Training, external consultant, pair programming |
| Timeline slip | Medium | High | Aggressive testing, clear scope definition |

---

## Conclusion

This plan delivers a **production-ready Tax Fraud Detection system** in 8 weeks with:

✅ **Complete Data Pipeline**: CSV → Graph → Model → API → UI  
✅ **State-of-Art ML**: GNN architecture with >85% accuracy  
✅ **Scalable Backend**: FastAPI + PostgreSQL + Redis  
✅ **Beautiful Frontend**: React dashboard with interactive graph  
✅ **Production Ready**: Monitoring, alerts, deployment automation  
✅ **Cost Efficient**: $300-400/month MVP (hackathon-friendly)  
✅ **Well Documented**: Architecture, API, user guides included  

**Next Steps**:
1. Assemble team (3-5 engineers)
2. Provision cloud resources (AWS/GCP)
3. Schedule Sprint 1 kickoff
4. Execute 8-week sprint cycle
5. Launch and iterate based on user feedback

---

## Appendix: Reference Links

### Technologies
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- FastAPI: https://fastapi.tiangolo.com/
- React: https://react.dev/
- PostgreSQL: https://www.postgresql.org/
- Docker: https://docs.docker.com/

### Datasets & Benchmarks
- OG-Bench (Open Graph Benchmark): https://ogb.stanford.edu/
- AMLCC (Anti-Money Laundering): Kaggle dataset
- Fraud Detection Datasets: https://www.kaggle.com/

### Best Practices
- GNN Survey: https://arxiv.org/abs/2003.03330
- Fraud Detection: https://arxiv.org/abs/1909.06889
- MLOps: https://ml-ops.systems/

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Author**: Senior Engineering Lead  
**Status**: APPROVED FOR IMPLEMENTATION
