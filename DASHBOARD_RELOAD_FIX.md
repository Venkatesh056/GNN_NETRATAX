# Dashboard Insights Fix - Incremental Learning Support

## Problem

Previously, dashboard insights were not updating after incremental learning (uploading new data and retraining the model). The backend would load model and data only during startup, and subsequent training would not refresh the in-memory data.

## Solution

Added a model reload endpoint that allows the backend to refresh all data and model weights without restarting the server.

### Changes Made

1. **Refactored Data Loading** (`/NETRA_TAX/backend/main.py`)
   - Created reusable `load_model_and_data()` function
   - This function loads/reloads:
     - Company and invoice data from CSV files
     - Graph data (nodes, edges)
     - GNN model weights
     - Node mappings
     - Fraud scores (recomputed from new model)

2. **Added Reload Endpoint** (`/NETRA_TAX/backend/main.py`)
   - New endpoint: `POST /api/model/reload`
   - Triggers reload of all data and model
   - Returns statistics about loaded data
   - Can be called programmatically or manually

3. **Automatic Reload Trigger** (`/tax-fraud-gnn/train_from_uploads.py`)
   - Training script now automatically calls reload endpoint
   - If backend is running, dashboard updates automatically
   - Gracefully handles when backend is not running

## Usage

### Configuration

The training script supports configuring the backend URL via environment variable:

```bash
# Default (localhost)
python train_from_uploads.py

# Custom backend URL
export BACKEND_URL=http://192.168.1.100:8000
python train_from_uploads.py

# Or inline
BACKEND_URL=http://production-server:8000 python train_from_uploads.py
```

### Automatic (Recommended)

After incremental learning, the training script will automatically reload the backend:

```bash
cd /home/runner/work/GNN_NETRATAX/GNN_NETRATAX/tax-fraud-gnn
python train_from_uploads.py
```

The script will:
1. Train the model with new data
2. Save updated model and graph
3. Automatically call `/api/model/reload` endpoint
4. Dashboard insights update immediately

### Manual Reload

If automatic reload fails or you need to manually trigger a reload:

#### Using curl:
```bash
curl -X POST http://localhost:8000/api/model/reload
```

#### Using Python:
```python
import requests

response = requests.post('http://localhost:8000/api/model/reload')
result = response.json()

print(f"Status: {result['status']}")
print(f"Companies: {result['statistics']['companies']}")
print(f"Fraud scores: {result['statistics']['fraud_scores_computed']}")
```

#### Using the browser/Postman:
- Method: `POST`
- URL: `http://localhost:8000/api/model/reload`
- No request body needed

### Response Format

Success response:
```json
{
  "status": "success",
  "message": "Model and data reloaded successfully",
  "timestamp": "2024-01-15T10:30:00",
  "statistics": {
    "companies": 1000,
    "invoices": 5000,
    "graph_nodes": 1000,
    "graph_edges": 8500,
    "model_loaded": true,
    "fraud_scores_computed": 1000
  }
}
```

Error response (500):
```json
{
  "detail": "Failed to reload model: [error details]"
}
```

## Workflow

### Complete Incremental Learning Workflow:

1. **Upload new data**
   - Place `companies.csv` and `invoices.csv` in upload folder
   
2. **Train model**
   ```bash
   python train_from_uploads.py
   ```

3. **Backend automatically reloads** (or manually call `/api/model/reload`)

4. **Dashboard shows updated insights** - no server restart needed!

### What Gets Updated:

- ✅ Company data (COMPANIES_DF)
- ✅ Invoice data (INVOICES_DF)
- ✅ Graph structure (GRAPH_DATA)
- ✅ Model weights (MODEL)
- ✅ Node mappings (NODE_MAPPINGS)
- ✅ **Fraud scores (FRAUD_SCORES)** - Recomputed from new model!

## Benefits

1. **No Server Restart Required**: Dashboard updates without downtime
2. **Real-time Updates**: Insights refresh immediately after training
3. **Automatic**: Training script handles reload automatically
4. **Safe**: Old data remains until new data loads successfully
5. **Stateful**: All global variables update atomically

## Error Handling

The reload endpoint handles errors gracefully:

- If files are missing, returns specific error
- If model loading fails, returns error without crashing
- Logs detailed errors for debugging
- Returns HTTP 500 with error details

## Testing

To verify the fix works:

1. Start the backend:
   ```bash
   cd /home/runner/work/GNN_NETRATAX/GNN_NETRATAX/NETRA_TAX/backend
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. Check initial statistics:
   ```bash
   curl http://localhost:8000/api/system/stats
   ```

3. Train with new data:
   ```bash
   cd /home/runner/work/GNN_NETRATAX/GNN_NETRATAX/tax-fraud-gnn
   python train_from_uploads.py
   ```

4. Verify statistics updated:
   ```bash
   curl http://localhost:8000/api/system/stats
   ```

5. Check dashboard shows new insights:
   - Open browser to dashboard
   - Verify metrics match new statistics

## Files Modified

1. `/NETRA_TAX/backend/main.py`
   - Added `load_model_and_data()` function
   - Added `/api/model/reload` endpoint
   - Refactored `startup_event()` to use new function

2. `/tax-fraud-gnn/train_from_uploads.py`
   - Added automatic reload trigger after training
   - Graceful error handling for offline backend

## Future Enhancements

Potential improvements:
- WebSocket notifications when reload completes
- Background task for reload (non-blocking)
- Reload queue for multiple concurrent requests
- Version tracking for model updates
- Rollback capability if reload fails

## Troubleshooting

### Dashboard not updating?

1. Check if backend is running:
   ```bash
   curl http://localhost:8000/api/health
   ```

2. Manually trigger reload:
   ```bash
   curl -X POST http://localhost:8000/api/model/reload
   ```

3. Check logs for errors:
   ```bash
   tail -f /path/to/backend/logs/app.log
   ```

### Reload endpoint returns error?

1. Verify model files exist:
   - `/tax-fraud-gnn/models/best_model.pt`
   - `/tax-fraud-gnn/data/processed/companies_processed.csv`
   - `/tax-fraud-gnn/data/processed/invoices_processed.csv`
   - `/tax-fraud-gnn/data/processed/graphs/graph_data.pt`

2. Check file permissions

3. Review logs for specific error

## Summary

This fix enables true incremental learning by allowing the dashboard to update insights without server restart. The solution is minimal, focused, and addresses the core issue while maintaining backward compatibility.
