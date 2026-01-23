"""
GNN Model Training Module
Implements Graph Neural Network for tax fraud detection
Enhanced with GAT, residual connections, and advanced training techniques
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GraphSAGE, GATConv, SAGEConv, BatchNorm
from torch_geometric.utils import add_self_loops, degree
import numpy as np
from pathlib import Path
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GNNFraudDetector(nn.Module):
    """
    Enhanced Graph Neural Network for fraud detection.
    Uses Graph Attention Networks (GAT) with multi-head attention,
    residual connections, layer normalization, and MLP classifier.
    """
    
    def __init__(self, in_channels, hidden_channels, out_channels, model_type="gat", 
                 num_heads=8, dropout=0.3, num_layers=4):
        super(GNNFraudDetector, self).__init__()
        
        self.model_type = model_type
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_bn = nn.LayerNorm(hidden_channels)
        
        if model_type == "gat":
            # Multi-layer GAT with residual connections
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            
            for i in range(num_layers):
                if i == 0:
                    self.convs.append(GATConv(hidden_channels, hidden_channels // num_heads, 
                                              heads=num_heads, dropout=dropout, concat=True))
                else:
                    self.convs.append(GATConv(hidden_channels, hidden_channels // num_heads, 
                                              heads=num_heads, dropout=dropout, concat=True))
                self.norms.append(nn.LayerNorm(hidden_channels))
            
        elif model_type == "gcn":
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for i in range(num_layers):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.norms.append(nn.LayerNorm(hidden_channels))
                
        elif model_type == "sage":
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for i in range(num_layers):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                self.norms.append(nn.LayerNorm(hidden_channels))
        else:
            # Default to GCN
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            for i in range(num_layers):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
                self.norms.append(nn.LayerNorm(hidden_channels))
        
        # MLP classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # Skip connection projections (for dimension matching if needed)
        self.skip_proj = nn.Linear(hidden_channels, hidden_channels)
    
    def forward(self, x, edge_index):
        """Forward pass with residual connections"""
        # Input projection
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph convolution layers with residual connections
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            identity = x  # Save for residual
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + identity  # Residual connection
        
        # Classification head
        x = self.classifier(x)
        
        return x


class GNNTrainer:
    """Trainer for GNN fraud detection model"""
    
    def __init__(self, data_path="../../data/processed", models_path="../../models", model_type="gcn"):
        self.data_path = Path(data_path)
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model_type = model_type
        self.data = None
        self.model = None
        self.train_idx = None
        self.val_idx = None
        self.test_idx = None
        
    def load_graph_data(self):
        """Load PyTorch Geometric graph data"""
        try:
            self.data = torch.load(self.data_path / "graphs" / "graph_data.pt")
            self.data = self.data.to(self.device)
            logger.info(f"Loaded graph data: {self.data}")
            return self.data
        except FileNotFoundError as e:
            logger.error(f"Graph data not found: {e}")
            raise
    
    def create_train_val_test_split(self, train_ratio=0.6, val_ratio=0.2):
        """
        Split nodes into train/val/test sets
        Uses stratified split to maintain fraud class balance
        """
        num_nodes = self.data.num_nodes
        y = self.data.y
        
        # Get indices of each class
        fraud_idx = np.where(y.cpu().numpy() == 1)[0]
        normal_idx = np.where(y.cpu().numpy() == 0)[0]
        
        logger.info(f"Total nodes: {num_nodes}")
        logger.info(f"Fraud nodes: {len(fraud_idx)} ({100*len(fraud_idx)/num_nodes:.2f}%)")
        logger.info(f"Normal nodes: {len(normal_idx)} ({100*len(normal_idx)/num_nodes:.2f}%)")
        
        # Split each class
        fraud_train = int(len(fraud_idx) * train_ratio)
        fraud_val = int(len(fraud_idx) * val_ratio)
        
        normal_train = int(len(normal_idx) * train_ratio)
        normal_val = int(len(normal_idx) * val_ratio)
        
        train_idx = np.concatenate([fraud_idx[:fraud_train], normal_idx[:normal_train]])
        val_idx = np.concatenate([
            fraud_idx[fraud_train:fraud_train+fraud_val],
            normal_idx[normal_train:normal_train+normal_val]
        ])
        test_idx = np.concatenate([
            fraud_idx[fraud_train+fraud_val:],
            normal_idx[normal_train+normal_val:]
        ])
        
        self.train_idx = torch.tensor(train_idx, dtype=torch.long, device=self.device)
        self.val_idx = torch.tensor(val_idx, dtype=torch.long, device=self.device)
        self.test_idx = torch.tensor(test_idx, dtype=torch.long, device=self.device)
        
        logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    def build_model(self, in_channels, hidden_channels=64, out_channels=2):
        """Create GNN model"""
        self.model = GNNFraudDetector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            model_type=self.model_type
        ).to(self.device)
        
        logger.info(f"Model created: {self.model}")
    
    def train_epoch(self, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        
        optimizer.zero_grad()
        out = self.model(self.data.x, self.data.edge_index)
        loss = criterion(out[self.train_idx], self.data.y[self.train_idx])
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def validate(self, criterion):
        """Validate model"""
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        loss = criterion(out[self.val_idx], self.data.y[self.val_idx])
        
        pred = out[self.val_idx].argmax(dim=1)
        accuracy = accuracy_score(self.data.y[self.val_idx].cpu(), pred.cpu())
        
        return loss.item(), accuracy
    
    @torch.no_grad()
    def test(self):
        """Evaluate on test set"""
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        pred = out[self.test_idx].argmax(dim=1)
        pred_proba = torch.softmax(out[self.test_idx], dim=1)[:, 1]
        
        y_true = self.data.y[self.test_idx].cpu().numpy()
        y_pred = pred.cpu().numpy()
        y_proba = pred_proba.cpu().numpy()
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
        }
        
        return metrics, y_true, y_pred, y_proba
    
    def train_model(self, epochs=100, lr=0.001, weight_decay=5e-4):
        """Train the model"""
        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING")
        logger.info("=" * 60)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Compute class weights to handle class imbalance (inverse frequency)
        try:
            y_all = self.data.y.cpu().numpy()
            unique, counts = np.unique(y_all, return_counts=True)
            weights = np.ones((2,), dtype=np.float32)
            if len(unique) == 2:
                for cls, cnt in zip(unique, counts):
                    # inverse frequency
                    weights[int(cls)] = 1.0 / float(cnt)
                # normalize weights to sum to number of classes
                weights = weights / weights.sum() * len(weights)
            weights_tensor = torch.tensor(weights, dtype=torch.float, device=self.device)
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)
            logger.info(f"Using class weights for loss: {weights}")
        except Exception as e:
            logger.warning(f"Could not compute class weights automatically: {e}")
            criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(optimizer, criterion)
            val_loss, val_acc = self.validate(criterion)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.models_path / "best_model.pt")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load(self.models_path / "best_model.pt"))
        logger.info("Loaded best model")
    
    def save_model(self):
        """Save model and metadata"""
        model_path = self.models_path / "fraud_detector_model.pt"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        metadata = {
            "model_type": self.model_type,
            "in_channels": self.data.x.shape[1],
            "device": str(self.device)
        }
        with open(self.models_path / "model_metadata.json", "w") as f:
            json.dump(metadata, f)
    
    def run_pipeline(self, epochs=100, lr=0.001):
        """Execute complete training pipeline"""
        try:
            # Load data
            logger.info("Loading graph data...")
            self.load_graph_data()
            
            # Create train/val/test split
            logger.info("Creating train/val/test split...")
            self.create_train_val_test_split()
            
            # Build model
            logger.info("Building model...")
            in_channels = self.data.x.shape[1]
            self.build_model(in_channels=in_channels)
            
            # Train
            logger.info("Training model...")
            self.train_model(epochs=epochs, lr=lr)
            
            # Test
            logger.info("Testing model...")
            metrics, y_true, y_pred, y_proba = self.test()
            
            logger.info("=" * 60)
            logger.info("TEST RESULTS")
            logger.info("=" * 60)
            for metric, value in metrics.items():
                logger.info(f"{metric.upper()}: {value:.4f}")
            logger.info("=" * 60)
            
            # Save results
            results = {
                "metrics": metrics,
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
            }
            with open(self.models_path / "results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            # Save model
            self.save_model()
            
            logger.info("✅ TRAINING COMPLETE")
            
            return metrics, y_true, y_pred
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise


if __name__ == "__main__":
    trainer = GNNTrainer(model_type="gcn")
    metrics, y_true, y_pred = trainer.run_pipeline(epochs=100, lr=0.001)
    print(f"\nFinal Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
