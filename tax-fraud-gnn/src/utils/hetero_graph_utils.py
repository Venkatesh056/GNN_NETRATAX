"""
Utility: Convert between graph formats for GNN training
Wraps new_datasets/graph_builder.py for use in tax-fraud-gnn pipeline
"""
from pathlib import Path
import sys

# Import the standalone graph builder
new_datasets_path = Path(__file__).parent.parent.parent / "new_datasets"
sys.path.insert(0, str(new_datasets_path))

try:
    from graph_builder import build_pyg_data, to_heterodata, build_networkx_graph, compute_company_features, read_inputs
    
    __all__ = [
        'build_pyg_data',
        'to_heterodata', 
        'build_networkx_graph',
        'compute_company_features',
        'read_inputs'
    ]
    
except ImportError as e:
    print(f"Warning: Could not import graph_builder from new_datasets: {e}")
    print("Make sure new_datasets/graph_builder.py exists")
    
    # Provide stubs
    def build_pyg_data(*args, **kwargs):
        raise NotImplementedError("graph_builder not available")
    
    __all__ = ['build_pyg_data']
