"""
Example: Using the Dashboard Reload Feature

This script demonstrates how to use the dashboard reload endpoint
after training a model with new data.
"""

import requests
import json
import time

# Configuration
BACKEND_URL = "http://localhost:8000"

def check_backend_status():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_current_stats():
    """Get current dashboard statistics"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/system/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error getting stats: {e}")
        return None

def reload_model():
    """Trigger model reload"""
    try:
        print("Triggering model reload...")
        response = requests.post(f"{BACKEND_URL}/api/model/reload", timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Model reloaded successfully!")
            print(f"   Status: {result['status']}")
            print(f"   Timestamp: {result['timestamp']}")
            print("\n   Statistics:")
            stats = result['statistics']
            print(f"   - Companies: {stats['companies']}")
            print(f"   - Invoices: {stats['invoices']}")
            print(f"   - Graph nodes: {stats['graph_nodes']}")
            print(f"   - Graph edges: {stats['graph_edges']}")
            print(f"   - Model loaded: {stats['model_loaded']}")
            print(f"   - Fraud scores computed: {stats['fraud_scores_computed']}")
            return True
        else:
            print(f"❌ Reload failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error during reload: {e}")
        return False

def main():
    print("=" * 80)
    print("Dashboard Reload Example")
    print("=" * 80)
    print()
    
    # Step 1: Check if backend is running
    print("Step 1: Checking backend status...")
    if not check_backend_status():
        print("❌ Backend is not running!")
        print(f"   Please start the backend at {BACKEND_URL}")
        print("   Command: uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        return
    print("✅ Backend is running\n")
    
    # Step 2: Get current statistics
    print("Step 2: Getting current statistics...")
    before_stats = get_current_stats()
    if before_stats:
        print(f"   Current companies: {before_stats.get('total_companies', 0)}")
        print(f"   Current invoices: {before_stats.get('total_invoices', 0)}")
        print(f"   High-risk entities: {before_stats.get('high_risk_entities', 0)}")
    print()
    
    # Step 3: Reload model (simulating after training)
    print("Step 3: Reloading model and data...")
    success = reload_model()
    print()
    
    if not success:
        print("❌ Reload failed!")
        return
    
    # Step 4: Get updated statistics
    print("Step 4: Verifying updated statistics...")
    time.sleep(1)  # Brief pause to ensure updates are processed
    after_stats = get_current_stats()
    if after_stats:
        print(f"   Updated companies: {after_stats.get('total_companies', 0)}")
        print(f"   Updated invoices: {after_stats.get('total_invoices', 0)}")
        print(f"   High-risk entities: {after_stats.get('high_risk_entities', 0)}")
    print()
    
    # Step 5: Summary
    print("=" * 80)
    print("✅ Dashboard Reload Complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Open dashboard in browser")
    print("2. Verify metrics have been updated")
    print("3. Check fraud summary shows new insights")
    print()

if __name__ == "__main__":
    main()
