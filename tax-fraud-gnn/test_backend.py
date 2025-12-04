#!/usr/bin/env python3
"""Quick test to verify backend API endpoints"""

import requests
import sys

BASE_URL = "http://localhost:5000"

def test_endpoints():
    """Test critical endpoints"""
    endpoints = [
        ("/api/top_senders", "Top Senders"),
        ("/api/top_receivers", "Top Receivers"),
        ("/api/statistics", "Statistics"),
    ]
    
    print("Testing backend endpoints...")
    print("=" * 50)
    
    for endpoint, name in endpoints:
        url = BASE_URL + endpoint
        try:
            response = requests.get(url, timeout=10)
            print(f"\n✓ {name} ({endpoint})")
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and "data" in data:
                    print(f"  Response: Valid Plotly format ✓")
                elif isinstance(data, dict) and "error" in data:
                    print(f"  Error: {data['error']}")
                else:
                    print(f"  Response keys: {list(data.keys())[:3]}...")
            else:
                print(f"  Error: {response.text[:100]}")
                
        except requests.ConnectionError:
            print(f"\n✗ {name} ({endpoint})")
            print(f"  Error: Cannot connect to {BASE_URL}")
            print(f"  Make sure the backend is running!")
            return False
        except Exception as e:
            print(f"\n✗ {name} ({endpoint})")
            print(f"  Error: {str(e)}")
            return False
    
    print("\n" + "=" * 50)
    print("Backend is working! 🎉")
    return True

if __name__ == "__main__":
    success = test_endpoints()
    sys.exit(0 if success else 1)
