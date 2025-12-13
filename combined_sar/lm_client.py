"""LM server client"""

import json
import requests
from typing import Optional, Dict, Any


class LMClient:
    """Client for LM planning server"""
    
    def __init__(self, server_url="http://localhost:8000", timeout=10.0):
        self.server_url = server_url
        self.plan_endpoint = f"{server_url}/plan"
        self.timeout = timeout  # 10 second timeout to avoid blocking
    
    def get_plan(self, observations: str, history: str) -> Optional[Dict[str, Any]]:
        """Request plan from LM server"""
        try:
            payload = {
                "observations": observations,
                "history": history,
            }
            resp = requests.post(self.plan_endpoint, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[LMClient] Error: {e}")
            return None
    
    def health_check(self) -> bool:
        """Check if server is healthy"""
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5.0)
            return resp.status_code == 200
        except:
            return False
