"""Integration Tests for ShiftGuard-SecLM FastAPI Service.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force UTF-8 on Windows consoles if needed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from fastapi.testclient import TestClient
from src.serving.api import app, load_model_on_startup


def test_api_endpoints():
    load_model_on_startup()
    client = TestClient(app)

    # 1. Health check
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    print("[PASS] API /health passed")

    # 2. Analyze endpoint
    req_body = {
        "task": "threat_analysis",
        "requirement": "Build a secure user authentication endpoint.",
        "prompt": "Create a FastAPI route /login verifying password hash.",
        "language": "python",
        "framework": "fastapi",
        "code": "def login(user, p): return db.find(user)",
    }
    res2 = client.post("/analyze", json=req_body)
    assert res2.status_code == 200
    data = res2.json()
    assert "threats" in data
    assert "risk" in data
    assert "explanation" in data
    print("[PASS] API /analyze passed")


if __name__ == "__main__":
    test_api_endpoints()
    print("\nAll FastAPI Serving Tests PASSED successfully!")
