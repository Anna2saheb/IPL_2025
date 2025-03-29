from fastapi.testclient import TestClient
from app import app  # Import your FastAPI app

client = TestClient(app)

def test_read_root():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_predict_endpoint():
    """Test the predict endpoint with sample data"""
    test_data = {
        "batsman": "Virat Kohli",
        "bowler": "Jasprit Bumrah"
    }
    response = client.post("/predict", data=test_data)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Virat Kohli" in response.text
    assert "Jasprit Bumrah" in response.text