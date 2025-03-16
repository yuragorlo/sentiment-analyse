from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Text Analysis API!"}

def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint():
    """Test the predict endpoint with different text inputs."""
    response = client.post(
        "/api/predict",
        json={"text": "I love this product! It's amazing."}
    )
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data
    assert data["sentiment"] == "positive"
    assert 0 <= data["confidence"] <= 1

    response = client.post(
        "/api/predict",
        json={"text": ""}
    )
    assert response.status_code == 422

    response = client.post(
        "/api/predict",
        json={"text": "a" * 6000}
    )
    assert response.status_code == 422
