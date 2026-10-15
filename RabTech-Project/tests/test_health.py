from fastapi import status


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "service" in data
    assert data["status"] == "operational"
    assert "documentation" in data
    assert "endpoints" in data

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data
    assert data["uptime_seconds"] >= 0

def test_ready_endpoint(client):
    response = client.get("/ready")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
