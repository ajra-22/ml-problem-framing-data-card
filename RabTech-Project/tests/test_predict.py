from fastapi import status


def test_predict_high_risk_customer(client, high_risk_payload):
    response = client.post("/predict", json=high_risk_payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["customer_id"] == high_risk_payload["customer_id"]
    assert data["churn_prediction"] == 1
    assert data["churn_label"] == "Churn"
    assert data["risk_level"] == "High"
    assert 0.8 <= data["churn_probability"] <= 1.0

    # Probability schema tests
    assert "stay" in data["probabilities"]
    assert "churn" in data["probabilities"]
    assert round(data["probabilities"]["stay"] + data["probabilities"]["churn"], 2) == 1.0
    assert data["model_name"] == "Logistic Regression"
    assert data["threshold"] == 0.5
    assert "X-Process-Time" in response.headers

def test_predict_low_risk_customer(client, low_risk_payload):
    response = client.post("/predict", json=low_risk_payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["customer_id"] == low_risk_payload["customer_id"]
    assert data["churn_prediction"] == 0
    assert data["churn_label"] == "Stay"
    assert data["risk_level"] == "Low"
    assert 0.0 <= data["churn_probability"] <= 0.3

def test_predict_without_customer_id(client):
    payload = {
        "tenure_months": 15.0,
        "support_tickets": 1,
        "monthly_spend_inr": 1299.0,
        "last_login_days": 2.0,
        "plan_type": "Pro",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["customer_id"] is None
    assert data["churn_prediction"] == 0

def test_predict_unseen_plan_type_handled_gracefully(client):
    payload = {
        "customer_id": "C_UNKNOWN",
        "tenure_months": 10.0,
        "support_tickets": 2,
        "monthly_spend_inr": 850.0,
        "last_login_days": 7.0,
        "plan_type": "EnterpriseVIP",  # Unseen during training
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["churn_prediction"] in [0, 1]

def test_predict_batch_endpoint(client, batch_payload):
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["total_records"] == len(batch_payload["customers"])
    assert len(data["predictions"]) == len(batch_payload["customers"])

    # High risk first item
    assert data["predictions"][0]["churn_prediction"] == 1
    assert data["predictions"][0]["risk_level"] == "High"

    # Low risk second item
    assert data["predictions"][1]["churn_prediction"] == 0
    assert data["predictions"][1]["risk_level"] == "Low"

def test_api_prefix_route(client, high_risk_payload):
    response = client.post("/api/v1/predict", json=high_risk_payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["churn_prediction"] == 1

def test_get_model_metadata(client):
    response = client.get("/metadata")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["champion"] == "Logistic Regression"
    assert "tenure_months" in data["features"]
    assert "plan_type" in data["features"]
    assert data["target"] == "churned"
    assert "C" in data["best_params"]
