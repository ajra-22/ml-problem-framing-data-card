from fastapi import status

HTTP_422 = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)


def test_missing_required_fields(client):
    # Missing plan_type and monthly_spend_inr
    payload = {
        "tenure_months": 5.0,
        "support_tickets": 1,
        "last_login_days": 10.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == HTTP_422
    data = response.json()
    assert "detail" in data
    # Ensure missing field errors are surfaced
    errors = [err["loc"][-1] for err in data["detail"]]
    assert "monthly_spend_inr" in errors
    assert "plan_type" in errors


def test_negative_numerical_values_rejected(client):
    payload = {
        "tenure_months": -4.0,  # invalid: ge=0
        "support_tickets": 2,
        "monthly_spend_inr": 499.0,
        "last_login_days": 10.0,
        "plan_type": "Basic",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == HTTP_422

    payload2 = {
        "tenure_months": 4.0,
        "support_tickets": -1,  # invalid: ge=0
        "monthly_spend_inr": 499.0,
        "last_login_days": 10.0,
        "plan_type": "Basic",
    }
    response2 = client.post("/predict", json=payload2)
    assert response2.status_code == HTTP_422

    payload3 = {
        "tenure_months": 4.0,
        "support_tickets": 1,
        "monthly_spend_inr": -100.0,  # invalid: ge=0
        "last_login_days": 10.0,
        "plan_type": "Basic",
    }
    response3 = client.post("/predict", json=payload3)
    assert response3.status_code == HTTP_422


def test_invalid_data_types(client):
    payload = {
        "tenure_months": "six_months",  # invalid string for float
        "support_tickets": 2,
        "monthly_spend_inr": 499.0,
        "last_login_days": 10.0,
        "plan_type": "Basic",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == HTTP_422


def test_empty_json_payload(client):
    response = client.post("/predict", json={})
    assert response.status_code == HTTP_422


def test_empty_batch_rejected(client):
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == HTTP_422
