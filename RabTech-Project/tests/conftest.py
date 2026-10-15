import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model_service import model_service


@pytest.fixture(scope="session", autouse=True)
def setup_model():
    """Ensure the model is loaded for the test session."""
    model_service.load_model()

@pytest.fixture(scope="session")
def client():
    """Create a TestClient with the lifespan context executed."""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture
def high_risk_payload():
    return {
        "customer_id": "C_HIGH",
        "tenure_months": 1.0,
        "support_tickets": 5,
        "monthly_spend_inr": 499.0,
        "last_login_days": 30.0,
        "plan_type": "Basic",
    }

@pytest.fixture
def low_risk_payload():
    return {
        "customer_id": "C_LOW",
        "tenure_months": 30.0,
        "support_tickets": 0,
        "monthly_spend_inr": 1499.0,
        "last_login_days": 1.0,
        "plan_type": "Pro",
    }

@pytest.fixture
def batch_payload(high_risk_payload, low_risk_payload):
    return {
        "customers": [
            high_risk_payload,
            low_risk_payload,
            {
                "customer_id": "C_MID",
                "tenure_months": 11.0,
                "support_tickets": 2,
                "monthly_spend_inr": 999.0,
                "last_login_days": 5.0,
                "plan_type": "Standard",
            },
        ]
    }
