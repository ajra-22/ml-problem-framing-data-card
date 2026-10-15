import pytest

from app.model_service import ModelService
from app.schemas import CustomerInput


def test_model_service_load_and_predict():
    service = ModelService()
    service.load_model()
    assert service.is_ready is True
    assert service.model is not None

    cust = CustomerInput(
        customer_id="C_UNIT",
        tenure_months=2.0,
        support_tickets=4,
        monthly_spend_inr=499.0,
        last_login_days=21.0,
        plan_type="Basic",
    )
    result = service.predict_single(cust)
    assert result.churn_prediction in [0, 1]
    assert 0.0 <= result.churn_probability <= 1.0
    assert result.risk_level in ["Low", "Medium", "High"]

def test_model_service_risk_levels():
    service = ModelService()
    assert service._determine_risk_level(0.1) == "Low"
    assert service._determine_risk_level(0.34) == "Low"
    assert service._determine_risk_level(0.35) == "Medium"
    assert service._determine_risk_level(0.50) == "Medium"
    assert service._determine_risk_level(0.64) == "Medium"
    assert service._determine_risk_level(0.65) == "High"
    assert service._determine_risk_level(0.99) == "High"

def test_model_service_batch():
    service = ModelService()
    service.load_model()
    batch = [
        CustomerInput(
            customer_id="C1",
            tenure_months=1.0,
            support_tickets=5,
            monthly_spend_inr=499.0,
            last_login_days=30.0,
            plan_type="Basic",
        ),
        CustomerInput(
            customer_id="C2",
            tenure_months=24.0,
            support_tickets=0,
            monthly_spend_inr=1499.0,
            last_login_days=1.0,
            plan_type="Pro",
        ),
    ]
    batch_resp = service.predict_batch(batch)
    assert batch_resp.total_records == 2
    assert len(batch_resp.predictions) == 2
    assert batch_resp.predictions[0].customer_id == "C1"
    assert batch_resp.predictions[1].customer_id == "C2"

def test_model_service_unloaded_raises():
    unloaded_service = ModelService()
    cust = CustomerInput(
        tenure_months=5.0,
        support_tickets=1,
        monthly_spend_inr=799.0,
        last_login_days=3.0,
        plan_type="Standard",
    )
    with pytest.raises(RuntimeError):
        unloaded_service.predict_single(cust)
