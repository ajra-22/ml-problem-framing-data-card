
from pydantic import BaseModel, ConfigDict, Field


class CustomerInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "C015",
                "tenure_months": 8,
                "support_tickets": 3,
                "monthly_spend_inr": 799.0,
                "last_login_days": 14,
                "plan_type": "Basic",
            }
        }
    )

    customer_id: str | None = Field(
        default=None,
        description="Optional customer identifier for tracking and correlation",
        examples=["C015"],
    )
    tenure_months: float = Field(
        ...,
        ge=0,
        description="Customer relationship length in months",
        examples=[8.0],
    )
    support_tickets: int = Field(
        ...,
        ge=0,
        description="Total support tickets raised by customer",
        examples=[3],
    )
    monthly_spend_inr: float = Field(
        ...,
        ge=0,
        description="Monthly subscription spend in Indian Rupees (INR)",
        examples=[799.0],
    )
    last_login_days: float = Field(
        ...,
        ge=0,
        description="Days elapsed since the customer last logged in",
        examples=[14.0],
    )
    plan_type: str = Field(
        ...,
        min_length=1,
        description="Subscription tier (e.g., 'Basic', 'Standard', 'Pro')",
        examples=["Basic"],
    )

class PredictionResponse(BaseModel):
    customer_id: str | None = Field(
        default=None, description="Customer identifier if provided in request"
    )
    churn_prediction: int = Field(
        ...,
        description="Binary churn prediction: 1 for Churned, 0 for Retained/Stay",
        examples=[1],
    )
    churn_label: str = Field(
        ...,
        description="Human-readable prediction label ('Churn' or 'Stay')",
        examples=["Churn"],
    )
    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model estimated probability of customer churning (0.0 to 1.0)",
        examples=[0.8924],
    )
    probabilities: dict[str, float] = Field(
        ...,
        description="Probability distribution across both classes ('stay' and 'churn')",
        examples=[{"stay": 0.1076, "churn": 0.8924}],
    )
    risk_level: str = Field(
        ...,
        description="Categorical risk classification: 'Low' (<0.35), 'Medium' (0.35-0.65), 'High' (>=0.65)",
        examples=["High"],
    )
    model_name: str = Field(
        ...,
        description="Name of the champion model serving the prediction",
        examples=["Logistic Regression"],
    )
    threshold: float = Field(
        ...,
        description="Decision threshold applied to probabilities",
        examples=[0.5],
    )

class BatchCustomerInput(BaseModel):
    customers: list[CustomerInput] = Field(
        ...,
        min_length=1,
        description="Batch list of customer payloads to evaluate",
    )

class BatchPredictionResponse(BaseModel):
    total_records: int = Field(..., description="Number of customer records evaluated")
    predictions: list[PredictionResponse] = Field(
        ..., description="List of predictions corresponding to input records"
    )

class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., examples=[True])
    app_name: str = Field(..., examples=["Customer Churn Prediction API"])
    version: str = Field(..., examples=["1.0.0"])
    uptime_seconds: float = Field(..., examples=[120.5])

class ModelMetadataResponse(BaseModel):
    champion: str
    best_params: dict[str, object]
    features: list[str]
    target: str
    selection_rule: str | None = None
    cv: str | None = None
    n_train: int | None = None
    n_test: int | None = None
