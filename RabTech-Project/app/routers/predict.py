import logging

from fastapi import APIRouter, HTTPException, status

from app.model_service import model_service
from app.schemas import (
    BatchCustomerInput,
    BatchPredictionResponse,
    CustomerInput,
    ModelMetadataResponse,
    PredictionResponse,
)

logger = logging.getLogger("churn_api.predict")

router = APIRouter(tags=["ML Inference"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Real-Time Single Customer Churn Prediction",
    description="Accepts customer features in a JSON payload and returns churn prediction probabilities, risk category, and binary decision.",
)
def predict_churn(payload: CustomerInput) -> PredictionResponse:
    if not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Champion model is not loaded. Service unavailable.",
        )
    try:
        return model_service.predict_single(payload)
    except Exception as exc:
        logger.exception("Inference error processing single prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc!s}",
        ) from exc


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch Customer Churn Prediction",
    description="Accepts multiple customer feature sets in a single JSON payload for vectorized, low-latency batch scoring.",
)
def predict_churn_batch(payload: BatchCustomerInput) -> BatchPredictionResponse:
    if not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Champion model is not loaded. Service unavailable.",
        )
    try:
        return model_service.predict_batch(payload.customers)
    except Exception as exc:
        logger.exception("Inference error processing batch prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {exc!s}",
        ) from exc


@router.get(
    "/metadata",
    response_model=ModelMetadataResponse,
    status_code=status.HTTP_200_OK,
    summary="Champion Model Metadata",
    description="Retrieves the champion model specifications, selected hyperparameters, cross-validation setup, and input feature requirements.",
)
def get_model_metadata() -> ModelMetadataResponse:
    return model_service.get_metadata()
