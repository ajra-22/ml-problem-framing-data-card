import time

from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.model_service import model_service
from app.schemas import HealthResponse

router = APIRouter(tags=["Health & Diagnostics"])

START_TIME = time.time()

@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service Health Check",
    description="Returns liveness status, model load state, application version, and uptime.",
)
def health_check() -> HealthResponse:
    uptime = round(time.time() - START_TIME, 2)
    return HealthResponse(
        status="healthy" if model_service.is_ready else "initializing",
        model_loaded=model_service.is_ready,
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        uptime_seconds=uptime,
    )

@router.get(
    "/ready",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness Probe",
    description="Checks whether the microservice and champion ML model are fully ready to accept inference requests.",
)
def readiness_check() -> HealthResponse:
    if not model_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet loaded into memory or service is unavailable.",
        )
    uptime = round(time.time() - START_TIME, 2)
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        uptime_seconds=uptime,
    )
