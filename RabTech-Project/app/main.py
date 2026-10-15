import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.model_service import model_service
from app.routers import health, predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("churn_api")

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager: loads ML model at startup and handles cleanup on shutdown."""
    logger.info("Initializing ML inference service...")
    try:
        model_service.load_model()
        logger.info("Model service successfully initialized and ready for inference.")
    except Exception:
        logger.exception("Failed to load ML model on startup")
    yield
    logger.info("Shutting down ML inference service.")

app = FastAPI(
    title=settings.APP_NAME,
    description="""
## Customer Churn Prediction REST API
Production-grade ML inference service packaging a trained and tuned Champion Logistic Regression Pipeline.

### Key Capabilities
* **Real-Time Scoring (`POST /predict`)**: Single-record prediction with probability distribution and risk tiering.
* **Batch Scoring (`POST /predict/batch`)**: Low-latency vectorized prediction over multiple customer profiles.
* **Model Metadata (`GET /metadata`)**: Introspection of champion architecture, hyperparameters, and features.
* **Health & Probes (`GET /health`, `GET /ready`)**: Kubernetes-ready liveness and readiness probes.
    """,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Timing & Logging Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.6f}s"
    return response

# Global unhandled exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error processing %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred while processing the request.",
            "path": request.url.path,
        },
    )

# Include Routers (at root level and with api prefix)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(predict.router, prefix=settings.API_PREFIX)

@app.get(
    "/",
    tags=["Root"],
    summary="API Root Information",
    description="Returns service status, metadata links, and documentation endpoints.",
)
def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational" if model_service.is_ready else "initializing",
        "champion_model": model_service.metadata.get("champion", "Logistic Regression"),
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_spec": "/openapi.json",
        },
        "endpoints": {
            "predict_single": "/predict",
            "predict_batch": "/predict/batch",
            "model_metadata": "/metadata",
            "health_check": "/health",
            "readiness_check": "/ready",
        },
    }
