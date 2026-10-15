import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "Customer Churn Prediction API")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH", str(BASE_DIR / "artifacts" / "champion_model.joblib")
    )
    METADATA_PATH: str = os.getenv(
        "METADATA_PATH", str(BASE_DIR / "artifacts" / "champion_model_metadata.json")
    )
    CHURN_THRESHOLD: float = float(os.getenv("CHURN_THRESHOLD", "0.5"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

settings = Settings()
