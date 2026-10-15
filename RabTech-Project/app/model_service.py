import json
import logging
import warnings
from pathlib import Path

import joblib
import pandas as pd

from app.config import settings
from app.schemas import (
    BatchPredictionResponse,
    CustomerInput,
    ModelMetadataResponse,
    PredictionResponse,
)

logger = logging.getLogger("uvicorn.error")

FEATURE_COLUMNS = [
    "tenure_months",
    "support_tickets",
    "monthly_spend_inr",
    "last_login_days",
    "plan_type",
]

class ModelService:
    def __init__(self, model_path: str | None = None, metadata_path: str | None = None):
        self.model_path = Path(model_path or settings.MODEL_PATH)
        self.metadata_path = Path(metadata_path or settings.METADATA_PATH)
        self.model = None
        self.metadata: dict = {}
        self.classes: list[int] = [0, 1]
        self.churn_class_idx: int = 1
        self.is_ready: bool = False

    def load_model(self) -> None:
        """Loads serialized scikit-learn pipeline and training metadata."""
        # Check fallback paths if configured path does not exist
        if not self.model_path.exists():
            fallbacks = [
                Path("artifacts/champion_model.joblib"),
                Path("R5.joblib"),
                Path("R3.joblib"),
            ]
            for fb in fallbacks:
                if fb.exists():
                    self.model_path = fb
                    break

        if not self.metadata_path.exists():
            fallbacks_meta = [
                Path("artifacts/champion_model_metadata.json"),
                Path("R7.json"),
            ]
            for fbm in fallbacks_meta:
                if fbm.exists():
                    self.metadata_path = fbm
                    break

        logger.info(f"Loading champion model from {self.model_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = joblib.load(self.model_path)

        if hasattr(self.model, "classes_"):
            self.classes = list(self.model.classes_)
            # Locate index for positive class (1 = churn)
            if 1 in self.classes:
                self.churn_class_idx = self.classes.index(1)
            else:
                self.churn_class_idx = len(self.classes) - 1

        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "champion": "Logistic Regression",
                "best_params": {"C": 1, "penalty": "l2"},
                "features": FEATURE_COLUMNS,
                "target": "churned",
            }

        self.is_ready = True
        logger.info(
            f"Champion model successfully loaded: {self.metadata.get('champion', 'Logistic Regression')}"
        )

    def _determine_risk_level(self, churn_prob: float) -> str:
        if churn_prob < 0.35:
            return "Low"
        elif churn_prob < 0.65:
            return "Medium"
        return "High"

    def predict_single(self, input_data: CustomerInput) -> PredictionResponse:
        """Runs real-time inference on a single customer record."""
        if not self.is_ready or self.model is None:
            raise RuntimeError("Model is not loaded or ready for inference.")

        row = {
            "tenure_months": input_data.tenure_months,
            "support_tickets": input_data.support_tickets,
            "monthly_spend_inr": input_data.monthly_spend_inr,
            "last_login_days": input_data.last_login_days,
            "plan_type": str(input_data.plan_type),
        }
        df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        proba = self.model.predict_proba(df)[0]
        churn_prob = float(proba[self.churn_class_idx])
        stay_prob = float(1.0 - churn_prob)

        threshold = settings.CHURN_THRESHOLD
        churn_pred = 1 if churn_prob >= threshold else 0
        churn_label = "Churn" if churn_pred == 1 else "Stay"
        risk_level = self._determine_risk_level(churn_prob)

        return PredictionResponse(
            customer_id=input_data.customer_id,
            churn_prediction=churn_pred,
            churn_label=churn_label,
            churn_probability=round(churn_prob, 4),
            probabilities={
                "stay": round(stay_prob, 4),
                "churn": round(churn_prob, 4),
            },
            risk_level=risk_level,
            model_name=self.metadata.get("champion", "Logistic Regression"),
            threshold=threshold,
        )

    def predict_batch(self, customers: list[CustomerInput]) -> BatchPredictionResponse:
        """Runs high-performance vectorised inference over a batch of customer records."""
        if not self.is_ready or self.model is None:
            raise RuntimeError("Model is not loaded or ready for inference.")

        rows = [
            {
                "tenure_months": c.tenure_months,
                "support_tickets": c.support_tickets,
                "monthly_spend_inr": c.monthly_spend_inr,
                "last_login_days": c.last_login_days,
                "plan_type": str(c.plan_type),
            }
            for c in customers
        ]
        df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)

        probas = self.model.predict_proba(df)
        threshold = settings.CHURN_THRESHOLD
        champion_name = self.metadata.get("champion", "Logistic Regression")

        predictions: list[PredictionResponse] = []
        for i, c in enumerate(customers):
            churn_prob = float(probas[i][self.churn_class_idx])
            stay_prob = float(1.0 - churn_prob)
            churn_pred = 1 if churn_prob >= threshold else 0
            churn_label = "Churn" if churn_pred == 1 else "Stay"
            risk_level = self._determine_risk_level(churn_prob)

            predictions.append(
                PredictionResponse(
                    customer_id=c.customer_id,
                    churn_prediction=churn_pred,
                    churn_label=churn_label,
                    churn_probability=round(churn_prob, 4),
                    probabilities={
                        "stay": round(stay_prob, 4),
                        "churn": round(churn_prob, 4),
                    },
                    risk_level=risk_level,
                    model_name=champion_name,
                    threshold=threshold,
                )
            )

        return BatchPredictionResponse(
            total_records=len(predictions),
            predictions=predictions,
        )

    def get_metadata(self) -> ModelMetadataResponse:
        """Returns champion model specification and training metadata."""
        return ModelMetadataResponse(
            champion=self.metadata.get("champion", "Logistic Regression"),
            best_params=self.metadata.get("best_params", {}),
            features=self.metadata.get("features", FEATURE_COLUMNS),
            target=self.metadata.get("target", "churned"),
            selection_rule=self.metadata.get("selection_rule"),
            cv=self.metadata.get("cv"),
            n_train=self.metadata.get("n_train"),
            n_test=self.metadata.get("n_test"),
        )

# Global singleton instance
model_service = ModelService()
