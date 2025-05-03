import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define model path and threshold
MODEL_PATH = os.environ.get("MODEL_PATH", "catboost_model.cbm")
PREDICTION_THRESHOLD = float(os.environ.get("PREDICTION_THRESHOLD", "0.487"))

# Feature lists
CATEGORICAL_FEATURES = [
    "occupation",
    "marital_status",
    "education",
    "housing_loan",
    "personal_loan",
    "contact_mode",
    "month",
    "week_day",
    "previous_outcome",
    "age_group",
    "education_marital",
    "month_weekday",
    "season",
    "loan_burden_score",
]

# Create validation lists
VALID_OCCUPATIONS = [
    "housemaid",
    "services",
    "admin.",
    "blue-collar",
    "technician",
    "retired",
    "management",
    "unemployed",
    "self-employed",
    "unknown",
    "entrepreneur",
    "student",
]

VALID_MARITAL_STATUSES = ["married", "single", "divorced", "unknown"]

VALID_EDUCATION_LEVELS = [
    "basic.4y",
    "high.school",
    "basic.6y",
    "basic.9y",
    "professional.course",
    "unknown",
    "university.degree",
    "illiterate",
]

VALID_CONTACT_MODES = ["cellular", "telephone"]

VALID_PREVIOUS_OUTCOMES = ["nonexistent", "failure", "success"]

VALID_MONTHS = [
    "jan",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]

VALID_DAYS = ["mon", "tue", "wed", "thu", "fri"]

# Constants for feature engineering
MONTH_TO_SEASON = {
    "mar": "spring",
    "apr": "spring",
    "may": "spring",
    "jun": "summer",
    "jul": "summer",
    "aug": "summer",
    "sep": "autumn",
    "oct": "autumn",
    "nov": "autumn",
    "dec": "winter",
    "jan": "winter",
    "feb": "winter",
}

# Global model variable
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events for the FastAPI app."""
    global model
    try:
        start_time = time.time()
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)
        load_time = time.time() - start_time
        logger.info(
            f"Model loaded successfully from {MODEL_PATH} in {load_time:.2f} seconds"
        )
        logger.info(f"Using static prediction threshold: {PREDICTION_THRESHOLD}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Could not load model: {str(e)}")

    yield

    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="API for predicting customer subscription to bank term deposit",
    version="1.0.2",
    lifespan=lifespan,
)


# Pydantic models for request validation
class CustomerData(BaseModel):
    """Input data schema for a single customer."""

    age: int = Field(..., description="Customer age", examples=[41], ge=18, le=100)
    occupation: str = Field(
        ..., description="Type of job/occupation", examples=["technician"]
    )
    marital_status: str = Field(..., description="Marital status", examples=["married"])
    education: str = Field(..., description="Education level", examples=["university"])
    housing_loan: str = Field(..., description="Has housing loan", examples=["yes"])
    personal_loan: str = Field(..., description="Has personal loan", examples=["no"])
    contact_mode: str = Field(
        ..., description="Contact communication mode", examples=["cellular"]
    )
    month: str = Field(
        ..., description="Last contact month (3-letter code)", examples=["may"]
    )
    week_day: str = Field(
        ...,
        description="Last contact day of the week (3-letter code)",
        examples=["mon"],
    )
    last_contact_duration: int = Field(
        ..., description="Last contact duration in seconds", examples=[180], ge=0
    )
    contacts_per_campaign: int = Field(
        ..., description="Number of contacts during this campaign", examples=[2], ge=0
    )
    previous_outcome: str = Field(
        ..., description="Outcome of previous campaign", examples=["success"]
    )
    emp_var_rate: float = Field(
        ..., description="Employment variation rate", examples=[-0.1]
    )
    cons_price_index: float = Field(
        ..., description="Consumer price index", examples=[93.2]
    )
    euri_3_month: float = Field(..., description="Euribor 3 month rate", examples=[0.6])

    @field_validator("age")
    def validate_age(cls, v):
        # Additional validation beyond the Field constraints
        if v > 100:
            raise ValueError("Age is unrealistically high (>100)")
        return v

    @field_validator("occupation")
    def validate_occupation(cls, v):
        normalized = v.lower()
        if normalized not in [occ.lower() for occ in VALID_OCCUPATIONS]:
            raise ValueError(f"Occupation must be one of {VALID_OCCUPATIONS}")
        return normalized

    @field_validator("marital_status")
    def validate_marital_status(cls, v):
        normalized = v.lower()
        if normalized not in VALID_MARITAL_STATUSES:
            raise ValueError(f"Marital status must be one of {VALID_MARITAL_STATUSES}")
        return normalized

    @field_validator("education")
    def validate_education(cls, v):
        normalized = v.lower()
        if normalized not in [edu.lower() for edu in VALID_EDUCATION_LEVELS]:
            raise ValueError(f"Education must be one of {VALID_EDUCATION_LEVELS}")
        return normalized

    @field_validator("housing_loan", "personal_loan")
    def validate_yes_no(cls, v, info):
        normalized = v.lower()
        if normalized not in ["yes", "no"]:
            raise ValueError(f"{info.field_name} must be 'yes' or 'no'")
        return normalized

    @field_validator("contact_mode")
    def validate_contact_mode(cls, v):
        normalized = v.lower()
        if normalized not in VALID_CONTACT_MODES:
            raise ValueError(f"Contact mode must be one of {VALID_CONTACT_MODES}")
        return normalized

    @field_validator("month")
    def validate_month(cls, v):
        normalized = v.lower()
        if normalized not in VALID_MONTHS:
            raise ValueError(f"Month must be one of {VALID_MONTHS}")
        return normalized

    @field_validator("week_day")
    def validate_week_day(cls, v):
        normalized = v.lower()
        if normalized not in VALID_DAYS:
            raise ValueError(f"Day of week must be one of {VALID_DAYS}")
        return normalized

    @field_validator("last_contact_duration")
    def validate_duration(cls, v):
        # Additional validation beyond the Field constraints
        if v > 3600:  # Example: 1 hour max
            raise ValueError("Contact duration is unrealistically long (>3600 seconds)")
        return v

    @field_validator("contacts_per_campaign")
    def validate_contacts(cls, v):
        # Additional validation beyond the Field constraints
        if v > 50:  # Example threshold
            raise ValueError(
                "Number of contacts per campaign is unrealistically high (>50)"
            )
        return v

    @field_validator("previous_outcome")
    def validate_previous_outcome(cls, v):
        normalized = v.lower()
        if normalized not in VALID_PREVIOUS_OUTCOMES:
            raise ValueError(
                f"Previous outcome must be one of {VALID_PREVIOUS_OUTCOMES}"
            )
        return normalized

    @field_validator("emp_var_rate")
    def validate_emp_var_rate(cls, v):
        # Typical range for employment variation rate
        if not -10.0 <= v <= 10.0:
            raise ValueError("Employment variation rate must be between -10.0 and 10.0")
        return v

    @field_validator("cons_price_index")
    def validate_cons_price_index(cls, v):
        # Typical range for consumer price index
        if not 80.0 <= v <= 110.0:
            raise ValueError("Consumer price index must be between 80.0 and 110.0")
        return v

    @field_validator("euri_3_month")
    def validate_euri_3_month(cls, v):
        # Typical range for Euribor 3 month rate
        if not -2.0 <= v <= 10.0:
            raise ValueError("Euribor 3 month rate must be between -2.0 and 10.0")
        return v


class PredictionRequest(BaseModel):
    """Request containing one or more customer records."""

    customers: List[CustomerData] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    """Response with prediction results."""

    predictions: List[Dict[str, Any]]
    model_version: str
    threshold_used: float
    processing_time_ms: float


class ErrorResponse(BaseModel):
    """Standardized error response."""

    status: str = "error"
    message: str
    detail: Union[str, Dict[str, Any], None] = None


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features required by the model.

    Parameters
    ----------
    df : pd.DataFrame
        Original customer data DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with added features.
    """
    # Make a copy to avoid modifying the input
    df = df.copy()

    # === Age binning ===
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 55, np.inf],
        labels=["young", "middle", "senior"],
        include_lowest=True,
    )

    # === Education × Marital cross-feature ===
    df["education_marital"] = (
        df["education"].astype(str) + "_" + df["marital_status"].astype(str)
    )

    # === Interest spread: CPI - Euribor ===
    df["interest_diff"] = df["cons_price_index"] - df["euri_3_month"]

    # === Month + weekday combined ===
    df["month_weekday"] = df["month"].astype(str) + "_" + df["week_day"].astype(str)

    # === Interest pressure ratio ===
    df["interest_pressure_ratio"] = df["cons_price_index"] / (df["euri_3_month"] + 1e-3)

    # === Log features (avoid skew) ===
    df["log_duration"] = np.log1p(df["last_contact_duration"])

    # === Month → Season mapping ===
    df["season"] = df["month"].map(MONTH_TO_SEASON)

    # === Loan burden score ===
    df["loan_burden_score"] = (df["housing_loan"] == "yes").astype(int) + (
        df["personal_loan"] == "yes"
    ).astype(int)

    # === Volatility score ===
    df["volatility_score"] = (
        df["emp_var_rate"].abs()
        + df["cons_price_index"].abs()
        + df["euri_3_month"].abs()
    )

    return df


def check_model_loaded():
    """Check if model is loaded, raise exception if not."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later.",
        )
    return model


@app.get("/health", response_model_exclude_none=True)
async def health_check():
    """Health check endpoint."""
    if model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=jsonable_encoder(
                ErrorResponse(message="Service unhealthy - model not loaded")
            ),
        )
    return {
        "status": "ok",
        "message": "Service is healthy",
        "threshold": PREDICTION_THRESHOLD,
        "model_path": MODEL_PATH,
    }


@app.get("/model-info")
async def model_info(model: CatBoostClassifier = Depends(check_model_loaded)):
    """Get information about the loaded model."""
    return {
        "model_type": "CatBoostClassifier",
        "num_features": len(model.feature_names_),
        "feature_names": model.feature_names_,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_file": str(MODEL_PATH),
        "prediction_threshold": PREDICTION_THRESHOLD,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={500: {"model": ErrorResponse}},
)
async def predict(
    request: PredictionRequest, model: CatBoostClassifier = Depends(check_model_loaded)
):
    """
    Make predictions for bank marketing customers.

    Returns predictions with probabilities and binary classes
    using the static threshold (PREDICTION_THRESHOLD).
    """
    start_time = time.time()

    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame(
            [customer.model_dump() for customer in request.customers]
        )

        # Add engineered features
        processed_df = add_engineered_features(input_df)

        # Ensure all categorical features are correctly typed
        for cat_feat in CATEGORICAL_FEATURES:
            if cat_feat in processed_df.columns:
                processed_df[cat_feat] = processed_df[cat_feat].astype(str)

        # Make predictions
        try:
            logger.info(f"Making prediction with shape: {processed_df.shape}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Features: {processed_df.columns.tolist()}")
                logger.debug(f"Sample row: {processed_df.iloc[0].to_dict()}")

            # Create CatBoost Pool with explicit categorical features
            pool = Pool(data=processed_df, cat_features=CATEGORICAL_FEATURES)

            probabilities = model.predict_proba(pool)[:, 1]
            prediction_classes = (probabilities >= PREDICTION_THRESHOLD).astype(int)
        except Exception as pred_error:
            logger.error(f"Prediction error details: {str(pred_error)}")
            logger.error(f"Categorical features: {CATEGORICAL_FEATURES}")
            for cat_feat in CATEGORICAL_FEATURES:
                if cat_feat in processed_df.columns:
                    logger.error(
                        f"Feature {cat_feat} - dtype: {processed_df[cat_feat].dtype}"
                    )
                    logger.error(
                        f"Sample values: {processed_df[cat_feat].head().tolist()}"
                    )
            raise

        # Prepare response
        predictions = []
        for i, (prob, pred_class) in enumerate(zip(probabilities, prediction_classes)):
            predictions.append(
                {
                    "customer_index": i,
                    "probability": float(prob),
                    "prediction": int(pred_class),
                    "prediction_label": "yes" if pred_class == 1 else "no",
                }
            )

        processing_time = time.time() - start_time

        return PredictionResponse(
            predictions=predictions,
            model_version=str(MODEL_PATH),
            threshold_used=PREDICTION_THRESHOLD,
            processing_time_ms=processing_time * 1000,
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        processing_time = time.time() - start_time

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=jsonable_encoder(
                ErrorResponse(message="Prediction failed", detail=str(e))
            ),
        )


@app.post(
    "/batch-predict",
    response_model=PredictionResponse,
    responses={500: {"model": ErrorResponse}},
)
async def batch_predict(
    request: PredictionRequest, model: CatBoostClassifier = Depends(check_model_loaded)
):
    """
    Alias for '/predict' - optimized for batch processing.
    """
    return await predict(request, model)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
