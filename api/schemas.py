from typing import Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    gender:           Literal["Male", "Female"]
    SeniorCitizen:    int   = Field(..., ge=0, le=1)
    Partner:          Literal["Yes", "No"]
    Dependents:       Literal["Yes", "No"]
    tenure:           int   = Field(..., ge=0)
    PhoneService:     Literal["Yes", "No"]
    MultipleLines:    Literal["Yes", "No", "No phone service"]
    InternetService:  Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity:   Literal["Yes", "No", "No internet service"]
    OnlineBackup:     Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport:      Literal["Yes", "No", "No internet service"]
    StreamingTV:      Literal["Yes", "No", "No internet service"]
    StreamingMovies:  Literal["Yes", "No", "No internet service"]
    Contract:         Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod:    Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges:   float = Field(..., gt=0)
    TotalCharges:     float = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": 1020.0,
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Churn olasılığı [0, 1]")
    churn_prediction:  bool  = Field(..., description="Churn tahmini")
    threshold_used:    float = Field(default=0.5)
