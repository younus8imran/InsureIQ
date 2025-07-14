from pydantic import BaseModel, Field

class InsuranceInput(BaseModel):
    age: int = Field(0, ge=18, le=100, description="Age in years")
    sex: str = Field("", pattern="^(male|female)$", description="Sex")
    bmi: float = Field(..., ge=0, description="Body Mass Index")
    children: int = Field(0, ge=0, lt=10, description="No. of childrens")
    smoker: str = Field("", pattern="^(yes|no)$", description="Smoking Status")
    region: str = Field(
        "",
        pattern="^(northwest|northeast|southeast|southwest)$",
        description="US region",
    )

class PredictionOut(BaseModel):
    charges: float = Field(..., description="Predictd yearly medical charge")
