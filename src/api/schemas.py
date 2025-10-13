from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    age: float = Field(..., description="Edad del individuo", ge=18, le=100)
    gender: str = Field(..., description="Género (Male/Female)")
    education_level: str = Field(..., description="Nivel educativo (Bachelor's/Master's/PhD)")
    job_title: str = Field(..., description="Título del trabajo")
    years_of_experience: float = Field(..., description="Años de experiencia laboral", ge=0, le=50)
    description: str = Field(..., description="Descripción del perfil profesional")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 32,
                "gender": "Male",
                "education_level": "Bachelor's",
                "job_title": "Software Engineer",
                "years_of_experience": 5,
                "description": "Experienced software engineer with strong background in Python and machine learning"
            }
        }


class PredictionResponse(BaseModel):
    predicted_salary: float = Field(..., description="Salario predicho en USD")
    
    class Config:
        schema_extra = {
            "example": {
                "predicted_salary": 90000.0
            }
        }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
