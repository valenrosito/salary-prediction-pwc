from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from src.utils.persistence import load_object
from src.config.settings import MODEL_DIR

app = FastAPI(
    title="Salary Prediction API",
    description="API para predecir salarios basándose en datos demográficos y profesionales",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
preprocessing_pipeline = None


@app.on_event("startup")
async def load_model():
    global model, preprocessing_pipeline
    try:
        model_path = Path(MODEL_DIR) / "predictor_model.joblib"
        pipeline_path = Path(MODEL_DIR) / "preprocessor_pipeline.joblib"
        
        model = load_object(model_path)
        preprocessing_pipeline = load_object(pipeline_path)
        
        print("✓ Modelo y pipeline cargados exitosamente")
    except Exception as e:
        print(f"✗ Error al cargar el modelo: {e}")
        raise


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Salary Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_salary(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        input_data = pd.DataFrame([{
            "Age": request.age,
            "Gender": request.gender,
            "Education Level": request.education_level,
            "Job Title": request.job_title,
            "Years of Experience": request.years_of_experience,
            "Description": request.description
        }])
        
        prediction = model.predict(input_data)
        
        predicted_salary = float(prediction[0])
        
        return PredictionResponse(predicted_salary=predicted_salary)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
