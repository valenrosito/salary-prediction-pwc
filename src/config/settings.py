import os
from pathlib import Path

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Rutas de datos
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Rutas de Modelos
MODEL_DIR = ROOT_DIR / "model"
PIPELINE_PATH = MODEL_DIR / "preprocessor_pipeline.joblib"
MODEL_PATH = MODEL_DIR / "predictor_model.joblib"

# Configuración de los archivos de datos
DATA_FILES = {
    "people": RAW_DATA_DIR / "people.csv",
    "salary": RAW_DATA_DIR / "salary.csv",
    "descriptions": RAW_DATA_DIR / "descriptions.csv",
}

# Parámetros del modelo (ejemplo inicial)
MODEL_CONFIG = {
    "random_state": 42,
    "target": "salary_usd",
    "features": ["gender", "seniority", "education_level"], # Se actualizará después del EDA
    "test_size": 0.2,
}