from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = ROOT_DIR / "models"
PIPELINE_PATH = MODEL_DIR / "preprocessor_pipeline.joblib"
MODEL_PATH = MODEL_DIR / "predictor_model.joblib"

DATA_FILES = {
    "people": RAW_DATA_DIR / "people.csv",
    "salary": RAW_DATA_DIR / "salary.csv",
    "descriptions": RAW_DATA_DIR / "descriptions.csv",
}

TARGET_VARIABLE = "Salary"

NUMERIC_FEATURES = ["Age", "Years of Experience"]

CATEGORICAL_FEATURES = ["Gender", "Education Level", "Job Title"]

TEXT_FEATURES = ["Description"]

PREPROCESSING_CONFIG = {
    "numeric_imputer_strategy": "median",
    "categorical_imputer_strategy": "most_frequent",
    "tfidf_max_features": 200,
    "tfidf_max_df": 0.95,
    "tfidf_min_df": 2,
}

MODEL_CONFIG = {
    "random_state": 42,
    "test_size": 0.2,
    "cv_folds": 5,
}