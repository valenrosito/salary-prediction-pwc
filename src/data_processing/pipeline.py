import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import set_config

from src.config.settings import (
    DATA_FILES,
    TARGET_VARIABLE,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TEXT_FEATURES,
    PREPROCESSING_CONFIG,
)

set_config(transform_output="pandas")


class DataPipeline:
    def __init__(self):
        self.preprocessing_pipeline = None
        self.feature_names = None
        
    def _merge_data(self):
        people_df = pd.read_csv(DATA_FILES["people"])
        salary_df = pd.read_csv(DATA_FILES["salary"])
        descriptions_df = pd.read_csv(DATA_FILES["descriptions"])
        
        merged_df = (
            people_df
            .merge(salary_df, on="id", how="inner")
            .merge(descriptions_df, on="id", how="inner")
        )
        
        return merged_df
    
    def _create_preprocessing_pipeline(self):
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(
                strategy=PREPROCESSING_CONFIG["numeric_imputer_strategy"]
            )),
            ("scaler", StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(
                strategy=PREPROCESSING_CONFIG["categorical_imputer_strategy"]
            )),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        
        text_transformer = TfidfVectorizer(
            max_features=PREPROCESSING_CONFIG["tfidf_max_features"],
            max_df=PREPROCESSING_CONFIG["tfidf_max_df"],
            min_df=PREPROCESSING_CONFIG["tfidf_min_df"],
            stop_words="english"
        )
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERIC_FEATURES),
                ("cat", categorical_transformer, CATEGORICAL_FEATURES),
                ("text", text_transformer, TEXT_FEATURES[0])
            ],
            remainder="drop"
        )
        
        return preprocessor
    
    def load_and_prepare_data(self):
        df = self._merge_data()
        
        df = df.dropna(subset=[TARGET_VARIABLE])
        
        df = df[df[TARGET_VARIABLE] >= 20000]
        
        df[TEXT_FEATURES[0]] = df[TEXT_FEATURES[0]].fillna("")
        
        X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES]
        y = df[TARGET_VARIABLE]
        
        return X, y, df
    
    def fit_transform(self, X, y=None):
        if self.preprocessing_pipeline is None:
            self.preprocessing_pipeline = self._create_preprocessing_pipeline()
        
        X_transformed = self.preprocessing_pipeline.fit_transform(X)
        
        return X_transformed
    
    def transform(self, X):
        if self.preprocessing_pipeline is None:
            raise ValueError("Pipeline no entrenado. Ejecuta fit_transform primero.")
        
        X_transformed = self.preprocessing_pipeline.transform(X)
        
        return X_transformed
    
    def get_feature_names(self):
        if self.preprocessing_pipeline is None:
            raise ValueError("Pipeline no entrenado. Ejecuta fit_transform primero.")
        
        feature_names = []
        
        for name, transformer, features in self.preprocessing_pipeline.transformers_:
            if name == "num":
                feature_names.extend(features)
            elif name == "cat":
                if hasattr(transformer.named_steps["encoder"], "get_feature_names_out"):
                    cat_features = transformer.named_steps["encoder"].get_feature_names_out(features)
                    feature_names.extend(cat_features)
            elif name == "text":
                if hasattr(transformer, "get_feature_names_out"):
                    text_features = transformer.get_feature_names_out()
                    feature_names.extend([f"text_{feat}" for feat in text_features])
        
        return feature_names
