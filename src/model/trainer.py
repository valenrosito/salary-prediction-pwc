import numpy as np
import warnings
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from lightgbm import LGBMRegressor
from typing import Tuple, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config.settings import MODEL_CONFIG
from src.model.evaluator import ModelEvaluator

warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
tf.get_logger().setLevel('ERROR')


class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim=None, epochs=100, batch_size=32, verbose=0):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
        self.scaler_mean_ = None
        self.scaler_std_ = None
        
    def _build_model(self):
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_dim=self.input_dim),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def fit(self, X, y):
        if isinstance(X, (list, tuple)):
            X = np.array(X)
        if isinstance(y, (list, tuple)):
            y = np.array(y)
            
        if self.input_dim is None:
            self.input_dim = X.shape[1]
        
        self.scaler_mean_ = np.mean(y)
        self.scaler_std_ = np.std(y)
        y_scaled = (y - self.scaler_mean_) / (self.scaler_std_ + 1e-8)
        
        self.model = self._build_model()
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.model.fit(
            X, y_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            callbacks=[early_stopping],
            validation_split=0.1
        )
        
        return self
    
    def predict(self, X):
        if isinstance(X, (list, tuple)):
            X = np.array(X)
            
        y_scaled = self.model.predict(X, verbose=0)
        y_pred = (y_scaled * self.scaler_std_) + self.scaler_mean_
        return y_pred.flatten()


class ModelTrainer:
    
    def __init__(self, preprocessing_pipeline):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.evaluator = ModelEvaluator()
        self.baseline_model = None
        self.final_model = None
        
    def train_baseline(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        
        print("\n" + "="*60)
        print("ENTRENANDO MODELO BASELINE (DummyRegressor)")
        print("="*60)
        
        baseline = Pipeline([
            ('preprocessor', self.preprocessing_pipeline),
            ('regressor', DummyRegressor(strategy="mean"))
        ])
        
        baseline.fit(X_train, y_train)
        self.baseline_model = baseline
        
        y_pred = baseline.predict(X_test)
        
        metrics = self.evaluator.calculate_metrics(y_test, y_pred)
        confidence_intervals = self.evaluator.calculate_confidence_intervals(y_test, y_pred)
        
        print(f"\nMétricas del Baseline:")
        print(f"  R² Score: {metrics['r2']:.4f} (CI: [{confidence_intervals['r2_ci'][0]:.4f}, {confidence_intervals['r2_ci'][1]:.4f}])")
        print(f"  RMSE:     {metrics['rmse']:.2f} (CI: [{confidence_intervals['rmse_ci'][0]:.2f}, {confidence_intervals['rmse_ci'][1]:.2f}])")
        print(f"  MAE:      {metrics['mae']:.2f} (CI: [{confidence_intervals['mae_ci'][0]:.2f}, {confidence_intervals['mae_ci'][1]:.2f}])")
        
        return {
            "model": baseline,
            "metrics": metrics,
            "confidence_intervals": confidence_intervals,
            "predictions": y_pred
        }
    
    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_params: Dict = None
    ) -> Dict:
        
        print("\n" + "="*60)
        print("ENTRENANDO MODELO AVANZADO (LGBMRegressor)")
        print("="*60)
        
        if model_params is None:
            model_params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 6,
                'num_leaves': 31,
                'min_child_samples': 20,
                'random_state': MODEL_CONFIG['random_state'],
                'verbose': -1
            }
        
        lgbm = LGBMRegressor(**model_params)
        
        final_pipeline = Pipeline([
            ('preprocessor', self.preprocessing_pipeline),
            ('regressor', lgbm)
        ])
        
        print("\nRealizando validación cruzada...")
        cv_scores = cross_val_score(
            final_pipeline, 
            X_train, 
            y_train,
            cv=MODEL_CONFIG['cv_folds'],
            scoring='r2',
            n_jobs=-1
        )
        
        print(f"  CV R² Scores: {cv_scores}")
        print(f"  CV R² Mean:   {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        print("\nEntrenando modelo final con todos los datos de entrenamiento...")
        final_pipeline.fit(X_train, y_train)
        self.final_model = final_pipeline
        
        print("\nEvaluando en conjunto de prueba...")
        y_pred = final_pipeline.predict(X_test)
        
        metrics = self.evaluator.calculate_metrics(y_test, y_pred)
        confidence_intervals = self.evaluator.calculate_confidence_intervals(y_test, y_pred)
        
        print(f"\nMétricas del Modelo Final:")
        print(f"  R² Score: {metrics['r2']:.4f} (CI: [{confidence_intervals['r2_ci'][0]:.4f}, {confidence_intervals['r2_ci'][1]:.4f}])")
        print(f"  RMSE:     {metrics['rmse']:.2f} (CI: [{confidence_intervals['rmse_ci'][0]:.2f}, {confidence_intervals['rmse_ci'][1]:.2f}])")
        print(f"  MAE:      {metrics['mae']:.2f} (CI: [{confidence_intervals['mae_ci'][0]:.2f}, {confidence_intervals['mae_ci'][1]:.2f}])")
        
        return {
            "model": final_pipeline,
            "metrics": metrics,
            "confidence_intervals": confidence_intervals,
            "predictions": y_pred,
            "cv_scores": cv_scores
        }
    
    def get_feature_importance(self, top_n: int = 20):
        if self.final_model is None:
            raise ValueError("Modelo no entrenado. Ejecuta train_model primero.")
        
        regressor = self.final_model.named_steps['regressor']
        
        if not hasattr(regressor, 'feature_importances_'):
            print("El modelo no tiene importancia de features.")
            return None
        
        preprocessor = self.final_model.named_steps['preprocessor']
        feature_names = []
        
        for name, transformer, features in preprocessor.transformers_:
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
        
        importances = regressor.feature_importances_
        
        feature_importance_dict = dict(zip(feature_names, importances))
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop {top_n} Features más importantes:")
        for i, (feature, importance) in enumerate(sorted_features[:top_n], 1):
            print(f"  {i:2d}. {feature:40s} {importance:.4f}")
        
        return sorted_features
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_params: Dict = None
    ) -> Dict:
        
        print("\n" + "="*60)
        print("ENTRENANDO MODELO RANDOM FOREST")
        print("="*60)
        
        if model_params is None:
            model_params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': MODEL_CONFIG['random_state'],
                'n_jobs': -1
            }
        
        rf = RandomForestRegressor(**model_params)
        
        rf_pipeline = Pipeline([
            ('preprocessor', self.preprocessing_pipeline),
            ('regressor', rf)
        ])
        
        print("\nRealizando validación cruzada...")
        cv_scores = cross_val_score(
            rf_pipeline, 
            X_train, 
            y_train,
            cv=MODEL_CONFIG['cv_folds'],
            scoring='r2',
            n_jobs=-1
        )
        
        print(f"  CV R² Scores: {cv_scores}")
        print(f"  CV R² Mean:   {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        print("\nEntrenando modelo final con todos los datos de entrenamiento...")
        rf_pipeline.fit(X_train, y_train)
        
        print("\nEvaluando en conjunto de prueba...")
        y_pred = rf_pipeline.predict(X_test)
        
        metrics = self.evaluator.calculate_metrics(y_test, y_pred)
        confidence_intervals = self.evaluator.calculate_confidence_intervals(y_test, y_pred)
        
        print(f"\nMétricas del Random Forest:")
        print(f"  R² Score: {metrics['r2']:.4f} (CI: [{confidence_intervals['r2_ci'][0]:.4f}, {confidence_intervals['r2_ci'][1]:.4f}])")
        print(f"  RMSE:     {metrics['rmse']:.2f} (CI: [{confidence_intervals['rmse_ci'][0]:.2f}, {confidence_intervals['rmse_ci'][1]:.2f}])")
        print(f"  MAE:      {metrics['mae']:.2f} (CI: [{confidence_intervals['mae_ci'][0]:.2f}, {confidence_intervals['mae_ci'][1]:.2f}])")
        
        return {
            "model": rf_pipeline,
            "metrics": metrics,
            "confidence_intervals": confidence_intervals,
            "predictions": y_pred,
            "cv_scores": cv_scores
        }
    
    def train_neural_network(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_params: Dict = None
    ) -> Dict:
        
        print("\n" + "="*60)
        print("ENTRENANDO RED NEURONAL (TensorFlow/Keras)")
        print("="*60)
        
        X_train_transformed = self.preprocessing_pipeline.transform(X_train)
        X_test_transformed = self.preprocessing_pipeline.transform(X_test)
        
        if hasattr(X_train_transformed, 'values'):
            X_train_transformed = X_train_transformed.values
        if hasattr(X_test_transformed, 'values'):
            X_test_transformed = X_test_transformed.values
        
        if model_params is None:
            model_params = {
                'input_dim': X_train_transformed.shape[1],
                'epochs': 150,
                'batch_size': 32,
                'verbose': 0
            }
        else:
            model_params['input_dim'] = X_train_transformed.shape[1]
        
        nn = KerasRegressorWrapper(**model_params)
        
        print("\nEntrenando red neuronal...")
        print(f"  Arquitectura: 256→128→64→32→1")
        print(f"  Épocas máximas: {model_params['epochs']}")
        print(f"  Batch size: {model_params['batch_size']}")
        
        nn.fit(X_train_transformed, y_train)
        
        print("\nEvaluando en conjunto de prueba...")
        y_pred = nn.predict(X_test_transformed)
        
        metrics = self.evaluator.calculate_metrics(y_test, y_pred)
        confidence_intervals = self.evaluator.calculate_confidence_intervals(y_test, y_pred)
        
        print(f"\nMétricas de la Red Neuronal:")
        print(f"  R² Score: {metrics['r2']:.4f} (CI: [{confidence_intervals['r2_ci'][0]:.4f}, {confidence_intervals['r2_ci'][1]:.4f}])")
        print(f"  RMSE:     {metrics['rmse']:.2f} (CI: [{confidence_intervals['rmse_ci'][0]:.2f}, {confidence_intervals['rmse_ci'][1]:.2f}])")
        print(f"  MAE:      {metrics['mae']:.2f} (CI: [{confidence_intervals['mae_ci'][0]:.2f}, {confidence_intervals['mae_ci'][1]:.2f}])")
        
        return {
            "model": nn,
            "metrics": metrics,
            "confidence_intervals": confidence_intervals,
            "predictions": y_pred
        }
