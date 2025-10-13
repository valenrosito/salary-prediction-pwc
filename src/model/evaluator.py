import numpy as np
import json
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple


class ModelEvaluator:
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        return {
            "r2": r2,
            "rmse": rmse,
            "mae": mae
        }
    
    def calculate_confidence_intervals(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        n_iterations: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        n_samples = len(y_true)
        
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        
        rng = np.random.RandomState(42)
        
        for _ in range(n_iterations):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            r2_scores.append(r2_score(y_true_boot, y_pred_boot))
            rmse_scores.append(np.sqrt(mean_squared_error(y_true_boot, y_pred_boot)))
            mae_scores.append(mean_absolute_error(y_true_boot, y_pred_boot))
        
        alpha = (1 - confidence_level) / 2
        lower_percentile = alpha * 100
        upper_percentile = (1 - alpha) * 100
        
        confidence_intervals = {
            "r2_ci": (
                np.percentile(r2_scores, lower_percentile),
                np.percentile(r2_scores, upper_percentile)
            ),
            "rmse_ci": (
                np.percentile(rmse_scores, lower_percentile),
                np.percentile(rmse_scores, upper_percentile)
            ),
            "mae_ci": (
                np.percentile(mae_scores, lower_percentile),
                np.percentile(mae_scores, upper_percentile)
            )
        }
        
        return confidence_intervals
    
    @staticmethod
    def log_metrics(
        metrics: Dict[str, float],
        confidence_intervals: Dict[str, Tuple[float, float]],
        model_name: str,
        filepath: Path
    ) -> None:
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        metrics_data = {
            "model_name": model_name,
            "metrics": {
                "r2": {
                    "value": metrics["r2"],
                    "ci_lower": confidence_intervals["r2_ci"][0],
                    "ci_upper": confidence_intervals["r2_ci"][1]
                },
                "rmse": {
                    "value": metrics["rmse"],
                    "ci_lower": confidence_intervals["rmse_ci"][0],
                    "ci_upper": confidence_intervals["rmse_ci"][1]
                },
                "mae": {
                    "value": metrics["mae"],
                    "ci_lower": confidence_intervals["mae_ci"][0],
                    "ci_upper": confidence_intervals["mae_ci"][1]
                }
            }
        }
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                existing_data.append(metrics_data)
            else:
                existing_data = [existing_data, metrics_data]
        else:
            existing_data = [metrics_data]
        
        with open(filepath, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        print(f"✓ Métricas guardadas en: {filepath}")
