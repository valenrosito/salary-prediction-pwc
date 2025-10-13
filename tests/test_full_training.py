import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split
from src.data_processing.pipeline import DataPipeline
from src.model.trainer import ModelTrainer
from src.model.evaluator import ModelEvaluator
from src.config.settings import MODEL_CONFIG, MODEL_DIR
from src.utils.persistence import save_object

def main():
    print("="*70)
    print(" ENTRENAMIENTO Y EVALUACIÓN DE MODELOS - PREDICCIÓN DE SALARIOS")
    print("="*70)
    
    print("\n1. Cargando y preparando datos...")
    pipeline = DataPipeline()
    X, y, df = pipeline.load_and_prepare_data()
    print(f"   ✓ Datos cargados: {df.shape}")
    
    print("\n2. Dividiendo datos en train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state']
    )
    print(f"   ✓ Train: {X_train.shape}, Test: {X_test.shape}")
    
    print("\n3. Creando pipeline de preprocesamiento...")
    X_train_transformed = pipeline.fit_transform(X_train)
    print(f"   ✓ Features generadas: {X_train_transformed.shape[1]}")
    
    trainer = ModelTrainer(pipeline.preprocessing_pipeline)
    evaluator = ModelEvaluator()
    
    all_results = {}
    
    all_results['baseline'] = trainer.train_baseline(X_train, y_train, X_test, y_test)
    
    all_results['lgbm'] = trainer.train_model(X_train, y_train, X_test, y_test)
    
    all_results['random_forest'] = trainer.train_random_forest(X_train, y_train, X_test, y_test)
    
    all_results['neural_network'] = trainer.train_neural_network(X_train, y_train, X_test, y_test)
    
    print("\n" + "="*80)
    print(" COMPARACIÓN DE TODOS LOS MODELOS")
    print("="*80)
    
    print(f"\n{'Modelo':<25} {'R² Score':<15} {'RMSE':<15} {'MAE':<15}")
    print("-"*80)
    
    for model_name, results in all_results.items():
        r2 = results['metrics']['r2']
        rmse = results['metrics']['rmse']
        mae = results['metrics']['mae']
        print(f"{model_name:<25} {r2:>6.4f}          {rmse:>10.2f}      {mae:>10.2f}")
    
    best_model_name = max(all_results.items(), key=lambda x: x[1]['metrics']['r2'])[0]
    best_r2 = all_results[best_model_name]['metrics']['r2']
    
    print("\n" + "-"*80)
    print(f"🏆 MEJOR MODELO: {best_model_name.upper()} (R² = {best_r2:.4f})")
    print("-"*80)
    
    print("\n" + "="*80)
    print(" MEJORAS RESPECTO AL BASELINE")
    print("="*80)
    
    baseline_metrics = all_results['baseline']['metrics']
    
    print(f"\n{'Modelo':<25} {'Mejora R²':<15} {'Reducción RMSE':<20} {'Reducción MAE':<15}")
    print("-"*80)
    
    for model_name, results in all_results.items():
        if model_name == 'baseline':
            continue
        
        r2_improvement = ((results['metrics']['r2'] - baseline_metrics['r2']) / abs(baseline_metrics['r2'])) * 100 if baseline_metrics['r2'] != 0 else 0
        rmse_reduction = ((baseline_metrics['rmse'] - results['metrics']['rmse']) / baseline_metrics['rmse']) * 100
        mae_reduction = ((baseline_metrics['mae'] - results['metrics']['mae']) / baseline_metrics['mae']) * 100
        
        print(f"{model_name:<25} {r2_improvement:>8.2f}%       {rmse_reduction:>10.2f}%          {mae_reduction:>10.2f}%")
    
    print("\n" + "="*80)
    print(" GUARDANDO MODELOS Y MÉTRICAS")
    print("="*80)
    
    metrics_path = Path(MODEL_DIR) / "metrics.json"
    
    for model_name, results in all_results.items():
        display_name = {
            'baseline': 'Baseline (DummyRegressor)',
            'lgbm': 'LGBMRegressor',
            'random_forest': 'Random Forest',
            'neural_network': 'Neural Network (TensorFlow)'
        }[model_name]
        
        evaluator.log_metrics(
            results['metrics'],
            results['confidence_intervals'],
            display_name,
            metrics_path
        )
    
    save_object(pipeline, Path(MODEL_DIR) / "preprocessor_pipeline.joblib")
    save_object(all_results[best_model_name]['model'], Path(MODEL_DIR) / "predictor_model.joblib")
    print(f"\n✓ Mejor modelo guardado: {best_model_name}")
    
    if best_model_name in ['lgbm', 'random_forest']:
        print("\n" + "="*80)
        print(f" IMPORTANCIA DE FEATURES ({best_model_name.upper()})")
        print("="*80)
        trainer.final_model = all_results[best_model_name]['model']
        trainer.get_feature_importance(top_n=15)
    
    print("\n" + "="*80)
    print(" ✓ PROCESO COMPLETADO EXITOSAMENTE")
    print("="*80)

if __name__ == "__main__":
    main()
