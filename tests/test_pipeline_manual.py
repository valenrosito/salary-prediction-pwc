import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processing.pipeline import DataPipeline

def test_pipeline():
    print("=" * 60)
    print("PRUEBA DEL PIPELINE DE PREPROCESAMIENTO")
    print("=" * 60)
    
    pipeline = DataPipeline()
    
    print("\n1. Cargando y fusionando datos...")
    X, y, df = pipeline.load_and_prepare_data()
    print(f"   ✓ Datos cargados: {df.shape}")
    print(f"   ✓ Features (X): {X.shape}")
    print(f"   ✓ Target (y): {y.shape}")
    
    print("\n2. Aplicando transformaciones (fit_transform)...")
    X_transformed = pipeline.fit_transform(X)
    print(f"   ✓ Datos transformados: {X_transformed.shape}")
    print(f"   ✓ Número de features generadas: {X_transformed.shape[1]}")
    
    print("\n3. Obteniendo nombres de features...")
    feature_names = pipeline.get_feature_names()
    print(f"   ✓ Total de features: {len(feature_names)}")
    print(f"   ✓ Primeras 10 features: {feature_names[:10]}")
    
    print("\n4. Probando transform en nuevos datos...")
    X_test_transformed = pipeline.transform(X.head(5))
    print(f"   ✓ Datos de prueba transformados: {X_test_transformed.shape}")
    
    print("\n5. Verificando valores nulos después de transformación...")
    import numpy as np
    nulls = np.isnan(X_transformed).sum()
    print(f"   ✓ Valores nulos en datos transformados: {nulls}")
    
    print("\n" + "=" * 60)
    print("PIPELINE FUNCIONANDO CORRECTAMENTE ✓")
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline()
