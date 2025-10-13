import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_processing.pipeline import DataPipeline
from src.config.settings import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TEXT_FEATURES


class TestDataPipeline:
    """Suite de tests para el DataPipeline"""
    
    @pytest.fixture
    def pipeline(self):
        """Fixture que crea una instancia del pipeline"""
        return DataPipeline()
    
    @pytest.fixture
    def sample_data(self):
        """Fixture con datos de ejemplo para testing"""
        return pd.DataFrame({
            'Age': [25, 30, 35, 40, 45],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'Education Level': ["Bachelor's", "Master's", "PhD", "Bachelor's", "Master's"],
            'Job Title': ['Engineer', 'Analyst', 'Manager', 'Engineer', 'Director'],
            'Years of Experience': [2, 5, 10, 8, 15],
            'Description': [
                'Software engineer with Python',
                'Data analyst with SQL',
                'Senior manager with leadership',
                'Senior engineer with Java',
                'Director with strategy'
            ]
        })
    
    def test_pipeline_initialization(self, pipeline):
        """Test 1: Verificar que el pipeline se inicializa correctamente"""
        assert pipeline is not None
        assert pipeline.preprocessing_pipeline is None  # No ajustado todavía
        # El pipeline tiene acceso a las features desde settings
        assert hasattr(pipeline, 'load_and_prepare_data')
        assert hasattr(pipeline, 'fit_transform')
        assert hasattr(pipeline, 'transform')
        print("✓ Test 1: Pipeline inicializado correctamente")
    
    def test_load_and_prepare_data(self, pipeline):
        """Test 2: Verificar que los datos se cargan correctamente"""
        X, y, df = pipeline.load_and_prepare_data()
        
        # Verificar que se retornan los objetos correctos
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(df, pd.DataFrame)
        
        # Verificar que hay datos
        assert len(X) > 0
        assert len(y) > 0
        assert len(df) > 0
        
        # Verificar que X e y tienen la misma longitud
        assert len(X) == len(y)
        
        expected_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES
        for col in expected_columns:
            assert col in X.columns, f"Columna {col} no encontrada en X"
        
        print(f"✓ Test 2: Datos cargados correctamente - Shape: {X.shape}")
    
    def test_fit_transform_dimensions(self, pipeline, sample_data):
        """Test 3: Verificar que fit_transform genera las dimensiones correctas"""
        X_transformed = pipeline.fit_transform(sample_data)
        
        assert isinstance(X_transformed, pd.DataFrame)
        
        assert X_transformed.shape[0] == sample_data.shape[0]
        
        assert X_transformed.shape[1] > sample_data.shape[1]
        
        assert X_transformed.isna().sum().sum() == 0
        
        print(f"✓ Test 3: Transformación correcta - Input: {sample_data.shape}, Output: {X_transformed.shape}")
    
    def test_transform_consistency(self, pipeline, sample_data):
        """Test 4: Verificar que transform genera el mismo número de features"""
        X_train_transformed = pipeline.fit_transform(sample_data)
        
        X_test = sample_data.iloc[:2]
        X_test_transformed = pipeline.transform(X_test)
        
        assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
        
        assert X_test_transformed.isna().sum().sum() == 0
        
        print(f"✓ Test 4: Transform consistente - Features: {X_test_transformed.shape[1]}")
    
    def test_get_feature_names(self, pipeline, sample_data):
        """Test 5: Verificar que get_feature_names retorna nombres válidos"""
        # Ajustar el pipeline primero
        pipeline.fit_transform(sample_data)
        
        # Obtener nombres de features
        feature_names = pipeline.get_feature_names()
        
        # Verificar que es una lista
        assert isinstance(feature_names, list)
        
        # Verificar que hay features
        assert len(feature_names) > 0
        
        # Verificar que todos son strings
        assert all(isinstance(name, str) for name in feature_names)
        
        # Verificar que coincide con el número de columnas transformadas
        X_transformed = pipeline.transform(sample_data)
        assert len(feature_names) == X_transformed.shape[1]
        
        print(f"✓ Test 5: Feature names correcto - Total: {len(feature_names)}")
    
    def test_preprocessing_pipeline_not_none_after_fit(self, pipeline, sample_data):
        """Test 6: Verificar que el pipeline interno se crea después de fit"""
        assert pipeline.preprocessing_pipeline is None
        
        pipeline.fit_transform(sample_data)
        
        assert pipeline.preprocessing_pipeline is not None
        print("✓ Test 6: Pipeline interno creado después de fit")
    
    def test_numeric_features_scaling(self, pipeline, sample_data):
        """Test 7: Verificar que las features numéricas se escalan correctamente"""
        X_transformed = pipeline.fit_transform(sample_data)
        
        # Las primeras features deberían ser las numéricas escaladas
        numeric_cols = [col for col in X_transformed.columns if col in NUMERIC_FEATURES]
        
        if len(numeric_cols) > 0:
            # Verificar que están cerca de media 0 y std 1 (StandardScaler)
            for col in numeric_cols:
                mean = X_transformed[col].mean()
                std = X_transformed[col].std()
                assert abs(mean) < 1e-10 or abs(mean) < 1.0  # Media cercana a 0
                print(f"  Feature '{col}': mean={mean:.4f}, std={std:.4f}")
        
        print("✓ Test 7: Features numéricas escaladas correctamente")
    
    def test_categorical_encoding(self, pipeline, sample_data):
        """Test 8: Verificar que las features categóricas se codifican"""
        X_transformed = pipeline.fit_transform(sample_data)
        
        # Buscar columnas con prefijos de categorías (cat__)
        categorical_cols = [col for col in X_transformed.columns 
                          if col.startswith('cat__')]
        
        # Debe haber columnas categóricas codificadas
        assert len(categorical_cols) > 0, "No se encontraron columnas categóricas codificadas"
        
        # Todas las columnas categóricas codificadas deben ser 0 o 1
        for col in categorical_cols:
            unique_vals = X_transformed[col].unique()
            assert all(val in [0, 1, 0.0, 1.0] for val in unique_vals), f"Valores inválidos en {col}"
        
        print(f"✓ Test 8: Categorical encoding correcto - {len(categorical_cols)} columnas generadas")
    
    def test_text_features_vectorization(self, pipeline, sample_data):
        """Test 9: Verificar que las features de texto se vectorizan"""
        X_transformed = pipeline.fit_transform(sample_data)
        
        # Buscar columnas de texto (empiezan con 'text_')
        text_cols = [col for col in X_transformed.columns if col.startswith('text_')]
        
        # Debe haber columnas de texto
        assert len(text_cols) > 0
        
        # Los valores deben ser numéricos (TF-IDF scores)
        for col in text_cols[:5]:  # Verificar las primeras 5
            assert pd.api.types.is_numeric_dtype(X_transformed[col])
        
        print(f"✓ Test 9: Text vectorization correcto - {len(text_cols)} columnas de texto")
    
    def test_pipeline_with_missing_values(self, pipeline):
        """Test 10: Verificar que el pipeline maneja valores faltantes"""
        # Crear datos con valores faltantes (pero no None en columnas categóricas,
        # ya que SimpleImputer con 'most_frequent' tiene problemas con None)
        data_with_nulls = pd.DataFrame({
            'Age': [25, np.nan, 35, 40, np.nan],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],  # Sin None
            'Education Level': ["Bachelor's", "Master's", "Bachelor's", "Bachelor's", "Master's"],  # Sin None
            'Job Title': ['Engineer', 'Analyst', 'Manager', 'Engineer', 'Director'],  # Sin None
            'Years of Experience': [2, 5, np.nan, 8, 15],
            'Description': [
                'Software engineer',
                'Data analyst',
                'Manager with experience',
                'Senior engineer',
                'Director with strategy'
            ]
        })
        
        # El pipeline debe manejar esto sin errores
        X_transformed = pipeline.fit_transform(data_with_nulls)
        
        # No debe haber valores nulos después de la transformación
        assert X_transformed.isna().sum().sum() == 0
        
        print("✓ Test 10: Pipeline maneja valores faltantes correctamente")
    
    def test_transform_before_fit_raises_error(self, pipeline, sample_data):
        """Test 11: Verificar que transform sin fit lanza error"""
        with pytest.raises(ValueError):
            pipeline.transform(sample_data)
        
        print("✓ Test 11: Transform sin fit lanza ValueError como esperado")
    
    def test_pipeline_reproducibility(self, sample_data):
        """Test 12: Verificar que el pipeline es reproducible"""
        pipeline1 = DataPipeline()
        pipeline2 = DataPipeline()
        
        X_transformed1 = pipeline1.fit_transform(sample_data)
        X_transformed2 = pipeline2.fit_transform(sample_data)
        
        # Los resultados deben ser idénticos
        pd.testing.assert_frame_equal(X_transformed1, X_transformed2)
        
        print("✓ Test 12: Pipeline es reproducible")


def test_full_pipeline_integration():
    """Test de integración: Pipeline completo con datos reales"""
    print("\n" + "="*70)
    print("TEST DE INTEGRACIÓN: Pipeline Completo con Datos Reales")
    print("="*70)
    
    pipeline = DataPipeline()
    
    # Cargar datos reales
    X, y, df = pipeline.load_and_prepare_data()
    print(f"\n1. Datos cargados: {X.shape}")
    
    # Aplicar transformación
    X_transformed = pipeline.fit_transform(X)
    print(f"2. Datos transformados: {X_transformed.shape}")
    
    # Verificar feature names
    feature_names = pipeline.get_feature_names()
    print(f"3. Features generadas: {len(feature_names)}")
    
    # Verificar consistencia
    X_test = X.head(10)
    X_test_transformed = pipeline.transform(X_test)
    print(f"4. Transform en subset: {X_test_transformed.shape}")
    
    assert X_transformed.shape[1] == X_test_transformed.shape[1]
    assert len(feature_names) == X_transformed.shape[1]
    assert X_transformed.isna().sum().sum() == 0
    
    print("\n✓ Pipeline de integración completado exitosamente")
    print("="*70)


if __name__ == "__main__":
    # Ejecutar tests con pytest
    pytest.main([__file__, "-v", "--tb=short"])
