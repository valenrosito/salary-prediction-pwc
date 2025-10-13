import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.main import app


class TestAPIStructure:
    """Tests de estructura básica de la API"""
    
    @pytest.fixture
    def client(self):
        """Fixture que crea un TestClient"""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test 1: Verificar que el endpoint raíz funciona"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        
        print("✓ Test 1: Endpoint raíz funciona correctamente")
    
    def test_health_endpoint_exists(self, client):
        """Test 2: Verificar que el endpoint /health existe"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verificar estructura
        assert "status" in data
        assert "model_loaded" in data
        
        print(f"✓ Test 2: Health endpoint responde - Status: {data['status']}")
    
    def test_openapi_docs_available(self, client):
        """Test 3: Verificar que la documentación está disponible"""
        # OpenAPI schema
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "paths" in schema
        assert "/predict" in schema["paths"]
        assert "/health" in schema["paths"]
        
        print("✓ Test 3: Documentación OpenAPI disponible")
    
    def test_invalid_endpoint_404(self, client):
        """Test 4: Verificar que endpoints inválidos retornan 404"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        print("✓ Test 4: Endpoint inválido retorna 404")


class TestAPIValidation:
    """Tests de validación de datos de la API"""
    
    @pytest.fixture
    def client(self):
        """Fixture que crea un TestClient"""
        return TestClient(app)
    
    @pytest.fixture
    def valid_payload(self):
        """Fixture con un payload válido"""
        return {
            "age": 32,
            "gender": "Male",
            "education_level": "Bachelor's",
            "job_title": "Software Engineer",
            "years_of_experience": 5,
            "description": "Experienced software engineer"
        }
    
    def test_predict_missing_required_field(self, client):
        """Test 5: Verificar error 422 cuando falta un campo requerido"""
        invalid_payload = {
            "age": 32,
            "gender": "Male",
            # Falta education_level
            "job_title": "Software Engineer",
            "years_of_experience": 5,
            "description": "Some description"
        }
        
        response = client.post("/predict", json=invalid_payload)
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        
        print("✓ Test 5: API retorna 422 cuando falta campo requerido")
    
    def test_predict_invalid_age_type(self, client, valid_payload):
        """Test 6: Verificar error 422 con tipo de dato incorrecto"""
        invalid_payload = valid_payload.copy()
        invalid_payload["age"] = "thirty two"  # String en vez de int
        
        response = client.post("/predict", json=invalid_payload)
        
        assert response.status_code == 422
        
        print("✓ Test 6: API retorna 422 con tipo de dato incorrecto")
    
    def test_predict_invalid_experience_type(self, client, valid_payload):
        """Test 7: Verificar error 422 con tipo de experiencia incorrecto"""
        invalid_payload = valid_payload.copy()
        invalid_payload["years_of_experience"] = "five years"
        
        response = client.post("/predict", json=invalid_payload)
        
        assert response.status_code == 422
        
        print("✓ Test 7: API valida tipo de years_of_experience")
    
    def test_predict_missing_all_fields(self, client):
        """Test 8: Verificar error 422 con payload vacío"""
        response = client.post("/predict", json={})
        
        assert response.status_code == 422
        data = response.json()
        
        # Debe haber múltiples errores de validación
        assert "detail" in data
        assert len(data["detail"]) > 0
        
        print(f"✓ Test 8: API valida payload vacío - {len(data['detail'])} errores")
    
    def test_predict_extra_fields_ignored(self, client, valid_payload):
        """Test 9: Verificar que campos extra no causan error"""
        payload_with_extra = valid_payload.copy()
        payload_with_extra["extra_field"] = "should be ignored"
        payload_with_extra["another_extra"] = 123
        
        response = client.post("/predict", json=payload_with_extra)
        
        # Puede ser 200 (si modelo cargado) o 503 (si no está cargado)
        # Lo importante es que no sea 422 (error de validación)
        assert response.status_code in [200, 503]
        
        print("✓ Test 9: API ignora campos extra correctamente")
    
    def test_predict_with_null_values(self, client, valid_payload):
        """Test 10: Verificar que valores null se rechazan"""
        invalid_payload = valid_payload.copy()
        invalid_payload["age"] = None
        
        response = client.post("/predict", json=invalid_payload)
        
        assert response.status_code == 422
        
        print("✓ Test 10: API rechaza valores null en campos requeridos")
    
    def test_predict_endpoint_accepts_valid_schema(self, client, valid_payload):
        """Test 11: Verificar que el schema válido pasa la validación"""
        response = client.post("/predict", json=valid_payload)
        
        # Si el modelo no está cargado, debe retornar 503, no 422
        # Si está cargado, debe retornar 200
        assert response.status_code in [200, 503], \
            f"Schema válido no debería causar error 422, obtuvo {response.status_code}"
        
        if response.status_code == 503:
            print("✓ Test 11: Schema válido - Modelo no cargado (503)")
        else:
            print("✓ Test 11: Schema válido - Predicción exitosa (200)")


class TestAPIErrorHandling:
    """Tests de manejo de errores de la API"""
    
    @pytest.fixture
    def client(self):
        """Fixture que crea un TestClient"""
        return TestClient(app)
    
    def test_predict_method_not_allowed(self, client):
        """Test 12: Verificar que métodos incorrectos retornan 405"""
        response = client.get("/predict")
        assert response.status_code == 405
        
        print("✓ Test 12: GET /predict retorna 405 (Method Not Allowed)")
    
    def test_health_accepts_get_only(self, client):
        """Test 13: Verificar que /health solo acepta GET"""
        response = client.post("/health")
        assert response.status_code == 405
        
        print("✓ Test 13: /health rechaza POST correctamente")
    
    def test_content_type_json_required(self, client):
        """Test 14: Verificar que se requiere JSON en /predict"""
        # Enviar datos como form data en vez de JSON
        response = client.post(
            "/predict",
            data="age=30&gender=Male",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert response.status_code == 422
        
        print("✓ Test 14: API requiere Content-Type application/json")


def test_integration_api_flow():
    """Test de integración: Flujo completo de la API"""
    print("\n" + "="*70)
    print("TEST DE INTEGRACIÓN: Flujo Completo de API")
    print("="*70)
    
    client = TestClient(app)
    
    # 1. Verificar root
    response = client.get("/")
    assert response.status_code == 200
    print("✓ 1. Root endpoint funcional")
    
    # 2. Verificar health
    response = client.get("/health")
    assert response.status_code == 200
    health_data = response.json()
    model_loaded = health_data.get("model_loaded", False)
    print(f"✓ 2. Health check - Modelo cargado: {model_loaded}")
    
    # 3. Verificar documentación
    response = client.get("/openapi.json")
    assert response.status_code == 200
    print("✓ 3. Documentación OpenAPI disponible")
    
    # 4. Verificar validación de esquema
    invalid_payload = {"age": "invalid"}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
    print("✓ 4. Validación de esquema funcional")
    
    # 5. Intentar predicción válida
    valid_payload = {
        "age": 30,
        "gender": "Male",
        "education_level": "Bachelor's",
        "job_title": "Engineer",
        "years_of_experience": 5,
        "description": "Software engineer"
    }
    response = client.post("/predict", json=valid_payload)
    
    if response.status_code == 200:
        print("✓ 5. Predicción exitosa (modelo cargado)")
        data = response.json()
        assert "predicted_salary" in data
        assert isinstance(data["predicted_salary"], (int, float))
        print(f"   Salario predicho: ${data['predicted_salary']:,.2f}")
    elif response.status_code == 503:
        print("⚠ 5. Predicción no disponible (modelo no cargado)")
        print("   Ejecuta tests/test_full_training.py para entrenar el modelo")
    
    print("="*70)
    print("✓ Flujo de API verificado correctamente")
    print("="*70)


if __name__ == "__main__":
    # Ejecutar tests con pytest
    pytest.main([__file__, "-v", "--tb=short"])
