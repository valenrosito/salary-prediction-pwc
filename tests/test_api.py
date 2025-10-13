import requests
import json

API_URL = "http://localhost:8000"

def test_health():
    print("\n" + "="*60)
    print("TEST: Health Check")
    print("="*60)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    

def test_predict():
    print("\n" + "="*60)
    print("TEST: Predicción Individual")
    print("="*60)
    
    payload = {
        "age": 32,
        "gender": "Male",
        "education_level": "Bachelor's",
        "job_title": "Software Engineer",
        "years_of_experience": 5,
        "description": "Experienced software engineer with strong background in Python and machine learning"
    }
    
    response = requests.post(f"{API_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Request: {json.dumps(payload, indent=2)}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_multiple_predictions():
    print("\n" + "="*60)
    print("TEST: Múltiples Predicciones")
    print("="*60)
    
    test_cases = [
        {
            "age": 28,
            "gender": "Female",
            "education_level": "Master's",
            "job_title": "Data Analyst",
            "years_of_experience": 3,
            "description": "Data analyst with strong analytical skills"
        },
        {
            "age": 45,
            "gender": "Male",
            "education_level": "PhD",
            "job_title": "Senior Data Scientist",
            "years_of_experience": 15,
            "description": "Senior data scientist with extensive ML experience"
        },
        {
            "age": 25,
            "gender": "Female",
            "education_level": "Bachelor's",
            "job_title": "Junior Developer",
            "years_of_experience": 2,
            "description": "Entry-level developer with JavaScript knowledge"
        }
    ]
    
    for i, payload in enumerate(test_cases, 1):
        print(f"\nCaso {i}:")
        response = requests.post(f"{API_URL}/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"  Perfil: {payload['job_title']} - {payload['education_level']}")
            print(f"  Experiencia: {payload['years_of_experience']} años")
            print(f"  ✓ Salario Predicho: ${result['predicted_salary']:,.2f}")
        else:
            print(f"  ✗ Error: {response.status_code}")


if __name__ == "__main__":
    print("="*60)
    print("PRUEBAS DE LA API DE PREDICCIÓN DE SALARIOS")
    print("="*60)
    print("\nAsegúrate de que la API esté ejecutándose en http://localhost:8000")
    print("Ejecuta: ./run_api.sh en otra terminal")
    
    try:
        test_health()
        test_predict()
        test_multiple_predictions()
        
        print("\n" + "="*60)
        print("✓ TODAS LAS PRUEBAS COMPLETADAS")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n✗ Error: No se pudo conectar a la API")
        print("Asegúrate de que la API esté ejecutándose con: ./run_api.sh")
    except Exception as e:
        print(f"\n✗ Error inesperado: {e}")
