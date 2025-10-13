# 📊 Predicción de Salarios - PwC Challenge

Sistema de Machine Learning para predecir salarios basado en características demográficas y profesionales.

---

## 🎯 ¿Qué Hace?

Predice salarios en función de: edad, experiencia, género, nivel educativo, título del puesto y descripción del rol.

---

## 📦 ¿Qué Incluye?

1. **Pipeline de Datos:** Preprocesamiento automatizado (380 features desde 6 variables)
2. **Modelos ML:** Baseline, LightGBM, Random Forest, Red Neuronal (TensorFlow)
3. **API REST:** FastAPI con endpoints `/predict` y `/health`
4. **Interfaz Web:** Streamlit con predicción individual y por lotes (CSV)
5. **Testing:** 43+ tests con pytest

---

## 🛠️ Tecnologías

- **ML:** scikit-learn, LightGBM, TensorFlow
- **API:** FastAPI, Pydantic, Uvicorn
- **Web:** Streamlit
- **Data:** Pandas, NumPy
- **Gestor de Paquetes:** uv

---

## 📈 Rendimiento en validación

### Modelo Final: LightGBM 🏆

| Métrica | Valor |
|---------|-------|
| R² Score | **0.8925** |
| RMSE | **$16,056** |
| MAE | **$9,712** |
| Mejora vs Baseline | **67% ↓** |

### Comparación de Modelos

| Modelo | R² | RMSE | MAE |
|--------|-----|------|-----|
| Baseline | -0.0023 | $48,984 | $40,309 |
| Random Forest | 0.8732 | $17,421 | $10,532 |
| Neural Network | 0.8878 | $16,389 | $10,045 |
| **LightGBM** | **0.8925** | **$16,056** | **$9,712** |


**Features más importantes:** Years of Experience, Age, términos "senior"/"manager", Education Level

---

## � Dataset

- **Registros:** 375 personas
- **Features:** 6 variables (Age, Gender, Education Level, Job Title, Years of Experience, Description)
- **Target:** Salary
- **Calidad:** Menos del 2% valores nulos, sin duplicados

---

## 📁 Estructura del Proyecto

```
salary-prediction-pwc/
├── src/              # Código fuente (config, pipeline, models, api, app)
├── data/             # Datos (raw/ + ejemplo_prediccion.csv)
├── models/           # Modelos entrenados (.joblib)
├── notebooks/        # Análisis EDA y reporte final
├── tests/            # Tests automatizados (pytest)
└── docs/             # Documentación adicional
```

---

## ✅ Requisitos Previos

- **Python:** 3.9 o superior
- **UV:** Gestor de paquetes (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

---

## �🚀 Instalación

```bash
git clone https://github.com/valenrosito/salary-prediction-pwc.git
cd salary-prediction-pwc
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
.venv/bin/python tests/test_full_training.py  # Entrenar modelo
```

---

## 🌐 Ejecutar API

```bash
./run_api.sh
```

**Docs:** http://localhost:8000/docs

### Ejemplo

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 32,
    "gender": "Male",
    "education_level": "Bachelor'\''s",
    "job_title": "Software Engineer",
    "years_of_experience": 5,
    "description": "Python developer with ML"
  }'
```

**Respuesta:** `{"predicted_salary": 95234.56}`

---

## 🎨 Ejecutar Streamlit

```bash
./run_streamlit.sh
```

**URL:** http://localhost:8501

---

## 👤 Autor

**Valentin Rosito** - [@valenrosito](https://github.com/valenrosito)

---

