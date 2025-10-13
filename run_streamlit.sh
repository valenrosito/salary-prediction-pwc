#!/bin/bash

echo "🎨 Iniciando Interfaz Gráfica de Predicción de Salarios"
echo "========================================================"
echo ""
echo "La aplicación estará disponible en: http://localhost:8501"
echo ""

cd "$(dirname "$0")"

.venv/bin/streamlit run src/app.py
