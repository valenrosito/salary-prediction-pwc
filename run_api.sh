#!/bin/bash

echo "🚀 Iniciando API de Predicción de Salarios"
echo "=========================================="
echo ""
echo "La API estará disponible en: http://localhost:8000"
echo "Documentación interactiva: http://localhost:8000/docs"
echo ""

cd "$(dirname "$0")"

.venv/bin/uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
