import joblib
from pathlib import Path
from typing import Any


def save_object(obj: Any, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)
    print(f"✓ Objeto guardado en: {filepath}")


def load_object(filepath: Path) -> Any:
    if not filepath.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {filepath}")
    
    obj = joblib.load(filepath)
    print(f"✓ Objeto cargado desde: {filepath}")
    return obj
