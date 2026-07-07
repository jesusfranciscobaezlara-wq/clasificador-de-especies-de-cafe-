# ============================================================
# api/routes.py  —  Endpoints FastAPI
# ============================================================
import os
import sys
import tempfile
from typing import Optional

# Ver la nota equivalente en main.py: evita que un print() con emoji crashee
# el proceso si la salida del servidor se redirige a un log sin UTF-8.
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from config.settings import PLANT_DESCRIPTIONS
from services.classifier import CoffeePlantClassifier

# Mismo criterio que main.py: datasets/ junto a la raíz del proyecto por
# defecto, salvo que COFFEE_BASE_DIR indique otra ubicación.
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.environ.get("COFFEE_BASE_DIR", os.path.join(_PROJECT_DIR, "datasets"))

app = FastAPI(
    title="Coffee Plant Classifier API",
    description="Clasificación de plantas de café con red neuronal de fusión (CNN + características botánicas)",
)

# ── Singleton a nivel de módulo ──
_classifier: CoffeePlantClassifier | None = None


@app.on_event("startup")
async def startup():
    global _classifier
    clf = CoffeePlantClassifier(
        train_folder  = os.path.join(BASE_DIR, "Training"),
        test_folder   = os.path.join(BASE_DIR, "Test"),
        results_folder= os.path.join(BASE_DIR, "Results"),
    )
    clf.load_models()
    _classifier = clf


@app.post("/predict/")
async def predict_image(
    file: UploadFile = File(...),
    true_label: Optional[str] = None,
):
    if _classifier is None or _classifier.trainer.model is None:
        return JSONResponse(
            status_code=503,
            content={"message": "Modelo no disponible. Entrene el modelo primero."},
        )

    suffix = os.path.splitext(os.path.basename(file.filename or ""))[1]
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix="coffee_upload_")
    try:
        with os.fdopen(fd, "wb") as buf:
            buf.write(await file.read())

        pred, proba = _classifier.predict_single_image(
            temp_path, true_label=true_label, show_image=False
        )
        if pred is None:
            return JSONResponse(status_code=400,
                                content={"message": "Error al procesar la imagen"})

        return {
            "prediction": pred,
            "display_name": _classifier.get_display_name(pred),
            "confidence": float(proba),
            "plant_info": PLANT_DESCRIPTIONS.get(pred, {}),
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error: {e}"})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_api(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run(app, host=host, port=port)