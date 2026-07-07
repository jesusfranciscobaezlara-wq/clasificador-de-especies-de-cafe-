# ============================================================
# vision/measurement.py  --  Cálculo de medidas de hoja (ArUco + JSON)
# Compartido entre main.py (consola) y vision/gui_app.py (GUI).
# ============================================================
import os
import json

import cv2

from vision.aruco import ArucoCalibrator
from vision.leaf_detector import LeafDetector


def load_leaf_json(image_path: str) -> dict:
    """Carga el archivo .json de mediciones si existe."""
    base = os.path.splitext(image_path)[0]
    json_path = base + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    return {}


def calcular_medidas_imagen(img_path: str) -> str:
    """Calcula las medidas de la hoja usando ArUco (o JSON) para mostrarlas al usuario."""
    img = cv2.imread(img_path)
    if img is None:
        return "Medidas: no disponibles"

    aruco = ArucoCalibrator()
    aruco.process_frame(img)
    if not aruco.calibrado or not aruco.area_deteccion:
        measures = load_leaf_json(img_path)
        if measures:
            return (f"Ancho: {measures.get('ancho_cm', '?')} cm  |  "
                    f"Alto: {measures.get('alto_cm', '?')} cm  |  "
                    f"Área: {measures.get('area_cm2', '?')} cm²  |  "
                    f"Ángulo: {measures.get('angulo_grados', '?')}°")
        return "Medidas: sin calibración (no se detectaron marcadores)"

    leaf_det = LeafDetector()
    cnt, rect_info, measurement = leaf_det.detect(
        img, aruco.area_deteccion, aruco.pixeles_por_cm
    )
    if measurement.alto > 0:
        return (f"Ancho: {measurement.ancho:.1f} cm  |  "
                f"Alto: {measurement.alto:.1f} cm  |  "
                f"Área: {measurement.area_cm2:.0f} cm²  |  "
                f"Ángulo: {measurement.angulo:.0f}°")
    return "Medidas: hoja no detectada"
