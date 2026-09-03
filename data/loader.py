# ============================================================
# data/loader.py  —  Lectura de imágenes desde disco
# ============================================================
import os
import cv2
import numpy as np
from tqdm import tqdm

from config.settings import IMAGE_EXTENSIONS, PLANT_DESCRIPTIONS, get_display_name, CNN_IMAGE_SIZE

def load_single_image(img_path: str, target_size=CNN_IMAGE_SIZE) -> np.ndarray | None:
    """
    Carga una imagen, la redimensiona a target_size y la devuelve como float32
    en el rango [0,1] en formato RGB.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        # Convertir a RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Redimensionar
        img = cv2.resize(img, target_size)
        # Normalizar a [0,1]
        img = img.astype(np.float32) / 255.0
        return img
    except Exception:
        return None

def load_images_from_folder(folder: str, target_size=CNN_IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """
    Recorre subcarpetas cuyo nombre coincida con PLANT_DESCRIPTIONS
    (excepto 'mifoto') y devuelve (images, labels).
    """
    images, labels = [], []
    valid_categories = set(PLANT_DESCRIPTIONS.keys()) - {"mifoto"}

    for species in os.listdir(folder):
        plant_path = os.path.join(folder, species)
        if not os.path.isdir(plant_path) or species not in valid_categories:
            continue

        img_paths = [
            os.path.join(plant_path, f)
            for f in os.listdir(plant_path)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

        print(f"\nProcesando {get_display_name(species)}:")
        plant_images = []
        for path in tqdm(img_paths, desc=f"  {get_display_name(species)}"):
            img = load_single_image(path, target_size)
            if img is not None:
                plant_images.append(img)

        if len(plant_images) < 5:
            print(f"  ⚠ Clase '{get_display_name(species)}' tiene pocas muestras ({len(plant_images)})")

        images.extend(plant_images)
        labels.extend([species] * len(plant_images))
        print(f"  ✅ {len(plant_images)} imágenes válidas cargadas")

    return np.array(images), np.array(labels)