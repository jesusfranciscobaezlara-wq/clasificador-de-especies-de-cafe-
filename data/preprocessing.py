# ============================================================
# data/preprocessing.py  —  Transformaciones sobre píxeles
# ============================================================
import cv2
import numpy as np
from config.settings import IMAGE_SIZE


def preprocess_image(img: np.ndarray) -> np.ndarray | None:
    """Recibe una imagen RGB uint8/float y devuelve float32 normalizada [0,1]."""
    if img is None:
        return None
    try:
        # Asegurar 3 canales
        if len(img.shape) == 2:
            img = np.stack((img,) * 3, axis=-1)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        img = cv2.resize(img, IMAGE_SIZE)

        # CLAHE en canal L del espacio LAB 
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)          
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)  

        # Reducción de ruido
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

        return img.astype(np.float32) / 255.0
    except Exception:
        return None