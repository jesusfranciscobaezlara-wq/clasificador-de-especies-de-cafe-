# ============================================================
# features/extractor.py  --  HOG + LBP + GLCM + color stats + botánicas
# ============================================================
import cv2
import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops, local_binary_pattern

from config.settings import IMAGE_SIZE
from features.botanical_features import extract_botanical_features

def extract_full_features(img: np.ndarray, mask: np.ndarray, contour: np.ndarray,
                          base_point: tuple = None) -> np.ndarray:
    """Extrae características visuales + botánicas."""
    visual = extract_advanced_features(img)
    # CORRECCIÓN: orden correcto: img, mask, contour, base_point
    botanical = extract_botanical_features(img, mask, contour, base_point)
    return np.concatenate([visual, botanical])

def extract_advanced_features(img: np.ndarray) -> np.ndarray:
    """Recibe imagen float32 RGB [0,1] de tamaño IMAGE_SIZE.
       Devuelve vector de features visuales: HOG + LBP + color + GLCM.
    """
    img_float = img.astype(np.float32)
    gray = cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY)

    # --- HOG ---
    cell_size = (IMAGE_SIZE[0] // 20, IMAGE_SIZE[1] // 20)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=cell_size,
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    # --- LBP ---
    radius, n_points = 2, 16
    gray_uint8 = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_uint8, n_points, radius, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(np.float32) / (lbp_hist.sum() + 1e-6)

    # --- Estadísticas de color ---
    hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2LAB)
    color_features = np.array([
        np.mean(img_float[:, :, 1]), np.std(img_float[:, :, 1]),   # verde
        np.mean(hsv[:, :, 0]),        np.std(hsv[:, :, 0]),          # hue
        np.mean(lab[:, :, 1]),        np.std(lab[:, :, 1]),           # a*
    ])

    # --- GLCM ---
    glcm = graycomatrix(gray_uint8, distances=[1], angles=[0], symmetric=True, normed=True)
    texture_features = np.array([
        graycoprops(glcm, "contrast")[0, 0],
        graycoprops(glcm, "dissimilarity")[0, 0],
        graycoprops(glcm, "homogeneity")[0, 0],
        graycoprops(glcm, "energy")[0, 0],
        graycoprops(glcm, "correlation")[0, 0],
    ])

    return np.concatenate([hog_features, lbp_hist, color_features, texture_features])

# extract_features_with_measurements se mantiene igual


def extract_features_with_measurements(
    img: np.ndarray,
    leaf_measurements: np.ndarray | None = None
) -> np.ndarray:
    """
    Extrae características visuales y, si se proporcionan, las concatena
    con medidas de la hoja (ancho, alto, área, ángulo).
    
    Parámetros
    ----------
    img : np.ndarray
        Imagen RGB float32 [0,1] de tamaño IMAGE_SIZE.
    leaf_measurements : np.ndarray o None
        Vector de 4 valores: [ancho_cm, alto_cm, area_cm2, angulo_grados].
        Si es None, se rellenan con ceros.
    
    Retorna
    -------
    np.ndarray
        Vector completo de características.
    """
    visual_feat = extract_advanced_features(img)
    if leaf_measurements is None:
        # Rellenar con ceros para mantener dimensión fija
        leaf_feat = np.zeros(4, dtype=np.float32)
    else:
        leaf_feat = np.array(leaf_measurements, dtype=np.float32).flatten()
        if len(leaf_feat) != 4:
            # Ajustar a 4 elementos por seguridad
            padded = np.zeros(4, dtype=np.float32)
            padded[:min(len(leaf_feat), 4)] = leaf_feat[:4]
            leaf_feat = padded
    return np.concatenate([visual_feat, leaf_feat])