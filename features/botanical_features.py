# ============================================================
# features/botanical_features.py  —  Limbo, nervaduras y borde
# ============================================================
import cv2
import numpy as np

# 1 ondulacion + 1 rugosidad + 1 densidad_venas + 24 hist_angs + 1 compacidad
BOTANICAL_FEATURE_DIM = 28


def extract_botanical_features(
    img: np.ndarray,
    mask: np.ndarray,
    contour: np.ndarray,
) -> np.ndarray:
    """
    Extrae características botánicas avanzadas: ondulación del borde,
    densidad y orientación de nervaduras, y compacidad del limbo.

    Parámetros
    ----------
    img : np.ndarray
        Imagen RGB (uint8) del ROI donde está la hoja.
    mask : np.ndarray
        Máscara binaria de la hoja (mismo tamaño que img).
    contour : np.ndarray
        Contorno de la hoja (coordenadas dentro del ROI).

    Retorna
    -------
    np.ndarray
        Vector de características botánicas (float32).
    """
    feat = []
    
    # ------------------------------------------------------------------
    # 1. ONDULACIÓN DEL BORDE (liso vs ondulado)
    # ------------------------------------------------------------------
    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull_perimeter = cv2.arcLength(hull, True)
    ondulacion = perimeter / hull_perimeter if hull_perimeter > 0 else 1.0
    feat.append(ondulacion)   # ≈1 liso, >1.1 ondulado
    
    # Rugosidad adicional: desviación estándar de la distancia al centroide
    moments = cv2.moments(contour)
    if moments['m00'] != 0:
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        dists = [np.sqrt((p[0][0]-cx)**2 + (p[0][1]-cy)**2) for p in contour]
        if np.mean(dists) > 0:
            rugosidad = np.std(dists) / np.mean(dists)
        else:
            rugosidad = 0.0
    else:
        rugosidad = 0.0
    feat.append(rugosidad)
    
    # ------------------------------------------------------------------
    # 2. NERVADURAS (venas): densidad y orientación
    # ------------------------------------------------------------------
    edges_masked, ang_deg = compute_vein_geometry(img, mask)
    total_pixels = np.count_nonzero(mask)
    densidad_venas = np.count_nonzero(edges_masked) / total_pixels if total_pixels > 0 else 0.0
    feat.append(densidad_venas)

    # Histograma de orientaciones de los bordes (24 bins, 0-180°)
    hist_angs, _ = np.histogram(ang_deg[mask > 0], bins=24, range=(0, 180))
    hist_angs = hist_angs.astype(np.float32) / (hist_angs.sum() + 1e-6)
    feat.extend(hist_angs)   # 24 valores
    
    # ------------------------------------------------------------------
    # 3. CARACTERÍSTICAS DEL LIMBO (lámina)
    # ------------------------------------------------------------------
    # Compacidad = (perímetro^2) / área
    if total_pixels > 0 and perimeter > 0:
        compacidad = (perimeter * perimeter) / total_pixels
    else:
        compacidad = 0.0
    feat.append(compacidad)
    
    return np.array(feat, dtype=np.float32)


def compute_vein_geometry(img: np.ndarray, mask: np.ndarray) -> tuple:
    """
    Bordes tipo Canny dentro de la máscara de la hoja y su ángulo de
    orientación (0-180°) en cada píxel. Es la misma base geométrica que usan
    densidad_venas y el histograma de orientación en extract_botanical_features;
    se expone aparte para que la ventana de explicabilidad pueda dibujar
    exactamente lo que el modelo mide, sin recalcularlo con otra lógica.

    Retorna
    -------
    edges_masked : np.ndarray
        Máscara binaria de bordes (Canny) dentro de la hoja.
    ang_deg : np.ndarray
        Ángulo del gradiente en cada píxel, en grados (0-180).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    edges = cv2.Canny(gray_eq, 30, 100)
    edges_masked = cv2.bitwise_and(edges, edges, mask=mask)
    gx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=3)
    _, ang = cv2.cartToPolar(gx, gy)
    ang_deg = np.degrees(ang) % 180
    return edges_masked, ang_deg


def get_contour_and_mask(img_rgb: np.ndarray) -> tuple:
    """
    Helper para obtener máscara y contorno de la hoja a partir de una imagen
    RGB (sin calibración ArUco). Útil cuando no se dispone de un ROI.
    
    Parámetros
    ----------
    img_rgb : np.ndarray
        Imagen RGB (uint8) completa.
    
    Retorna
    -------
    mask : np.ndarray
        Máscara binaria de la hoja.
    contour : np.ndarray
        Contorno más grande (mayor área).
    """
    # Segmentación simple por color verde
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Operaciones morfológicas para limpiar
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos y quedarse con el de mayor área
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(mask), np.array([])
    contour = max(cnts, key=cv2.contourArea)
    return mask, contour