# ============================================================
# vision/leaf_detector.py  —  Contorno, color y tamaño de hoja
# ============================================================
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class LeafMeasurement:
    ancho: float = 0.0
    alto: float = 0.0
    angulo: float = 0.0
    area_cm2: float = 0.0
    color: str = "DESCONOCIDO"
    tamano: str = "N/H"
    # ── NUEVO: puntos reales para dibujar (coordenadas globales) ──
    pt_apice:   tuple | None = None   # extremo distal del limbo
    pt_base:    tuple | None = None   # extremo proximal del limbo
    pt_ancho_L: tuple | None = None   # borde izquierdo del ancho máximo
    pt_ancho_R: tuple | None = None   # borde derecho del ancho máximo


class LeafDetector:
    def detect(
        self,
        frame: np.ndarray,
        area_deteccion: tuple,
        pixeles_por_cm: float | None,
    ) -> tuple[np.ndarray | None, tuple | None, "LeafMeasurement"]:
        """
        Devuelve (contorno_completo, rect_info, medición).
        rect_info = (box_global, w, h, angle, center_global)
        """
        measurement = LeafMeasurement()
        if not area_deteccion:
            return None, None, measurement

        x1, y1, x2, y2 = area_deteccion
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, None, measurement

        mascara   = self._detectar_hoja_multicolor(roi)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return None, None, measurement

        # contorno completo de la hoja (incluye pecíolo)
        cnt = max(contornos, key=cv2.contourArea)
        measurement.color = self._detectar_color(roi, cnt)

        if pixeles_por_cm:
            measurement.area_cm2 = cv2.contourArea(cnt) / (pixeles_por_cm ** 2)
            measurement.tamano   = self._clasificar_tamano(measurement.area_cm2)

        # ── Separar pecíolo → limbo (puntos reales del contorno, sin hull) ──
        cnt_limbo = self._separar_peciolo(cnt)

        # ── Bounding box sobre el LIMBO filtrado → ajusta a la hoja en cualquier ángulo ──
        rect_info = self._compute_rect(cnt_limbo, x1, y1)
        if rect_info:
            _, w, h, angle, _ = rect_info
            measurement.angulo = angle

        if pixeles_por_cm:
            largo, ancho, pt_ap, pt_ba, pt_L, pt_R = self._medir_limbo(
                cnt_limbo, x1, y1, pixeles_por_cm
            )
            measurement.alto       = largo
            measurement.ancho      = ancho
            measurement.pt_apice   = pt_ap
            measurement.pt_base    = pt_ba
            measurement.pt_ancho_L = pt_L
            measurement.pt_ancho_R = pt_R

        return cnt, rect_info, measurement

    # ──────────────────────────────────────────────────────────────
    #  NUEVO: Separar pecíolo del limbo foliar
    # ──────────────────────────────────────────────────────────────
    def _separar_peciolo(self, cnt: np.ndarray, umbral_rel: float = 0.08) -> np.ndarray:
        """
        Detecta y elimina el pecíolo del contorno.
        
        Estrategia: proyecta el contorno sobre su eje principal (PCA) y
        divide en franjas transversales. El pecíolo es la región donde el
        ancho cae por debajo de `umbral_rel` × ancho_máximo. Se conserva
        sólo la parte del limbo y se retorna su casco convexo.
        
        umbral_rel: fracción del ancho máximo bajo la cual se considera pecíolo.
                    Aumentar si se corta demasiado; reducir si queda pecíolo.
        """
        pts = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts) < 15:
            return cnt

        mean, eigvec = cv2.PCACompute(pts, mean=None)
        eje      = eigvec[0]   # dirección del largo
        eje_perp = eigvec[1]   # dirección del ancho
        centrado    = pts - mean
        proj_mayor  = centrado @ eje
        proj_menor  = centrado @ eje_perp

        # dividir en franjas a lo largo del eje mayor
        n     = 40
        mn, mx = proj_mayor.min(), proj_mayor.max()
        paso  = (mx - mn) / n

        anchos = []
        for i in range(n):
            t0, t1 = mn + i * paso, mn + (i + 1) * paso
            mask = (proj_mayor >= t0) & (proj_mayor < t1)
            if mask.sum() < 2:
                anchos.append(0.0)
            else:
                anchos.append(float(proj_menor[mask].max() - proj_menor[mask].min()))

        anchos    = np.array(anchos, dtype=np.float32)
        ancho_max = anchos.max()
        if ancho_max == 0:
            return cnt

        umbral = umbral_rel * ancho_max
        franjas_limbo = np.where(anchos > umbral)[0]
        if len(franjas_limbo) == 0:
            return cnt

        # conservar sólo los puntos del contorno que pertenecen al limbo
        # SIN convex hull → preserva la forma real del limbo
        inicio_limbo = mn + franjas_limbo[0] * paso
        mask_limbo   = proj_mayor >= inicio_limbo
        if mask_limbo.sum() < 5:
            return cnt

        pts_limbo = pts[mask_limbo].astype(np.int32).reshape(-1, 1, 2)
        return pts_limbo   # contorno filtrado, sin hull

    # ──────────────────────────────────────────────────────────────
    #  NUEVO: Medir largo ápice→base y ancho máximo del limbo
    # ──────────────────────────────────────────────────────────────
    def _medir_limbo(
        self,
        cnt_limbo: np.ndarray,
        x1: int,
        y1: int,
        pixeles_por_cm: float,
    ) -> tuple:
        """
        Retorna (largo_cm, ancho_cm,
                 pt_apice_global, pt_base_global,
                 pt_ancho_L_global, pt_ancho_R_global).

        largo_cm : distancia ápice → base del limbo (sin pecíolo)
        ancho_cm : ancho máximo perpendicular al eje mayor
        """
        pts = cnt_limbo.reshape(-1, 2).astype(np.float32)
        if len(pts) < 5:
            return 0.0, 0.0, None, None, None, None

        mean, eigvec = cv2.PCACompute(pts, mean=None)
        eje      = eigvec[0]
        eje_perp = eigvec[1]
        centrado   = pts - mean
        proj_mayor = centrado @ eje
        proj_menor = centrado @ eje_perp

        # ── Largo: punto más distal (ápice) y más proximal (base) ──
        idx_apice = int(np.argmax(proj_mayor))
        idx_base  = int(np.argmin(proj_mayor))
        pt_apice  = pts[idx_apice]
        pt_base   = pts[idx_base]
        largo_px  = float(np.linalg.norm(pt_apice - pt_base))

        # ── Ancho máximo: barrer franjas perpendicularmente ──
        n = 60
        mn_t, mx_t = proj_mayor.min(), proj_mayor.max()
        paso = (mx_t - mn_t) / n
        ancho_max_px = 0.0
        pt_L = pt_R = None

        for i in range(n):
            t0, t1 = mn_t + i * paso, mn_t + (i + 1) * paso
            mask = (proj_mayor >= t0) & (proj_mayor < t1)
            if mask.sum() < 2:
                continue
            mn_m = proj_menor[mask].min()
            mx_m = proj_menor[mask].max()
            ancho_px = mx_m - mn_m
            if ancho_px > ancho_max_px:
                ancho_max_px = ancho_px
                tc   = (t0 + t1) / 2
                pt_L = mean + tc * eje + mn_m * eje_perp
                pt_R = mean + tc * eje + mx_m * eje_perp

        largo_cm = largo_px  / pixeles_por_cm
        ancho_cm = ancho_max_px / pixeles_por_cm

        # convertir a coordenadas globales del frame
        offset = np.array([x1, y1], dtype=np.float32)

        def to_global(p):
            if p is None:
                return None
            arr = (np.array(p).flatten() + offset).astype(int)
            return (int(arr[0]), int(arr[1]))

        return (
            largo_cm, ancho_cm,
            to_global(pt_apice), to_global(pt_base),
            to_global(pt_L), to_global(pt_R),
        )

    # ──────────────────────────────────────────────────────────────
    #  Bounding box del limbo (sin pecíolo)
    # ──────────────────────────────────────────────────────────────
    def _compute_rect(self, cnt, x1, y1) -> tuple | None:
        """
        minAreaRect sobre el contorno del LIMBO filtrado.
        Retorna (box_global, w, h, angle_deg, center_global).
        w = dimensión mayor (largo), h = dimensión menor (ancho).
        """
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect

        # Garantizar rw = dimensión mayor
        if rw < rh:
            rw, rh = rh, rw
            angle += 90

        # Normalizar ángulo a [0, 180): orientación del eje mayor (línea sin
        # sentido de dirección, por lo que el período natural es 180°, no 90°).
        angle = angle % 360
        if angle > 180:
            angle = 360 - angle

        box        = cv2.boxPoints(((cx, cy), (rw, rh), angle)).astype(np.int32)
        box_global = box + np.array([x1, y1])
        center_global = (int(x1 + cx), int(y1 + cy))
        return box_global, rw, rh, angle, center_global

    # ──────────────────────────────────────────────────────────────
    #  Segmentación de color de hoja  (sin cambios)
    # ──────────────────────────────────────────────────────────────
    def _detectar_hoja_multicolor(self, roi: np.ndarray) -> np.ndarray:
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([30, 30, 30]),  np.array([90, 255, 255]))
        mask |= cv2.inRange(hsv, np.array([0, 30, 30]),  np.array([20, 255, 150]))
        mask |= cv2.inRange(hsv, np.array([15, 40, 40]), np.array([35, 255, 255]))

        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        )
        mask |= cv2.adaptiveThreshold(
            cv2.GaussianBlur(gray, (5, 5), 0), 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2,
        )
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask)
        for cnt in contornos:
            if cv2.contourArea(cnt) > 500:
                cv2.drawContours(filled, [cnt], -1, 255, -1)
        return filled

    def _detectar_color(self, roi: np.ndarray, contorno: np.ndarray) -> str:
        try:
            mask = np.zeros(roi.shape[:2], np.uint8)
            cv2.drawContours(mask, [contorno], -1, 255, -1)
            pixels = roi[mask == 255].reshape(-1, 1, 3)
            h_mean = np.mean(cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV)[:, :, 0])
            if 30 <= h_mean <= 90:       return "VERDE"
            if 10 <= h_mean < 30:        return "AMARILLO"
            if h_mean < 10 or h_mean > 170: return "CAFE"
            return "DESCONOCIDO"
        except Exception:
            return "DESCONOCIDO"

    def _clasificar_tamano(self, area_cm2: float) -> str:
        if area_cm2 < 50:   return "CHICA"
        if area_cm2 < 150:  return "MEDIANA"
        return "GRANDE"