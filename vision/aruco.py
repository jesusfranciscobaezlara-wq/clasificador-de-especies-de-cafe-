# ============================================================
# vision/aruco.py  —  Detección y calibración con marcadores
# ============================================================
import math
import cv2
import numpy as np

if hasattr(cv2.aruco, 'Dictionary_get'):
    _get_dict = cv2.aruco.Dictionary_get
elif hasattr(cv2.aruco, 'getPredefinedDictionary'):
    _get_dict = cv2.aruco.getPredefinedDictionary
else:
    raise ImportError("No se encontró función para crear diccionario ArUco")

if hasattr(cv2.aruco, 'DetectorParameters_create'):
    _create_params = cv2.aruco.DetectorParameters_create
elif hasattr(cv2.aruco, 'DetectorParameters'):
    _create_params = cv2.aruco.DetectorParameters
else:
    _create_params = lambda: None

if hasattr(cv2.aruco, 'detectMarkers'):
    def _detect_markers(image, dictionary, parameters=None):
        return cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)
else:
    def _detect_markers(image, dictionary, parameters=None):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters or cv2.aruco.DetectorParameters())
        return detector.detectMarkers(image)


class ArucoCalibrator:
    def __init__(self):
        self.diccionarios = [
            _get_dict(cv2.aruco.DICT_5X5_50),
            _get_dict(cv2.aruco.DICT_4X4_50),
            _get_dict(cv2.aruco.DICT_6X6_50),
            _get_dict(cv2.aruco.DICT_7X7_50),
            _get_dict(cv2.aruco.DICT_5X5_100),
        ]
        self.params = _create_params()
        if self.params is not None:
            self.params.adaptiveThreshWinSizeMin = 3
            self.params.adaptiveThreshWinSizeMax = 23
            self.params.adaptiveThreshWinSizeStep = 10
            self.params.adaptiveThreshConstant = 7.0
            self.params.minMarkerPerimeterRate = 0.02
            self.params.maxMarkerPerimeterRate = 0.4
            self.params.polygonalApproxAccuracyRate = 0.05
            self.params.minCornerDistanceRate = 0.05
            if hasattr(self.params, 'cornerRefinementMethod'):
                self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.distancia_horizontal_marcadores_cm = 23.5
        self.distancia_vertical_marcadores_cm = 17.5

        self.color_hoja = "DESCONOCIDO"
        self.tamano_hoja = "N/H"
        self.area_real_hoja_cm2 = 0.0
        self.medicion = {'ancho': 0.0, 'alto': 0.0, 'angulo': 0.0}
        self.marcadores_posiciones: dict = {}
        self.pixeles_por_cm: float | None = None
        self.calibrado = False
        self.area_deteccion: tuple | None = None

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]
        max_dim = 800
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale)

        found = False
        for dicc in self.diccionarios:
            corners, ids, rejected = _detect_markers(gray, dicc, self.params)
            if ids is None:
                continue
            unique_ids = set(int(i) for i in ids.flatten())
            if len(unique_ids) != 4:
                # Cantidad de marcadores distinta a la esperada (jig de 4
                # esquinas): ambiguo, no confiar en esta detección.
                continue
            if scale != 1.0:
                corners = [(c / scale).astype(np.int32) for c in corners]
            self.marcadores_posiciones.clear()
            for i, id_val in enumerate(ids.flatten()):
                centro = tuple(np.mean(corners[i][0], axis=0).astype(int))
                self.marcadores_posiciones[int(id_val)] = centro
            found = True
            break

        if not found:
            self.marcadores_posiciones.clear()

        if len(self.marcadores_posiciones) >= 4:
            if not self.calibrado:
                self.calibrar()
            self.area_deteccion = self._calcular_area()
        else:
            # Sin los 4 marcadores visibles en el frame actual, el ROI
            # dejó de ser válido: no reutilizar uno obsoleto.
            self.area_deteccion = None

    def reset(self) -> None:
        self.marcadores_posiciones.clear()
        self.calibrado = False
        self.pixeles_por_cm = None
        self.area_deteccion = None

    # ------------------------------------------------------------------
    def calibrar(self) -> bool:
        ordenados = self._ordenar_marcadores()
        if not ordenados:
            return False
        sup_izq, sup_der, inf_der, inf_izq = ordenados
        d_h = math.dist(sup_izq, sup_der)
        d_v = math.dist(sup_izq, inf_izq)
        if d_h > 0 and d_v > 0:
            self.pixeles_por_cm = (
                d_h / self.distancia_horizontal_marcadores_cm +
                d_v / self.distancia_vertical_marcadores_cm
            ) / 2
            self.calibrado = True
            return True
        return False

    # ------------------------------------------------------------------
    def _ordenar_marcadores(self) -> list | None:
        if len(self.marcadores_posiciones) < 4:
            return None
        puntos = sorted(self.marcadores_posiciones.values(), key=lambda p: p[1])
        superiores = sorted(puntos[:2], key=lambda p: p[0])
        inferiores = sorted(puntos[2:], key=lambda p: p[0])
        return [superiores[0], superiores[1], inferiores[1], inferiores[0]]

    def _calcular_area(self) -> tuple | None:
        ordenados = self._ordenar_marcadores()
        if not ordenados:
            return None
        sup_izq, sup_der, inf_der, inf_izq = ordenados
        x1 = min(sup_izq[0], inf_izq[0]) + 10
        y1 = min(sup_izq[1], sup_der[1]) + 10
        x2 = max(sup_der[0], inf_der[0]) - 10
        y2 = max(inf_izq[1], inf_der[1]) - 10
        if x1 < x2 and y1 < y2:
            return (int(x1), int(y1), int(x2), int(y2))
        return None

    # ------------------------------------------------------------------
    def draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        display = frame.copy()
        for id_m, (x, y) in self.marcadores_posiciones.items():
            cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(display, str(id_m), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if self.area_deteccion:
            x1, y1, x2, y2 = self.area_deteccion
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return display