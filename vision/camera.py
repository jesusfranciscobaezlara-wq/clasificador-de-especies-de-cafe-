# ============================================================
# vision/camera.py  —  Loop de cámara con matplotlib
# ============================================================
import os
import time
import json
import math

import cv2
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

from vision.aruco import ArucoCalibrator
from vision.leaf_detector import LeafDetector, LeafMeasurement
from config.settings import IMAGE_EXTENSIONS

# Mismo lenguaje visual que vision/explain_window.py ("Predecir una imagen
# sin guardar"), para que los dos modos de Modo Prueba se sientan parte del
# mismo sistema.
_BG = "#0f0f23"
_PANEL = "#151530"
_TEAL = "#5DCAA5"
_GRAY_FILL = "#1a1a3e"
_GRAY_BORDER = "#444461"
_DIM_ALPHA = 0.5
_HEAT_CMAP = LinearSegmentedColormap.from_list("heat", ["#000000", "#ffcc33", "#ff3333"])

_COLOR_DISPLAY = {"VERDE": "Verde", "AMARILLO": "Amarillo", "CAFE": "Café", "DESCONOCIDO": "Desconocido"}
_TAMANO_DISPLAY = {"CHICA": "Chica", "MEDIANA": "Mediana", "GRANDE": "Grande", "N/H": "N/H"}


class CameraSession:
    """Gestiona el ciclo de vida de la cámara y delega la lógica a sus módulos."""

    def __init__(self, classifier):
        self.classifier    = classifier
        self.aruco         = ArucoCalibrator()
        self.leaf_det      = LeafDetector()

        self.running       = False
        self.cap           = None
        self.current_frame = None

        self.last_pred      = "Analizando..."
        self.last_conf      = 0.0
        self.last_pred_key  = None
        # Resultado de explicabilidad (Grad-CAM + probabilidades por clase),
        # más costoso que una predicción simple: se recalcula cada varios
        # frames, no en cada uno (ver _update_frame).
        self.last_cam        = None
        self.last_probs      = None
        self.last_pred_idx   = None
        self.last_names      = None
        self.last_class_keys = None
        self._thumb_cache: dict[str, np.ndarray] = {}

        self.measurement    = LeafMeasurement()
        self.cnt            = None
        self.rect_info      = None

        # ── Resiliencia ante fallos de la cámara (ver _try_recover) ──
        # El backend MSMF de Windows a veces deja de entregar frames a
        # medio uso (error -1072873822 / 0xC00D3EA2, muy reportado con
        # OpenCV en Windows) sin que la app se entere; antes, un ret=False
        # se ignoraba en silencio para siempre y la ventana quedaba
        # congelada sin avisar. Ahora se cuenta, se avisa en pantalla y se
        # reintenta reabrir la cámara; si tras varios intentos sigue sin
        # responder, se cierra la sesión en vez de quedar "trabada".
        self._camera_id           = 0
        self._consecutive_fails   = 0
        self._reconnect_attempts  = 0
        self._FAIL_THRESHOLD      = 15   # ~0.5s a 30fps antes de considerarlo un problema
        self._RECONNECT_COOLDOWN  = 30   # frames entre cada intento de reconexión
        self._MAX_RECONNECT_ATTEMPTS = 10

        # ── Suavizado temporal (EMA) para estabilizar mediciones ──
        self._ema_alto      = 0.0
        self._ema_ancho     = 0.0
        self._ema_alpha     = 0.15   # 0.0=congelado  1.0=sin suavizado
        # Suavizado de puntos de dibujo
        self._ema_apice     = None
        self._ema_base      = None
        self._ema_ancho_L   = None
        self._ema_ancho_R   = None

    # ------------------------------------------------------------------
    def run(self, camera_id: int = 0) -> None:
        self._camera_id = camera_id
        # El backend por defecto en Windows (MSMF) es el que produce los
        # cuelgues de frame reportados; DirectShow es más estable para
        # webcams. De paso, se silencian los WARN internos de OpenCV sobre
        # frames perdidos (ya los manejamos nosotros mismos abajo) sin
        # ocultar errores reales.
        cv2.setLogLevel(2)
        if not self._open_capture(camera_id):
            print(f"❌ Cámara {camera_id} no disponible")
            return
        self.running = True

        plt.ion()
        fig = plt.figure(figsize=(16, 7.2), facecolor=_BG)
        fig.canvas.mpl_connect("key_press_event", self._on_key)

        gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.10,
                              left=0.03, right=0.98, top=0.90, bottom=0.08)
        gs_right = gs[1].subgridspec(2, 1, height_ratios=[0.35, 1], hspace=0.35)
        ax_video   = fig.add_subplot(gs[0])
        ax_meas    = fig.add_subplot(gs_right[0])
        ax_species = fig.add_subplot(gs_right[1])

        ani = animation.FuncAnimation(   # noqa: F841
            fig, self._update_frame,
            fargs=(ax_video, ax_meas, ax_species),
            interval=33, cache_frame_data=False,
        )
        self._print_instructions(camera_id)
        plt.show(block=True)

        self.cap.release()
        plt.close()
        # Sin esto, plt.ion() sigue activo para el resto del programa: los
        # plt.show() de otras ventanas (evaluación de imágenes) dejan de
        # bloquear y se abren y cierran en el mismo instante.
        plt.ioff()
        print("\n✅ Modo cámara finalizado")

    # ------------------------------------------------------------------
    def _open_capture(self, camera_id: int) -> bool:
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        self.cap = cap
        if not cap.isOpened():
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return True

    # ------------------------------------------------------------------
    def _update_frame(self, frame_num, ax_video, ax_meas, ax_species):
        if not self.running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            self._consecutive_fails += 1
            if self._consecutive_fails >= self._FAIL_THRESHOLD:
                self._try_recover(ax_video)
            return
        self._consecutive_fails = 0

        self.current_frame = frame.copy()

        if frame_num % 10 == 0:
            data = self.classifier.analyze_frame(frame)
            if data is not None:
                self.last_cam        = data["cam"]
                self.last_probs      = data["probs"]
                self.last_pred_idx   = data["pred_idx"]
                self.last_names      = data["names"]
                self.last_class_keys = data["class_keys"]
                self.last_pred       = self.classifier.get_display_name(data["pred"])
                self.last_conf       = data["conf"]
                self.last_pred_key   = data["pred"]

        self.aruco.process_frame(frame)
        if self.aruco.calibrado and self.aruco.area_deteccion:
            self.cnt, self.rect_info, self.measurement = self.leaf_det.detect(
                frame, self.aruco.area_deteccion, self.aruco.pixeles_por_cm
            )
            self._aplicar_ema()

        display   = self.aruco.draw_overlay(frame)
        display   = self._draw_leaf_overlay(display)
        frame_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        ax_video.clear()
        ax_video.imshow(frame_rgb)
        if self.last_cam is not None:
            # Grad-CAM: qué zonas de la última imagen analizada pesaron más
            # en la predicción, superpuesto sobre el frame actual.
            cam_resized = cv2.resize(self.last_cam, (frame_rgb.shape[1], frame_rgb.shape[0]))
            ax_video.imshow(cam_resized, cmap=_HEAT_CMAP, alpha=0.35, vmin=0, vmax=1)
        ax_video.set_title("Cámara en Tiempo Real", color="white",
                           fontsize=14, fontweight="bold", pad=10)
        ax_video.axis("off")
        for spine in ax_video.spines.values():
            spine.set_visible(True)
            spine.set_color(_TEAL)
            spine.set_linewidth(2)
        ax_video.text(0.5, -0.04, "Q: Salir     ESPACIO: Capturar     R: Reiniciar calibración",
                      transform=ax_video.transAxes, ha="center", color="#9a9ab0", fontsize=10)

        self._draw_measurement_panel(ax_meas)
        self._draw_species_panel(ax_species)

    # ------------------------------------------------------------------
    def _try_recover(self, ax_video):
        """Se llama tras varios frames seguidos sin poder leer de la
        cámara. Avisa en pantalla y, cada _RECONNECT_COOLDOWN frames,
        intenta reabrirla. Si se supera _MAX_RECONNECT_ATTEMPTS sin éxito,
        cierra la sesión con un mensaje claro en vez de quedar "trabada"
        reintentando en silencio para siempre."""
        if self._consecutive_fails % self._RECONNECT_COOLDOWN != 0:
            self._draw_reconnecting_message(ax_video)
            return

        self._reconnect_attempts += 1
        if self._reconnect_attempts > self._MAX_RECONNECT_ATTEMPTS:
            print("\n❌ Se perdió la señal de la cámara y no se pudo recuperar. Cerrando modo cámara.")
            self.running = False
            plt.close()
            return

        print(f"\n⚠️ Señal de cámara perdida, reintentando ({self._reconnect_attempts}/{self._MAX_RECONNECT_ATTEMPTS})...")
        if self.cap is not None:
            self.cap.release()
        self._open_capture(self._camera_id)
        self._draw_reconnecting_message(ax_video)

    def _draw_reconnecting_message(self, ax_video):
        ax_video.clear()
        ax_video.set_facecolor(_BG)
        ax_video.axis("off")
        ax_video.set_title("Cámara en Tiempo Real", color="white",
                           fontsize=14, fontweight="bold", pad=10)
        ax_video.text(0.5, 0.5, "⚠ Señal de cámara perdida\nReintentando...",
                      transform=ax_video.transAxes, ha="center", va="center",
                      color="#ffcc66", fontsize=15, fontweight="bold")

    # ------------------------------------------------------------------
    def _aplicar_ema(self):
        """Suaviza mediciones y puntos de dibujo con media móvil exponencial."""
        m = self.measurement
        a = self._ema_alpha

        # Escalares: largo y ancho
        if m.alto > 0:
            if self._ema_alto == 0.0:
                self._ema_alto  = m.alto
                self._ema_ancho = m.ancho
            else:
                self._ema_alto  = a * m.alto  + (1 - a) * self._ema_alto
                self._ema_ancho = a * m.ancho + (1 - a) * self._ema_ancho
            m.alto  = round(self._ema_alto,  1)
            m.ancho = round(self._ema_ancho, 1)

        # Puntos de dibujo: suavizar coordenadas x, y por separado
        def suavizar_pt(nuevo, anterior):
            if nuevo is None:
                return anterior
            if anterior is None:
                return nuevo
            x = int(a * nuevo[0] + (1 - a) * anterior[0])
            y = int(a * nuevo[1] + (1 - a) * anterior[1])
            return (x, y)

        self._ema_apice   = suavizar_pt(m.pt_apice,   self._ema_apice)
        self._ema_base    = suavizar_pt(m.pt_base,    self._ema_base)
        self._ema_ancho_L = suavizar_pt(m.pt_ancho_L, self._ema_ancho_L)
        self._ema_ancho_R = suavizar_pt(m.pt_ancho_R, self._ema_ancho_R)

        m.pt_apice   = self._ema_apice
        m.pt_base    = self._ema_base
        m.pt_ancho_L = self._ema_ancho_L
        m.pt_ancho_R = self._ema_ancho_R

    # ------------------------------------------------------------------
    def _draw_leaf_overlay(self, frame):
        display = frame.copy()

        if self.rect_info and self.aruco.area_deteccion:
            box, w, h, angle, center = self.rect_info
            x1, y1, _, _ = self.aruco.area_deteccion

            # ── Relleno semitransparente de la hoja (contorno completo) ──
            if self.cnt is not None:
                overlay    = display.copy()
                cnt_global = self.cnt + np.array([x1, y1])
                cv2.drawContours(overlay, [cnt_global], -1, (0, 255, 0), -1)
                cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
                cv2.drawContours(display, [cnt_global], -1, (0, 255, 0), 2)

            # ── Bounding box del LIMBO (amarillo) ──
            cv2.drawContours(display, [box], 0, (0, 255, 255), 2)
            cv2.circle(display, center, 6, (0, 255, 255), -1)

            m = self.measurement

            # ── Línea de LARGO  ápice → base  (azul) ──
            if m.pt_apice and m.pt_base:
                cv2.line(display, m.pt_apice, m.pt_base, (255, 0, 0), 3)
                mid_largo = (
                    (m.pt_apice[0] + m.pt_base[0]) // 2,
                    (m.pt_apice[1] + m.pt_base[1]) // 2,
                )
                cv2.putText(
                    display, f"{m.alto:.1f} cm",
                    (mid_largo[0] + 6, mid_largo[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2,
                )
            else:
                # fallback: eje del bounding box si aún no hay puntos
                angle_rad = math.radians(angle)
                dx, dy    = math.cos(angle_rad), math.sin(angle_rad)
                hw        = w / 2
                p1 = (int(center[0] - dx * hw), int(center[1] - dy * hw))
                p2 = (int(center[0] + dx * hw), int(center[1] + dy * hw))
                cv2.line(display, p1, p2, (255, 0, 0), 3)

            # ── Línea de ANCHO MÁXIMO perpendicular (rojo) ──
            if m.pt_ancho_L and m.pt_ancho_R:
                cv2.line(display, m.pt_ancho_L, m.pt_ancho_R, (0, 0, 255), 3)
                mid_ancho = (
                    (m.pt_ancho_L[0] + m.pt_ancho_R[0]) // 2,
                    (m.pt_ancho_L[1] + m.pt_ancho_R[1]) // 2,
                )
                cv2.putText(
                    display, f"{m.ancho:.1f} cm",
                    (mid_ancho[0] + 6, mid_ancho[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )
            else:
                angle_rad = math.radians(angle)
                dx, dy = math.cos(angle_rad), math.sin(angle_rad)
                px, py = -dy, dx
                hh     = h / 2
                p3 = (int(center[0] - px * hh), int(center[1] - py * hh))
                p4 = (int(center[0] + px * hh), int(center[1] + py * hh))
                cv2.line(display, p3, p4, (0, 0, 255), 3)

        return display

    # ------------------------------------------------------------------
    def _draw_measurement_panel(self, ax):
        """4 tarjetas con las mediciones de la hoja (Ancho, Alto, Color,
        Tamaño), en el mismo estilo de tarjeta que las barras de rasgos
        botánicos de ExplainWindow."""
        ax.clear()
        ax.set_facecolor(_PANEL)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
        ax.set_title("Medición de la hoja", color="white", fontsize=13, pad=8, loc="left")

        m = self.measurement
        cards = [
            ("Ancho",  f"{m.ancho:.1f} cm", _TEAL),
            ("Alto",   f"{m.alto:.1f} cm",  _TEAL),
            ("Color",  _COLOR_DISPLAY.get(m.color, m.color),   "#ffcc66"),
            ("Tamaño", _TAMANO_DISPLAY.get(m.tamano, m.tamano), "#ffcc66"),
        ]
        cw, ch, gx, y0 = 22, 40, 2.5, 30
        for i, (label, value, color) in enumerate(cards):
            x0 = gx + i * (cw + gx)
            ax.add_patch(FancyBboxPatch((x0, y0), cw, ch, boxstyle="round,pad=0,rounding_size=2.2",
                                        facecolor=_GRAY_FILL, edgecolor=_GRAY_BORDER, linewidth=1.0))
            ax.text(x0 + cw / 2, y0 + ch - 9, label, color="#9a9ab0", fontsize=9.5, ha="center")
            ax.text(x0 + cw / 2, y0 + 11, value, color=color, fontsize=13, fontweight="bold", ha="center")

    # ------------------------------------------------------------------
    def _class_thumbnail(self, plant_key: str, size: int = 100) -> np.ndarray | None:
        if plant_key in self._thumb_cache:
            return self._thumb_cache[plant_key]
        folder = os.path.join(self.classifier.train_folder, plant_key)
        if not os.path.exists(folder):
            return None
        files = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)]
        if not files:
            return None
        img = cv2.imread(os.path.join(folder, files[0]))
        if img is None:
            return None
        thumb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size))
        self._thumb_cache[plant_key] = thumb
        return thumb

    def _draw_species_panel(self, ax):
        """Tira de las 5 especies con miniatura y probabilidad, igual que
        el cuadrante "4) Predicción final" de ExplainWindow: la predicha
        resaltada, las demás atenuadas."""
        ax.clear()
        ax.set_facecolor(_BG)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
        ax.set_title("Predicción", color="white", fontsize=13, pad=8, loc="left")

        if self.last_probs is None:
            ax.text(50, 50, "Analizando...", color="#9a9ab0", fontsize=12, ha="center", va="center")
            return

        names, class_keys = self.last_names, self.last_class_keys
        probs, pred_idx = self.last_probs, self.last_pred_idx
        pw = 100 / len(names)
        for j, (name, key, p) in enumerate(zip(names, class_keys, probs)):
            cx = pw * j + pw / 2
            is_winner = (j == pred_idx)
            alpha = 1.0 if is_winner else _DIM_ALPHA
            pct = round(float(p) * 100)
            edge = _TEAL if is_winner else _GRAY_BORDER
            fill = "#123b30" if is_winner else _GRAY_FILL
            ax.add_patch(FancyBboxPatch((pw * j + 2, 8), pw - 4, 62, boxstyle="round,pad=0,rounding_size=2",
                                        linewidth=2.2 if is_winner else 0.8, edgecolor=edge, facecolor=fill,
                                        zorder=4, alpha=alpha))
            thumb = self._class_thumbnail(key)
            if thumb is not None:
                ax.imshow(thumb, extent=(cx - pw * 0.42, cx + pw * 0.42, 14, 64), zorder=5, alpha=alpha)
            name_color = _TEAL if is_winner else "white"
            ax.text(cx, 76, name, color=name_color, fontsize=10, ha="center",
                   weight="bold" if is_winner else "normal", alpha=alpha)
            ax.text(cx, 87, f"{pct}%", color="#c8c8d8", fontsize=9, ha="center", alpha=alpha)
        ax.set_aspect("auto")

    # ------------------------------------------------------------------
    def _on_key(self, event):
        if event.key == "q":
            self.running = False
            plt.close()
        elif event.key == " " and self.current_frame is not None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            img_filename = f"captura_{timestamp}.jpg"
            json_filename = f"captura_{timestamp}.json"
            img_path = os.path.join(
                self.classifier.test_folder, "mifoto", img_filename
            )
            json_path = os.path.join(
                self.classifier.test_folder, "mifoto", json_filename
            )
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            cv2.imwrite(img_path, self.current_frame)

            # Guardar mediciones de la hoja como JSON
            m = self.measurement
            leaf_data = {
                "ancho_cm": m.ancho,
                "alto_cm": m.alto,
                "area_cm2": m.area_cm2,
                "angulo_grados": m.angulo
            }
            with open(json_path, 'w') as f:
                json.dump(leaf_data, f)

            print(f"\n📸 Captura guardada: {img_path}")
            print(f"📏 Mediciones guardadas: {json_path}")
        elif event.key == "r":
            self.aruco.reset()
            self.measurement = LeafMeasurement()
            print("\n🔄 Calibración reiniciada")

    @staticmethod
    def _print_instructions(camera_id: int) -> None:
        print(f"\n✅ Cámara {camera_id} iniciada")
        print("\n🎮 CONTROLES:")
        print("   Q          → Salir")
        print("   ESPACIO    → Capturar frame")
        print("   R          → Reiniciar calibración ArUco\n")
