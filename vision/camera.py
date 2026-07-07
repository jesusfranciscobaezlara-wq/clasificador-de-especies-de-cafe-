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

from vision.aruco import ArucoCalibrator
from vision.leaf_detector import LeafDetector, LeafMeasurement


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
        self.measurement    = LeafMeasurement()
        self.cnt            = None
        self.rect_info      = None
        self.show_details   = False

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
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"❌ Cámara {camera_id} no disponible")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True

        plt.ion()
        fig = plt.figure(figsize=(18, 7), facecolor="#1a1a1a")
        fig.canvas.mpl_connect("key_press_event", self._on_key)
        ax_video = plt.subplot(1, 2, 1, facecolor="#1a1a1a")
        ax_info  = plt.subplot(1, 2, 2, facecolor="#0a0a1a")
        plt.tight_layout()

        ani = animation.FuncAnimation(   # noqa: F841
            fig, self._update_frame,
            fargs=(ax_video, ax_info),
            interval=33, cache_frame_data=False,
        )
        self._print_instructions(camera_id)
        plt.show(block=True)

        self.cap.release()
        plt.close()
        print("\n✅ Modo cámara finalizado")

    # ------------------------------------------------------------------
    def _update_frame(self, frame_num, ax_video, ax_info):
        if not self.running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_frame = frame.copy()

        if frame_num % 10 == 0:
            pred, conf = self.classifier.predict_frame(frame)
            if pred:
                self.last_pred     = self.classifier.get_display_name(pred)
                self.last_conf     = conf
                self.last_pred_key = pred

        self.aruco.process_frame(frame)
        if self.aruco.calibrado and self.aruco.area_deteccion:
            self.cnt, self.rect_info, self.measurement = self.leaf_det.detect(
                frame, self.aruco.area_deteccion, self.aruco.pixeles_por_cm
            )
            self._aplicar_ema()

        display   = self.aruco.draw_overlay(frame)
        display   = self._draw_leaf_overlay(display)
        frame_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        cv2.putText(frame_rgb,
                    f"Cafe: {self.last_pred} ({self.last_conf:.1%})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_rgb,
                    "Q:Salir | SPACE:Capturar | S:Detalles | R:Reiniciar",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        ax_video.clear()
        ax_video.imshow(frame_rgb)
        ax_video.set_title("SISTEMA INTEGRADO", color="white",
                           fontsize=14, fontweight="bold")
        ax_video.axis("off")

        self._draw_info_panel(ax_info)

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
    def _draw_info_panel(self, ax):
        ax.clear()
        ax.set_facecolor("#0a0a1a")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        m = self.measurement
        rows = [
            ("Café",        self.last_pred,          "#ffff00"),
            ("Confianza",   f"{self.last_conf:.1%}",  "#cccccc"),
            ("Ancho",       f"{m.ancho:.1f} cm",      "#ffff00"),
            ("Alto",        f"{m.alto:.1f} cm",       "#ffff00"),
            ("Ángulo",      f"{m.angulo:.0f}°",       "#88ff88"),
            ("Color",       m.color,                  "#88ff88"),
            ("Tamaño",      m.tamano,                 "#88ff88"),
            ("Área",        f"{m.area_cm2:.0f} cm²",  "#aaaaaa"),
        ]
        for i, (key, val, color) in enumerate(rows):
            y = 9.0 - i * 1.0
            ax.text(1, y, key, fontsize=10, color="#aaaaaa")
            ax.text(9, y, val, fontsize=11, fontweight="bold",
                    color=color, ha="right")

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
        elif event.key == "s":
            self.show_details = not self.show_details
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
        print("   S          → Mostrar/ocultar detalles")
        print("   R          → Reiniciar calibración ArUco\n")