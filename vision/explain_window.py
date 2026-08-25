# ============================================================
# vision/explain_window.py  --  Ventana interactiva de explicabilidad
#                              para "Predecir una imagen (sin guardar)"
# ============================================================
import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import TextBox

from config.settings import IMAGE_EXTENSIONS

_BG = "#0f0f23"
_PANEL = "#151530"
_TEAL = "#5DCAA5"
_GRAY_FILL = "#1a1a3e"
_GRAY_BORDER = "#444441"
_DIM_ALPHA = 0.5
_HEAT_CMAP = LinearSegmentedColormap.from_list("heat", ["#000000", "#ffcc33", "#ff3333"])
_C_ONDULACION = "#00e5ff"
_C_RUGOSIDAD = "#7c4dff"
_C_DENSIDAD_VENAS = "#ffea00"
_C_ORIENTACION_VENAS = "#ff6d00"
_C_COMPACIDAD = "#00e676"

_VISIBLE_ROWS = 10
_ROW_STEP = 8
_ROW_Y0 = 12

# La animación de la hoja y la de las barras corren juntas en el mismo
# bucle (mismo t de 0 a 1) para no sumar tiempos de espera; ~45 pasos a
# ~0.055s cada uno son unos 2.5s en total.
_ANIM_STEPS = 45
_ANIM_PAUSE = 0.055
# Ventanas [inicio, fin] de t (0-1) en las que aparece cada capa de la hoja,
# igual que en el GIF: primero el mapa de calor, luego el contorno y por
# último las venas.
_STAGE_CAM = (0.00, 0.18)
_STAGE_CONTORNO = (0.12, 0.38)
_STAGE_RUGOSIDAD = (0.32, 0.58)
_STAGE_DENSIDAD = (0.52, 0.82)
_STAGE_ORIENTACION = (0.75, 1.0)


def _seg(t: float, start: float, end: float) -> float:
    if t <= start:
        return 0.0
    if t >= end:
        return 1.0
    return (t - start) / (end - start)


def _axes_y_per_x(ax, fig) -> float:
    """Cuántas unidades de datos en y equivalen, en tamaño físico en
    pantalla, a una unidad de datos en x. Los ejes del panel son mucho más
    anchos que altos en unidades de datos (0-100 en x, unas pocas filas en
    y), así que sin este factor cualquier ícono "cuadrado" sale estirado
    verticalmente. Se calcula con el tamaño real de los ejes en vez de un
    número fijo para que no dependa del tamaño de la ventana."""
    pos = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    ax_w_in = max(pos.width * fig_w, 1e-6)
    ax_h_in = max(pos.height * fig_h, 1e-6)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_units_per_in = abs(x1 - x0) / ax_w_in
    y_units_per_in = abs(y1 - y0) / ax_h_in
    return y_units_per_in / x_units_per_in


def _draw_trait_icon(ax, kind: str, cx: float, cy: float, color: str, h: float):
    """Ícono chico junto al título de cada rasgo: una pista visual rápida
    de qué se está midiendo, sin ocupar el espacio de un dibujo completo."""
    if kind == "wave":
        t = np.linspace(0, 2 * np.pi, 40)
        ax.plot(cx + (t - np.pi) * 0.16, cy + h * np.sin(t * 2.2), color=color, lw=1.4)
    elif kind == "zigzag":
        xs = cx + np.linspace(-0.5, 0.5, 7)
        ys = cy + np.array([0, 1, -0.4, 1, -0.6, 0.8, 0]) * h
        ax.plot(xs, ys, color=color, lw=1.4)
    elif kind == "lines":
        for dx in (-0.35, 0, 0.35):
            ax.plot([cx + dx, cx + dx * 0.5], [cy - h, cy + h], color=color, lw=1.3)
    elif kind == "fan":
        base = (cx, cy - h)
        for ang_deg in (60, 78, 90, 102, 120):
            ang = np.radians(ang_deg)
            ax.plot([base[0], base[0] + 0.42 * np.cos(ang)],
                   [base[1], base[1] + 2 * h * np.sin(ang)], color=color, lw=1.2)
    elif kind == "oval":
        t = np.linspace(0, 2 * np.pi, 40)
        ax.plot(cx + 0.32 * np.cos(t), cy + h * 1.1 * np.sin(t), color=color, lw=1.4)


def _bucket(frac: float, labels: list[str]) -> str:
    idx = min(len(labels) - 1, int(max(0.0, min(1.0, frac)) * len(labels)))
    return labels[idx]


def _trait_rows(bot_feat) -> list[tuple]:
    """Traduce el vector de 30 características botánicas crudas a las 6
    filas del panel: (título en lenguaje llano, palabra baja, palabra alta,
    fracción 0-1 para la barra, texto del valor, color, ícono). Las escalas
    (1.0-2.0, 0-0.4, etc.) son de referencia para dar intuición, no
    umbrales botánicos exactos."""
    ond = float(bot_feat[0])
    ond_frac = (ond - 1.0) / 1.0
    ond_label = _bucket(ond_frac, ["Liso", "Ligeramente ondulado", "Ondulado", "Muy ondulado"])

    rug = float(bot_feat[1])
    rug_frac = rug / 0.4
    rug_label = _bucket(rug_frac, ["Muy uniforme", "Bastante uniforme", "Algo irregular", "Muy irregular"])

    dens = float(bot_feat[2])
    dens_frac = dens / 1.0
    dens_label = _bucket(dens_frac, ["Muy pocas", "Pocas", "Densidad media", "Muchas"])

    hist = bot_feat[3:27]
    pico = float(np.max(hist)) if len(hist) else 0.0
    orient_frac = (pico - 1 / 24) / (0.35 - 1 / 24)
    orient_label = _bucket(orient_frac, ["Dispersas", "Moderadamente paralelas", "Bastante paralelas", "Muy paralelas"])

    compac = float(bot_feat[27])
    compac_frac = (compac - 12.0) / (30.0 - 12.0)
    compac_label = _bucket(compac_frac, ["Muy compacta", "Compacta", "Alargada", "Muy alargada"])

    return [
        (f"Forma del borde: {ond_label}", "Liso", "Muy ondulado", ond_frac, f"{ond:.2f}", _C_ONDULACION, "wave"),
        (f"Textura del contorno: {rug_label}", "Uniforme", "Irregular", rug_frac, f"{rug:.2f}", _C_RUGOSIDAD, "zigzag"),
        (f"Cantidad de venas: {dens_label}", "Pocas", "Muchas", dens_frac, f"{dens:.2f}", _C_DENSIDAD_VENAS, "lines"),
        (f"Patrón de las venas: {orient_label}", "Dispersas", "Paralelas", orient_frac, f"pico: {pico:.0%}",
         _C_ORIENTACION_VENAS, "fan"),
        (f"Forma general: {compac_label}", "Compacta", "Alargada", compac_frac, f"{compac:.2f}",
         _C_COMPACIDAD, "oval"),
    ]


class ExplainWindow:
    """
    Ventana única para "Predecir una imagen (sin guardar)": muestra en 4
    cuadrantes la hoja (Grad-CAM + rasgos botánicos en colores), los valores
    de esos rasgos, un selector siempre disponible (número + lista con
    scroll) y la predicción final con una muestra de cada especie. Cambiar
    la imagen en el cuadrante 3 actualiza los otros tres al instante, sin
    cerrar ni volver a abrir nada.
    """

    def __init__(self, classifier, mifoto_dir: str):
        self.classifier = classifier
        self.mifoto_dir = mifoto_dir
        self.files = sorted(
            f for f in os.listdir(mifoto_dir) if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        self._cache: dict[str, dict] = {}
        self._thumb_cache: dict[str, np.ndarray] = {}
        self.list_offset = 0
        self.current_idx = 0

        self.fig = plt.figure(figsize=(14, 12), facecolor=_BG)
        self._build_layout()
        self._select(0)

    # ------------------------------------------------------------
    def show(self):
        plt.show()
        plt.close(self.fig)

    # ------------------------------------------------------------
    def _build_layout(self):
        gs = self.fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                                   hspace=0.26, wspace=0.14, top=0.89, bottom=0.04,
                                   left=0.04, right=0.97)
        gs_bottom = gs[1, :].subgridspec(1, 2, width_ratios=[0.72, 1.28], wspace=0.05)
        gs3 = gs_bottom[0].subgridspec(2, 1, height_ratios=[0.13, 0.87], hspace=0.4)

        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1])
        self.ax_box = self.fig.add_subplot(gs3[0])
        self.ax_list = self.fig.add_subplot(gs3[1])
        self.ax4 = self.fig.add_subplot(gs_bottom[1])

        # El cuadro de texto no se ve "clicable" por defecto: se le pone un
        # borde brillante permanente y una instrucción al lado, en vez de
        # depender de que el usuario adivine que hay que hacer clic primero.
        for spine in self.ax_box.spines.values():
            spine.set_color(_TEAL)
            spine.set_linewidth(1.6)
        self.text_box = TextBox(self.ax_box, "N°: ", initial="1",
                                color="#1a1a3e", hovercolor="#22224a")
        self.text_box.label.set_color("#9a9ab0")
        self.text_box.text_disp.set_color("white")
        self.text_box.on_submit(self._on_submit)
        box_pos = self.ax_box.get_position()
        self.fig.text(box_pos.x0, box_pos.y1 + 0.006,
                      "haz clic aquí, escribe el número y presiona Enter",
                      color="#ffcc66", fontsize=9, style="italic")

        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click_list)

    # ------------------------------------------------------------
    def _analyze(self, filename: str) -> dict | None:
        if filename not in self._cache:
            path = os.path.join(self.mifoto_dir, filename)
            data = self.classifier.analyze_for_explanation(path)
            if data is None:
                return None
            self._cache[filename] = data
        return self._cache[filename]

    def _class_thumbnail(self, plant_key: str, size: int = 110) -> np.ndarray | None:
        if plant_key in self._thumb_cache:
            return self._thumb_cache[plant_key]
        folder = os.path.join(self.classifier.train_folder, plant_key)
        if not os.path.exists(folder):
            return None
        files = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)]
        if not files:
            return None
        img = cv2.imread(os.path.join(folder, random.choice(files)))
        if img is None:
            return None
        thumb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size))
        self._thumb_cache[plant_key] = thumb
        return thumb

    # ------------------------------------------------------------
    def _select(self, idx: int):
        if not self.files:
            return
        idx = max(0, min(idx, len(self.files) - 1))
        data = self._analyze(self.files[idx])
        if data is None:
            print(f"No se pudo procesar {self.files[idx]}.")
            return
        self.current_idx = idx
        if idx < self.list_offset:
            self.list_offset = idx
        elif idx >= self.list_offset + _VISIBLE_ROWS:
            self.list_offset = idx - _VISIBLE_ROWS + 1

        self._draw_leaf(data)
        self._draw_traits(data)
        self._draw_list()
        self._draw_species(data)
        self.fig.suptitle(f"Predecir una imagen (sin guardar) — {data['img_filename']}",
                          color="white", fontsize=15, y=0.965)
        self._animate_reveal()

    def _on_submit(self, text: str):
        text = text.strip()
        if not text:
            return
        try:
            n = int(text)
        except ValueError:
            print("Número inválido.")
            return
        if n < 1 or n > len(self.files):
            print(f"Número inválido. Debe ser entre 1 y {len(self.files)}.")
            return
        self._select(n - 1)

    def _on_scroll(self, event):
        if event.inaxes != self.ax_list:
            return
        step = 3 if event.button == "up" else -3
        max_offset = max(0, len(self.files) - _VISIBLE_ROWS)
        self.list_offset = max(0, min(self.list_offset - step, max_offset))
        self._draw_list()
        self.fig.canvas.draw_idle()

    def _on_click_list(self, event):
        if event.inaxes != self.ax_list or event.ydata is None:
            return
        row = int((event.ydata - _ROW_Y0 + _ROW_STEP / 2) // _ROW_STEP)
        idx = self.list_offset + row
        if 0 <= row < _VISIBLE_ROWS and idx < len(self.files):
            # set_val() ya dispara el evento "submit" del cuadro de texto
            # (ver TextBox.set_val en matplotlib), que llama a _on_submit()
            # y este a su vez a _select(). Llamarlo aquí también hacía que
            # la predicción se calculara y animara dos veces por cada clic.
            self.text_box.set_val(str(idx + 1))

    # ------------------------------------------------------------
    def _draw_leaf(self, data: dict):
        """Crea los artistas de la hoja vacíos/apagados; _animate_reveal()
        los va llenando por etapas (mapa de calor -> contorno -> venas ->
        estípulas), igual que en el GIF."""
        ax = self.ax1
        ax.cla()
        self._leaf = {}
        img_uint8 = data["img_uint8"]
        ax.imshow(img_uint8)
        self._leaf["cam_im"] = ax.imshow(data["cam"], cmap=_HEAT_CMAP, alpha=0.0, vmin=0, vmax=1)

        contour = data["contour"]
        if contour is not None and len(contour) > 0:
            mask = data["mask"]
            rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
            r, g, b = tuple(int(_C_COMPACIDAD.lstrip("#")[j:j + 2], 16) / 255 for j in (0, 2, 4))
            rgba[..., 0] = r; rgba[..., 1] = g; rgba[..., 2] = b
            rgba[..., 3] = (mask > 0) * 1.0
            self._leaf["mask_im"] = ax.imshow(rgba, alpha=0.0)

            contour_pts = contour.reshape(-1, 2)
            (contour_line,) = ax.plot([], [], color=_C_ONDULACION, lw=2.0)
            self._leaf["contour_line"] = contour_line
            self._leaf["contour_pts"] = contour_pts

            centroid = data["centroid"]
            if centroid is not None:
                sample = contour_pts[::6]
                xs, ys = [], []
                for (px, py) in sample:
                    xs += [centroid[0], px, np.nan]
                    ys += [centroid[1], py, np.nan]
                (spokes_line,) = ax.plot([], [], color=_C_RUGOSIDAD, lw=0.6, alpha=0.65)
                self._leaf["spokes_line"] = spokes_line
                self._leaf["spokes_xy"] = (np.array(xs), np.array(ys), len(sample))

            edge_ys, edge_xs = np.nonzero(data["edges"])
            if len(edge_xs) > 0:
                dens_idx = np.arange(0, len(edge_xs), 4)
                scat = ax.scatter([], [], s=3, color=_C_DENSIDAD_VENAS, alpha=0.85, linewidths=0)
                self._leaf["dens_scat"] = scat
                self._leaf["dens_xy"] = np.column_stack([edge_xs[dens_idx], edge_ys[dens_idx]])

                orient_idx = np.arange(0, len(edge_xs), 22)
                ang_deg = data["ang_deg"]
                xs, ys = [], []
                for k in orient_idx:
                    x0, y0 = edge_xs[k], edge_ys[k]
                    theta = np.radians(ang_deg[y0, x0])
                    dx, dy = 4.5 * np.cos(theta), 4.5 * np.sin(theta)
                    xs += [x0 - dx, x0 + dx, np.nan]
                    ys += [y0 - dy, y0 + dy, np.nan]
                (orient_line,) = ax.plot([], [], color=_C_ORIENTACION_VENAS, lw=1.1, alpha=0.9)
                self._leaf["orient_line"] = orient_line
                self._leaf["orient_xy"] = (np.array(xs), np.array(ys), len(orient_idx))
        else:
            ax.text(0.5, 0.5, "No se detectó la hoja", color="white", ha="center", va="center",
                   transform=ax.transAxes)

        ax.set_title("1) La hoja — la red mira la imagen y sus rasgos botánicos",
                     color="white", fontsize=13, pad=8)
        ax.axis("off")

    def _update_leaf(self, t: float):
        leaf = self._leaf
        if "cam_im" in leaf:
            leaf["cam_im"].set_alpha(0.4 * _seg(t, *_STAGE_CAM))

        p_contorno = _seg(t, *_STAGE_CONTORNO)
        if "mask_im" in leaf:
            leaf["mask_im"].set_alpha(0.30 * p_contorno)
        if "contour_line" in leaf and p_contorno > 0:
            pts = leaf["contour_pts"]
            n = max(2, int(len(pts) * p_contorno))
            leaf["contour_line"].set_data(pts[:n, 0], pts[:n, 1])

        p_rug = _seg(t, *_STAGE_RUGOSIDAD)
        if "spokes_line" in leaf and p_rug > 0:
            xs, ys, n_segs = leaf["spokes_xy"]
            n = max(1, int(n_segs * p_rug)) * 3
            leaf["spokes_line"].set_data(xs[:n], ys[:n])

        p_dens = _seg(t, *_STAGE_DENSIDAD)
        if "dens_scat" in leaf and p_dens > 0:
            xy = leaf["dens_xy"]
            n = max(1, int(len(xy) * p_dens))
            leaf["dens_scat"].set_offsets(xy[:n])

        p_orient = _seg(t, *_STAGE_ORIENTACION)
        if "orient_line" in leaf and p_orient > 0:
            xs, ys, n_segs = leaf["orient_xy"]
            n = max(1, int(n_segs * p_orient)) * 3
            leaf["orient_line"].set_data(xs[:n], ys[:n])

    # ------------------------------------------------------------
    def _draw_traits(self, data: dict):
        """Crea las barras en 0 ; _animate_reveal() las llena como barras
        de carga hasta su valor real. Cada fila se lee en lenguaje llano
        (ej. "Forma del borde: Ligeramente ondulado") con una escala en
        palabras en vez de solo un número crudo, para alguien que no sepa
        qué es cada rasgo."""
        ax = self.ax2
        ax.cla()
        self._bars: list[tuple] = []       # (rect, target_width, delay)
        self._bar_markers: list[tuple] = []  # (polygon, delay)
        rows = _trait_rows(data["bot_feat"])
        ax.set_facecolor(_PANEL)
        ax.set_xlim(0, 100); ax.set_ylim(len(rows), -1); ax.axis("off")
        ax.set_title("2) Características botánicas", color="white", fontsize=13, pad=10, loc="left")
        y_per_x = _axes_y_per_x(ax, self.fig)
        icon_h = 0.5 * y_per_x

        for idx, (title, lo, hi, frac, value_txt, color, kind) in enumerate(rows):
            row_y = idx
            delay = idx * 0.05
            _draw_trait_icon(ax, kind, 2.2, row_y - 0.30, color, icon_h)
            ax.text(5, row_y - 0.30, title, color="white", fontsize=11, weight="bold", va="center")
            ax.text(99, row_y - 0.30, value_txt, color="#666680", fontsize=8.5, ha="right", va="center")
            ax.text(1, row_y + 0.04, lo, color="#8a8aa0", fontsize=8)
            ax.text(99, row_y + 0.04, hi, color="#8a8aa0", fontsize=8, ha="right")

            track_y, track_h = row_y + 0.12, 0.22
            ax.add_patch(Rectangle((1.5, track_y), 97, track_h, facecolor="#2a2a4a", edgecolor="none"))
            target_w = 97 * max(0.0, min(1.0, frac))
            rect = Rectangle((1.5, track_y), 0, track_h, facecolor=color, alpha=0.6, edgecolor="none")
            ax.add_patch(rect)
            self._bars.append((rect, target_w, delay))

            px = 1.5 + target_w
            marker = Polygon([[px - 0.8, track_y + track_h + 0.20], [px + 0.8, track_y + track_h + 0.20],
                              [px, track_y + track_h + 0.02]],
                             closed=True, facecolor=color, edgecolor="white", linewidth=0.6, zorder=5, alpha=0)
            ax.add_patch(marker)
            self._bar_markers.append((marker, delay))

    def _update_bars(self, t: float):
        for rect, target, delay in self._bars:
            p = _seg(t, delay, min(1.0, delay + 0.65))
            rect.set_width(target * p)
        for marker, delay in self._bar_markers:
            marker.set_alpha(_seg(t, delay + 0.55, min(1.0, delay + 0.8)))

    # ------------------------------------------------------------
    def _draw_list(self):
        ax = self.ax_list
        ax.cla()
        ax.set_facecolor(_PANEL)
        ax.set_xlim(0, 100); ax.set_ylim(100, 0); ax.axis("off")
        ax.set_title("3) Elegir imagen", color="white", fontsize=13, pad=8, loc="left")
        ax.text(1, 4, f"Lista con scroll ({len(self.files)} imágenes) — rueda del mouse o clic:",
               color="#9a9ab0", fontsize=9.5)
        end = min(self.list_offset + _VISIBLE_ROWS, len(self.files))
        for row, i in enumerate(range(self.list_offset, end)):
            y = _ROW_Y0 + row * _ROW_STEP
            active = (i == self.current_idx)
            if active:
                ax.add_patch(FancyBboxPatch((1, y - 3.4), 97, 6.4, boxstyle="round,pad=0,rounding_size=1",
                                            edgecolor=_TEAL, facecolor="#123b30", linewidth=1.1))
            color = _TEAL if active else "#c8c8d8"
            ax.text(3, y, f"{i + 1}.", color=color, fontsize=9.5)
            ax.text(9, y, self.files[i], color=color, fontsize=9.5)

    def _draw_species(self, data: dict):
        """Predicción final: la especie elegida a opacidad completa, las
        otras cuatro atenuadas al 50% para que resalte sin desaparecer."""
        ax = self.ax4
        ax.cla()
        ax.set_facecolor(_BG)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
        ax.set_title("4) Predicción final", color="white", fontsize=13, pad=8, loc="left")
        names, class_keys, probs, pred_idx = data["names"], data["class_keys"], data["probs"], data["pred_idx"]
        pw = 100 / len(names)
        for j, (name, cls_key, p) in enumerate(zip(names, class_keys, probs)):
            cx = pw * j + pw / 2
            is_winner = (j == pred_idx)
            alpha = 1.0 if is_winner else _DIM_ALPHA
            pct = round(float(p) * 100)
            edge = _TEAL if is_winner else _GRAY_BORDER
            fill = "#123b30" if is_winner else _GRAY_FILL
            ax.add_patch(FancyBboxPatch((pw * j + 2, 8), pw - 4, 62, boxstyle="round,pad=0,rounding_size=2",
                                        linewidth=2.2 if is_winner else 0.8, edgecolor=edge, facecolor=fill,
                                        zorder=4, alpha=alpha))
            thumb = self._class_thumbnail(cls_key)
            if thumb is not None:
                ax.imshow(thumb, extent=(cx - pw * 0.44, cx + pw * 0.44, 12, 66), zorder=5, alpha=alpha)
            name_color = _TEAL if is_winner else "white"
            ax.text(cx, 78, name, color=name_color, fontsize=10.5, ha="center",
                   weight="bold" if is_winner else "normal", alpha=alpha)
            sub = f"{pct}% - la elegida" if is_winner else f"{pct}%"
            ax.text(cx, 88, sub, color="#c8c8d8", fontsize=9.5, ha="center", alpha=alpha)
        ax.set_aspect("auto")

    # ------------------------------------------------------------
    def _animate_reveal(self):
        """La hoja se revela por etapas (como el GIF) y las barras suben
        como barras de carga, las dos cosas a la vez en unos 2.5s."""
        for step in range(1, _ANIM_STEPS + 1):
            t = step / _ANIM_STEPS
            self._update_leaf(t)
            self._update_bars(t)
            plt.pause(_ANIM_PAUSE)
