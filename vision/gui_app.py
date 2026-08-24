# ============================================================
# vision/gui_app.py  --  Interfaz gráfica (Tkinter) del sistema de café
# Se ejecuta desde main.py en la raíz del proyecto.
# ============================================================
import os
import sys

# Ver la nota equivalente en main.py: evita que un print() con emoji crashee
# el programa si la consola no está en UTF-8.
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import threading
import io

# Para gráficos del monitor (requiere matplotlib)
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Ajustar sys.path para que las importaciones del proyecto funcionen
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_PROJECT_DIR)

# --- Importaciones del proyecto ---
from services.classifier import CoffeePlantClassifier
from vision.camera import CameraSession
from vision.measurement import calcular_medidas_imagen
from config.settings import PLANT_DESCRIPTIONS, get_display_name, IMAGE_EXTENSIONS

# Ruta base de datasets (configurable por variable de entorno)
BASE_DIR = os.environ.get("COFFEE_BASE_DIR", os.path.join(BASE_PROJECT_DIR, "datasets"))


def build_classifier():
    return CoffeePlantClassifier(
        train_folder   = os.path.join(BASE_DIR, "Training"),
        test_folder    = os.path.join(BASE_DIR, "Test"),
        results_folder = os.path.join(BASE_DIR, "Results"),
    )


def _pick_sample_image(classifier, plant_class):
    """Elige una imagen de ejemplo al azar de la carpeta de entrenamiento de la clase."""
    folder = os.path.join(classifier.train_folder, plant_class)
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(IMAGE_EXTENSIONS)]
        if files:
            return os.path.join(folder, random.choice(files))
    return None

# ============================================================
# Aplicación principal
# ============================================================
class CoffeeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🌱 Clasificador de Plantas de Café")
        self.geometry("1000x700")
        self.configure(bg="#1a1a2e")

        self.classifier = build_classifier()
        self.model_ready = self.classifier.load_models()

        # Páginas
        self.frames = {}
        for Page in (StartPage, TrainPage, TestMenuPage, ImageTestPage):
            frame = Page(self, self.classifier)
            self.frames[Page] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, page_class):
        frame = self.frames[page_class]
        frame.tkraise()

    def refresh_model_status(self):
        self.model_ready = self.classifier.load_models()


# ============================================================
# Páginas
# ============================================================
class StartPage(tk.Frame):
    def __init__(self, parent, classifier):
        super().__init__(parent, bg="#1a1a2e")
        self.classifier = classifier
        self.parent = parent

        title = tk.Label(self, text="🌱 Sistema de Clasificación de Café",
                         font=("Segoe UI", 24, "bold"), fg="#e94560", bg="#1a1a2e")
        title.pack(pady=60)

        btn_frame = tk.Frame(self, bg="#1a1a2e")
        btn_frame.pack(expand=True)

        btn_train = tk.Button(btn_frame, text="📚 Modo Entrenamiento",
                              font=("Segoe UI", 14), bg="#0f3460", fg="white",
                              activebackground="#e94560", activeforeground="white",
                              width=20, height=2, bd=0, cursor="hand2",
                              command=lambda: parent.show_frame(TrainPage))
        btn_train.pack(pady=15)

        btn_test = tk.Button(btn_frame, text="🔍 Modo Prueba",
                             font=("Segoe UI", 14), bg="#0f3460", fg="white",
                             activebackground="#e94560", activeforeground="white",
                             width=20, height=2, bd=0, cursor="hand2",
                             command=lambda: parent.show_frame(TestMenuPage))
        btn_test.pack(pady=15)

        self.lbl_status = tk.Label(self, text="", font=("Segoe UI", 10),
                                   bg="#1a1a2e", fg="#aaaaaa")
        self.lbl_status.pack(side=tk.BOTTOM, pady=15)
        self.update_status()

    def update_status(self):
        if self.classifier and self.classifier.trainer.model:
            classes = getattr(self.classifier.trainer, "class_list", [])
            self.lbl_status.config(text=f"Modelo cargado: {len(classes)} clases ({', '.join(classes)})")
        else:
            self.lbl_status.config(text="Modelo no disponible")


class TrainPage(tk.Frame):
    """Modo Entrenamiento: entrenar modelo + evaluar imágenes con feedback."""
    def __init__(self, parent, classifier):
        super().__init__(parent, bg="#1a1a2e")
        self.classifier = classifier
        self.parent = parent

        # Título
        title = tk.Label(self, text="Modo Entrenamiento", font=("Segoe UI", 20, "bold"),
                         fg="#e94560", bg="#1a1a2e")
        title.pack(pady=20)

        # Notebook con pestañas: Entrenar | Evaluar
        notebook = ttk.Notebook(self)   # ahora sí está definido
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Pestaña 1: Entrenamiento del modelo
        self.train_tab = tk.Frame(notebook, bg="#1a1a2e")
        notebook.add(self.train_tab, text="Entrenar modelo")
        self._build_train_tab()

        # Pestaña 2: Evaluación interactiva
        self.eval_tab = tk.Frame(notebook, bg="#1a1a2e")
        notebook.add(self.eval_tab, text="Evaluar imágenes")
        self._build_eval_tab()

        # Botón volver
        btn_back = tk.Button(self, text="← Volver al inicio", font=("Segoe UI", 10),
                             bg="#333344", fg="white", activebackground="#555566",
                             bd=0, cursor="hand2",
                             command=lambda: parent.show_frame(StartPage))
        btn_back.pack(pady=10)

    def _build_train_tab(self):
        frame = tk.Frame(self.train_tab, bg="#1a1a2e")
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.train_log = tk.Text(frame, height=12, width=80, font=("Consolas", 10),
                                 bg="#0f0f23", fg="#e0e0e0", state=tk.DISABLED,
                                 relief=tk.FLAT, padx=10, pady=10)
        self.train_log.pack(pady=10)

        btn_train = tk.Button(frame, text="Iniciar Entrenamiento", font=("Segoe UI", 12, "bold"),
                              bg="#00aa55", fg="white", activebackground="#008844",
                              width=20, height=2, bd=0, cursor="hand2",
                              command=self.start_training)
        btn_train.pack(pady=10)

    def _build_eval_tab(self):
        frame = tk.Frame(self.eval_tab, bg="#1a1a2e")
        frame.pack(fill=tk.BOTH, expand=True)

        # Controles superiores
        ctrl = tk.Frame(frame, bg="#1a1a2e")
        ctrl.pack(fill=tk.X, padx=10, pady=5)

        self.btn_load_eval = tk.Button(ctrl, text="Cargar imagen (Test/mifoto)",
                                       font=("Segoe UI", 11), bg="#0f3460", fg="white",
                                       command=self.load_eval_image)
        self.btn_load_eval.pack(side=tk.LEFT, padx=5)

        self.btn_predict_eval = tk.Button(ctrl, text="Predecir", font=("Segoe UI", 11, "bold"),
                                          bg="#00aa55", fg="white", state=tk.DISABLED,
                                          command=self.predict_eval)
        self.btn_predict_eval.pack(side=tk.LEFT, padx=5)

        # Panel de imágenes
        img_frame = tk.Frame(frame, bg="#1a1a2e")
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        img_frame.columnconfigure(0, weight=1)
        img_frame.columnconfigure(1, weight=1)
        img_frame.rowconfigure(0, weight=1)

        self.lbl_test_img = tk.Label(img_frame, bg="#0f0f23", text="Imagen de prueba",
                                     fg="white", font=("Segoe UI", 11))
        self.lbl_test_img.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.lbl_pred_img = tk.Label(img_frame, bg="#0f0f23", text="Imagen de predicción",
                                     fg="white", font=("Segoe UI", 11))
        self.lbl_pred_img.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # Panel de resultados y feedback
        res_frame = tk.Frame(frame, bg="#16213e", height=140)
        res_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)

        self.lbl_pred_eval = tk.Label(res_frame, text="Predicción: ---",
                                      font=("Segoe UI", 12, "bold"), fg="white", bg="#16213e")
        self.lbl_pred_eval.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.lbl_conf_eval = tk.Label(res_frame, text="Confianza: ---",
                                      font=("Segoe UI", 11), fg="white", bg="#16213e")
        self.lbl_conf_eval.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.lbl_medidas_eval = tk.Label(res_frame, text="Medidas: ---",
                                         font=("Segoe UI", 11), fg="#ffcc44", bg="#16213e")
        self.lbl_medidas_eval.grid(row=2, column=0, padx=10, pady=5, sticky="w")

        self.txt_info_eval = tk.Text(res_frame, height=6, width=35, font=("Segoe UI", 10),
                                     bg="#0f0f23", fg="white", state=tk.DISABLED, relief=tk.FLAT)
        self.txt_info_eval.grid(row=0, column=1, rowspan=3, padx=10, pady=5)

        # Botones de feedback
        feed_frame = tk.Frame(res_frame, bg="#16213e")
        feed_frame.grid(row=3, column=0, columnspan=2, pady=10)
        self.btn_correct = tk.Button(feed_frame, text="✔ Correcto", font=("Segoe UI", 10, "bold"),
                                     bg="#00aa55", fg="white", state=tk.DISABLED,
                                     command=self.feedback_correct)
        self.btn_correct.pack(side=tk.LEFT, padx=10)
        self.btn_incorrect = tk.Button(feed_frame, text="✘ Incorrecto", font=("Segoe UI", 10, "bold"),
                                       bg="#aa0000", fg="white", state=tk.DISABLED,
                                       command=self.feedback_incorrect)
        self.btn_incorrect.pack(side=tk.LEFT, padx=10)

        self.eval_img_path = None
        self.eval_pred = None

    # ------------------ Lógica de entrenamiento ------------------
    def start_training(self):
        self.train_log.config(state=tk.NORMAL)
        self.train_log.delete("1.0", tk.END)
        self.train_log.insert(tk.END, "⏳ Cargando datos y entrenando modelo...\n")
        self.train_log.config(state=tk.DISABLED)
        threading.Thread(target=self._run_training, daemon=True).start()

    def _run_training(self):
        # Se ejecuta en un hilo secundario: no tocar widgets de Tk aquí
        # directamente, solo agendar la actualización en el hilo principal.
        try:
            success = self.classifier.train_and_save()
        except Exception as e:
            self.after(0, self._on_training_done, False, str(e))
            return
        self.after(0, self._on_training_done, success, None)

    def _on_training_done(self, success, error):
        self.train_log.config(state=tk.NORMAL)
        if error is not None:
            self.train_log.insert(tk.END, f"❌ Error: {error}\n")
        elif success:
            self.train_log.insert(tk.END, "✅ Entrenamiento completado con éxito.\n")
            self.train_log.insert(tk.END, "El modelo se ha guardado correctamente.\n")
            self.parent.refresh_model_status()
        else:
            self.train_log.insert(tk.END, "❌ Falló el entrenamiento.\n")
        self.train_log.see(tk.END)
        self.train_log.config(state=tk.DISABLED)

    # ------------------ Evaluación interactiva ------------------
    def load_eval_image(self):
        mifoto_dir = os.path.join(self.classifier.test_folder, "mifoto")
        os.makedirs(mifoto_dir, exist_ok=True)
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen de prueba",
            initialdir=mifoto_dir,
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return
        self.eval_img_path = file_path
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "No se pudo cargar la imagen.")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._mostrar_imagen(img_rgb, self.lbl_test_img)
        self.btn_predict_eval.config(state=tk.NORMAL)
        self._clear_eval_results()

    def predict_eval(self):
        if not self.eval_img_path or not self.classifier.trainer.model:
            return
        pred, proba = self.classifier.predict_single_image(
            self.eval_img_path, show_image=False
        )
        if pred is None:
            messagebox.showerror("Error", "No se pudo procesar.")
            return
        self.eval_pred = pred
        display = get_display_name(pred)
        self.lbl_pred_eval.config(text=f"Predicción: {display}")
        self.lbl_conf_eval.config(text=f"Confianza: {proba:.2%}")

        # Medidas de hoja (ArUco + LeafDetector o JSON)
        medidas_txt = calcular_medidas_imagen(self.eval_img_path)
        self.lbl_medidas_eval.config(text=medidas_txt)

        if pred in PLANT_DESCRIPTIONS:
            info = PLANT_DESCRIPTIONS[pred]
            txt = f"Nombre: {info['nombre_cientifico']}\nAltitud: {info['altitud_ideal']}\nUso: {info['uso']}"
        else:
            txt = "Sin información."
        self.txt_info_eval.config(state=tk.NORMAL)
        self.txt_info_eval.delete("1.0", tk.END)
        self.txt_info_eval.insert(tk.END, txt)
        self.txt_info_eval.config(state=tk.DISABLED)

        sample = self._get_sample_image(pred)
        if sample:
            sample_img = cv2.cvtColor(cv2.imread(sample), cv2.COLOR_BGR2RGB)
            self._mostrar_imagen(sample_img, self.lbl_pred_img, "")
        else:
            self.lbl_pred_img.config(image="", text="Sin ejemplo")

        self.btn_correct.config(state=tk.NORMAL)
        self.btn_incorrect.config(state=tk.NORMAL)

    def feedback_correct(self):
        if not self.eval_img_path or not self.eval_pred:
            return
        # No se reinicia error_count aquí: debe comportarse igual que el
        # flujo de consola (main.py), que solo lo resetea tras un reentrenamiento.
        self.classifier.move_to_class_folder(self.eval_img_path, self.eval_pred, dataset="train")
        self.classifier.trainer.save(self.classifier.error_count, self.classifier.retrain_count)
        self._reset_eval()
        messagebox.showinfo("Éxito", "Imagen movida a entrenamiento (acierto).")

    def feedback_incorrect(self):
        if not self.eval_img_path:
            return
        self.classifier.move_to_mifoto(self.eval_img_path, dataset="test")
        self.classifier.learn_from_mistake(self.eval_img_path)
        self._reset_eval()
        messagebox.showinfo("Registrado", "Imagen movida a Test/mifoto (error).")

    def _reset_eval(self):
        self.eval_img_path = None
        self.eval_pred = None
        self.lbl_test_img.config(image="", text="Imagen de prueba")
        self.lbl_pred_img.config(image="", text="Imagen de predicción")
        self.btn_predict_eval.config(state=tk.DISABLED)
        self.btn_correct.config(state=tk.DISABLED)
        self.btn_incorrect.config(state=tk.DISABLED)
        self._clear_eval_results()

    def _clear_eval_results(self):
        self.lbl_pred_eval.config(text="Predicción: ---")
        self.lbl_conf_eval.config(text="Confianza: ---")
        self.lbl_medidas_eval.config(text="Medidas: ---")
        self.txt_info_eval.config(state=tk.NORMAL)
        self.txt_info_eval.delete("1.0", tk.END)
        self.txt_info_eval.config(state=tk.DISABLED)

    def _mostrar_imagen(self, img_rgb, label, texto=""):
        max_w, max_h = 400, 300
        h, w = img_rgb.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        im = Image.fromarray(img_resized)
        imgtk = ImageTk.PhotoImage(image=im)
        label.imgtk = imgtk
        label.config(image=imgtk, text=texto)

    def _get_sample_image(self, plant_class):
        return _pick_sample_image(self.classifier, plant_class)


class TestMenuPage(tk.Frame):
    """Menú del modo prueba (solo visualización, sin guardar datos)."""
    def __init__(self, parent, classifier):
        super().__init__(parent, bg="#1a1a2e")
        self.parent = parent

        title = tk.Label(self, text="Modo Prueba", font=("Segoe UI", 20, "bold"),
                         fg="#e94560", bg="#1a1a2e")
        title.pack(pady=40)

        btn_frame = tk.Frame(self, bg="#1a1a2e")
        btn_frame.pack()

        opciones = [
            ("🖼️  Usar imágenes", lambda: parent.show_frame(ImageTestPage)),
            ("📷  Usar cámara en vivo", self.start_camera),
            ("📊  Ver progreso del sistema", self.show_progress),
            ("📈  Monitor de entrenamientos", self.run_monitor),
            ("⬅  Volver", lambda: parent.show_frame(StartPage)),
        ]
        for texto, comando in opciones:
            btn = tk.Button(btn_frame, text=texto, font=("Segoe UI", 13),
                            bg="#0f3460", fg="white", activebackground="#e94560",
                            width=25, height=2, bd=0, cursor="hand2",
                            command=comando)
            btn.pack(pady=8)

    def start_camera(self):
        cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened() and cap.read()[0]:
                cameras.append(i)
            cap.release()
        if not cameras:
            messagebox.showwarning("Sin cámara", "No se detectaron cámaras disponibles.")
            return
        session = CameraSession(self.parent.classifier)
        # CameraSession.run() abre una ventana de matplotlib/Tk y bloquea
        # hasta que se cierra (plt.show(block=True)). Ocultamos la ventana
        # principal mientras tanto para no dejarla "congelada" en pantalla;
        # no se ejecuta en un hilo aparte porque Tkinter/TkAgg no son
        # seguros para usarse desde múltiples hilos a la vez.
        self.parent.withdraw()
        try:
            session.run(cameras[0])
        finally:
            self.parent.deiconify()

    def show_progress(self):
        win = tk.Toplevel(self)
        win.title("Progreso del Sistema")
        win.geometry("600x400")
        win.configure(bg="#1a1a2e")
        txt = tk.Text(win, bg="#0f0f23", fg="white", font=("Consolas", 10))
        txt.pack(fill=tk.BOTH, expand=True)

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        try:
            self.parent.classifier.show_learning_progress()
            output = buffer.getvalue()
        finally:
            sys.stdout = old_stdout
        txt.insert(tk.END, output)
        txt.config(state=tk.DISABLED)

    def run_monitor(self):
        # Abrir ventana del monitor integrada en la GUI
        MonitorWindow(self.parent)


class ImageTestPage(tk.Frame):
    """Prueba de imágenes sin guardar datos (solo visualización)."""
    def __init__(self, parent, classifier):
        super().__init__(parent, bg="#1a1a2e")
        self.classifier = classifier
        self.parent = parent
        self.current_image_path = None
        self.current_image_name = ""
        self.image_list = []
        self.mifoto_dir = os.path.join(self.classifier.test_folder, "mifoto")
        os.makedirs(self.mifoto_dir, exist_ok=True)

        main_frame = tk.Frame(self, bg="#1a1a2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.lbl_test_img = tk.Label(main_frame, bg="#0f0f23", text="Imagen de prueba",
                                     fg="white", font=("Segoe UI", 11))
        self.lbl_test_img.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.lbl_pred_img = tk.Label(main_frame, bg="#0f0f23", text="Imagen de predicción",
                                     fg="white", font=("Segoe UI", 11))
        self.lbl_pred_img.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        bottom = tk.Frame(self, bg="#16213e", height=180)
        bottom.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)

        self.lbl_nombre = tk.Label(bottom, text="Imagen: ---", font=("Segoe UI", 11, "bold"),
                                   fg="#88ddff", bg="#16213e")
        self.lbl_nombre.grid(row=0, column=0, padx=10, pady=2, sticky="w")

        self.lbl_pred = tk.Label(bottom, text="Predicción: ---", font=("Segoe UI", 12, "bold"),
                                 fg="white", bg="#16213e")
        self.lbl_pred.grid(row=1, column=0, padx=10, pady=2, sticky="w")
        self.lbl_conf = tk.Label(bottom, text="Confianza: ---", font=("Segoe UI", 11),
                                 fg="white", bg="#16213e")
        self.lbl_conf.grid(row=2, column=0, padx=10, pady=2, sticky="w")
        self.lbl_medidas = tk.Label(bottom, text="Medidas: ---", font=("Segoe UI", 11),
                                    fg="#ffcc44", bg="#16213e")
        self.lbl_medidas.grid(row=3, column=0, padx=10, pady=2, sticky="w")

        self.txt_info = tk.Text(bottom, height=6, width=35, font=("Segoe UI", 10),
                                bg="#0f0f23", fg="white", state=tk.DISABLED, relief=tk.FLAT)
        self.txt_info.grid(row=0, column=1, rowspan=4, padx=10, pady=5)

        ctrl_frame = tk.Frame(bottom, bg="#16213e")
        ctrl_frame.grid(row=4, column=0, columnspan=2, pady=10)

        self.lbl_count = tk.Label(ctrl_frame, text="Imágenes disponibles: 0",
                                  font=("Segoe UI", 10), fg="#88ddff", bg="#16213e")
        self.lbl_count.pack(side=tk.LEFT, padx=5)

        self.entry_num = tk.Entry(ctrl_frame, width=5, font=("Segoe UI", 10),
                                  bg="#0f0f23", fg="white", insertbackground="white")
        self.entry_num.pack(side=tk.LEFT, padx=5)
        # Enter en el campo de número también dispara la carga de la imagen
        self.entry_num.bind("<Return>", lambda event: self.load_image_by_number())

        self.btn_load_num = tk.Button(ctrl_frame, text="Cargar #", font=("Segoe UI", 10),
                                      bg="#0f3460", fg="white", command=self.load_image_by_number)
        self.btn_load_num.pack(side=tk.LEFT, padx=5)

        self.btn_load = tk.Button(ctrl_frame, text="Examinar...", font=("Segoe UI", 10),
                                  bg="#0f3460", fg="white", command=self.load_image_dialog)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.btn_predict = tk.Button(ctrl_frame, text="Predecir", font=("Segoe UI", 11, "bold"),
                                     bg="#00aa55", fg="white", state=tk.DISABLED,
                                     command=self.predict)
        self.btn_predict.pack(side=tk.LEFT, padx=5)

        btn_back = tk.Button(ctrl_frame, text="Volver", font=("Segoe UI", 11),
                             bg="#333344", fg="white",
                             command=lambda: parent.show_frame(TestMenuPage))
        btn_back.pack(side=tk.LEFT, padx=20)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        self.refresh_image_list()

    def refresh_image_list(self):
        if not os.path.exists(self.mifoto_dir):
            self.image_list = []
        else:
            self.image_list = [f for f in os.listdir(self.mifoto_dir)
                               if f.lower().endswith(IMAGE_EXTENSIONS)]
        self.image_list.sort()
        self.lbl_count.config(text=f"Imágenes disponibles: {len(self.image_list)}")

    def load_image_by_number(self):
        self.refresh_image_list()
        if not self.image_list:
            messagebox.showinfo("Sin imágenes", f"No hay imágenes en {self.mifoto_dir}")
            return
        try:
            num = int(self.entry_num.get().strip())
        except ValueError:
            messagebox.showerror("Entrada inválida", "Debes ingresar un número entero.")
            return
        if num < 1 or num > len(self.image_list):
            messagebox.showerror("Número fuera de rango",
                                 f"Ingresa un número entre 1 y {len(self.image_list)}")
            return
        filename = self.image_list[num - 1]
        file_path = os.path.join(self.mifoto_dir, filename)
        if not os.path.isfile(file_path):
            messagebox.showerror("Error", f"El archivo {filename} no existe.")
            return
        self._load_and_show_image(file_path)

    def load_image_dialog(self):
        os.makedirs(self.mifoto_dir, exist_ok=True)
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen de prueba",
            initialdir=self.mifoto_dir,
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return
        self._load_and_show_image(file_path)

    def _load_and_show_image(self, file_path):
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "No se pudo cargar la imagen.")
            return
        self.current_image_path = file_path
        self.current_image_name = os.path.basename(file_path)
        self.lbl_nombre.config(text=f"Imagen: {self.current_image_name}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._mostrar_imagen(img_rgb, self.lbl_test_img)
        self.btn_predict.config(state=tk.NORMAL)
        # Foco en el botón Predecir para que Enter dispare la predicción directamente
        self.btn_predict.focus_set()
        # Limpiar resultados anteriores
        self.lbl_pred.config(text="Predicción: ---")
        self.lbl_conf.config(text="Confianza: ---")
        self.lbl_medidas.config(text="Medidas: ---")
        self.txt_info.config(state=tk.NORMAL)
        self.txt_info.delete("1.0", tk.END)
        self.txt_info.config(state=tk.DISABLED)
        self.lbl_pred_img.config(image="", text="Imagen de predicción")
        self.refresh_image_list()

    def predict(self):
        if not self.current_image_path or not self.classifier.trainer.model:
            return
        pred, proba = self.classifier.predict_single_image(
            self.current_image_path, show_image=False
        )
        if pred is None:
            messagebox.showerror("Error", "No se pudo procesar la imagen.")
            return
        display = get_display_name(pred)
        self.lbl_pred.config(text=f"Predicción: {display}")
        self.lbl_conf.config(text=f"Confianza: {proba:.2%}")

        medidas_txt = calcular_medidas_imagen(self.current_image_path)
        self.lbl_medidas.config(text=medidas_txt)

        if pred in PLANT_DESCRIPTIONS:
            info = PLANT_DESCRIPTIONS[pred]
            txt = f"Nombre: {info['nombre_cientifico']}\nAltitud: {info['altitud_ideal']}\nUso: {info['uso']}"
        else:
            txt = "Sin información."
        self.txt_info.config(state=tk.NORMAL)
        self.txt_info.delete("1.0", tk.END)
        self.txt_info.insert(tk.END, txt)
        self.txt_info.config(state=tk.DISABLED)

        sample = self._get_sample_image(pred)
        if sample:
            sample_img = cv2.cvtColor(cv2.imread(sample), cv2.COLOR_BGR2RGB)
            self._mostrar_imagen(sample_img, self.lbl_pred_img, "")
        else:
            self.lbl_pred_img.config(image="", text="Sin ejemplo")

    def _mostrar_imagen(self, img_rgb, label, texto=""):
        max_w, max_h = 400, 300
        h, w = img_rgb.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_rgb, (new_w, new_h))
        im = Image.fromarray(img_resized)
        imgtk = ImageTk.PhotoImage(image=im)
        label.imgtk = imgtk
        label.config(image=imgtk, text=texto)

    def _get_sample_image(self, plant_class):
        return _pick_sample_image(self.classifier, plant_class)

# ============================================================
# Monitor de entrenamiento (ventana interna)
# ============================================================
class MonitorWindow(tk.Toplevel):
    """Ventana del monitor de entrenamiento (gráfica, dentro de la GUI)."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Monitor de Entrenamientos")
        self.geometry("900x700")
        self.configure(bg="#1a1a2e")

        try:
            from tools.training_monitor import load_summary
            self.summary = load_summary()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar summary.json:\n{e}", parent=self)
            self.destroy()
            return

        if not self.summary:
            messagebox.showinfo("Sin datos", "Aún no hay sesiones de entrenamiento registradas.", parent=self)
            self.destroy()
            return

        # Barra superior
        top_bar = tk.Frame(self, bg="#0f3460", height=40)
        top_bar.pack(fill=tk.X)

        tk.Label(top_bar, text=f"Sesiones registradas: {len(self.summary)}",
                 fg="white", bg="#0f3460", font=("Segoe UI", 11, "bold")).pack(side=tk.LEFT, padx=10)

        self.btn_plot_all = tk.Button(top_bar, text="Gráfico Resumen", command=self.plot_summary,
                                      bg="#e94560", fg="white", font=("Segoe UI", 10))
        self.btn_plot_all.pack(side=tk.RIGHT, padx=10, pady=5)

        # Lista de sesiones
        list_frame = tk.Frame(self, bg="#1a1a2e")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.sessions_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                           bg="#0f0f23", fg="white", font=("Consolas", 10),
                                           selectbackground="#e94560", selectforeground="white")
        self.sessions_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.sessions_listbox.yview)

        for i, s in enumerate(self.summary, 1):
            text = (f"#{i:<3} {s['timestamp'][9:]}  "
                    f"Acc: {s['best_val_acc']:.1f}%  "
                    f"Loss: {s['best_val_loss']:.4f}  "
                    f"Ep: {s['epochs_run']}  "
                    f"Clases: {s['num_classes']}  "
                    f"{', '.join(s.get('class_list', []))}")
            self.sessions_listbox.insert(tk.END, text)

        # Botones inferiores
        btn_frame = tk.Frame(self, bg="#1a1a2e")
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Ver detalle de la sesión seleccionada", command=self.view_session_detail,
                  bg="#0f3460", fg="white", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Actualizar lista", command=self.refresh_list,
                  bg="#0f3460", fg="white", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cerrar", command=self.destroy,
                  bg="#333344", fg="white", font=("Segoe UI", 10)).pack(side=tk.RIGHT, padx=20)

    def refresh_list(self):
        try:
            from tools.training_monitor import load_summary
            self.summary = load_summary()
        except:
            pass
        self.sessions_listbox.delete(0, tk.END)
        for i, s in enumerate(self.summary, 1):
            text = (f"#{i:<3} {s['timestamp'][9:]}  "
                    f"Acc: {s['best_val_acc']:.1f}%  "
                    f"Loss: {s['best_val_loss']:.4f}  "
                    f"Ep: {s['epochs_run']}  "
                    f"Clases: {s['num_classes']}  "
                    f"{', '.join(s.get('class_list', []))}")
            self.sessions_listbox.insert(tk.END, text)

    def view_session_detail(self):
        sel = self.sessions_listbox.curselection()
        if not sel:
            messagebox.showinfo("Selecciona una sesión", "Haz clic en una sesión de la lista.")
            return
        idx = sel[0]
        timestamp = self.summary[idx]['timestamp']
        try:
            from tools.training_monitor import load_session
            session = load_session(timestamp)
            if session:
                self._show_session_detail_window(session)
            else:
                messagebox.showerror("Error", f"No se encontró el archivo de sesión {timestamp}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _show_session_detail_window(self, session):
        detail_win = tk.Toplevel(self)
        detail_win.title(f"Sesión {session['timestamp']}")
        detail_win.geometry("800x500")
        detail_win.configure(bg="#1a1a2e")

        history = session.get('history', [])
        if not history:
            tk.Label(detail_win, text="Sin historial de epochs.", fg="white", bg="#1a1a2e",
                     font=("Segoe UI", 12)).pack(pady=20)
            return

        epochs = [e['epoch'] for e in history]
        tr_loss = [e['train_loss'] for e in history]
        val_loss = [e['val_loss'] for e in history]
        tr_acc = [e['train_acc'] for e in history]
        val_acc = [e['val_acc'] for e in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor('#1a1a2e')

        ax1.plot(epochs, tr_loss, 'o-', color="#4488ff", label="Train Loss")
        ax1.plot(epochs, val_loss, 's-', color="#ff6644", label="Val Loss")
        ax1.set_title("Loss", color="white")
        ax1.set_xlabel("Epoch", color="white")
        ax1.set_ylabel("Loss", color="white")
        ax1.tick_params(colors="white")
        ax1.legend(facecolor="#1a1a3e", edgecolor="#333355", labelcolor="white")
        ax1.set_facecolor("#0f0f23")
        ax1.spines[:].set_color("#333355")

        ax2.plot(epochs, tr_acc, 'o-', color="#4488ff", label="Train Acc")
        ax2.plot(epochs, val_acc, 's-', color="#00ff88", label="Val Acc")
        ax2.set_title("Accuracy", color="white")
        ax2.set_xlabel("Epoch", color="white")
        ax2.set_ylabel("Accuracy (%)", color="white")
        ax2.tick_params(colors="white")
        ax2.legend(facecolor="#1a1a3e", edgecolor="#333355", labelcolor="white")
        ax2.set_facecolor("#0f0f23")
        ax2.spines[:].set_color("#333355")

        canvas = FigureCanvasTkAgg(fig, master=detail_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def plot_summary(self):
        if not self.summary:
            return
        plot_win = tk.Toplevel(self)
        plot_win.title("Resumen de Entrenamientos")
        plot_win.geometry("900x600")
        plot_win.configure(bg="#1a1a2e")

        sessions = list(range(1, len(self.summary) + 1))
        val_accs = [s['best_val_acc'] for s in self.summary]
        val_losses = [s['best_val_loss'] for s in self.summary]
        epochs_run = [s['epochs_run'] for s in self.summary]
        num_classes = [s['num_classes'] for s in self.summary]

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.patch.set_facecolor('#1a1a2e')

        axes[0,0].bar(sessions, val_accs, color="#4488ff", edgecolor="#333355")
        axes[0,0].set_title("Val Accuracy por Sesión", color="white")
        axes[0,0].set_xlabel("Sesión", color="white")
        axes[0,0].set_ylabel("%", color="white")
        axes[0,0].tick_params(colors="white")
        axes[0,0].set_facecolor("#0f0f23")
        axes[0,0].spines[:].set_color("#333355")

        axes[0,1].plot(sessions, val_losses, 'o-', color="#ff6644")
        axes[0,1].set_title("Val Loss por Sesión", color="white")
        axes[0,1].set_xlabel("Sesión", color="white")
        axes[0,1].set_ylabel("Loss", color="white")
        axes[0,1].tick_params(colors="white")
        axes[0,1].set_facecolor("#0f0f23")
        axes[0,1].spines[:].set_color("#333355")

        axes[1,0].bar(sessions, epochs_run, color="#ffcc44", edgecolor="#333355")
        axes[1,0].set_title("Epochs Ejecutados", color="white")
        axes[1,0].set_xlabel("Sesión", color="white")
        axes[1,0].set_ylabel("Epochs", color="white")
        axes[1,0].tick_params(colors="white")
        axes[1,0].set_facecolor("#0f0f23")
        axes[1,0].spines[:].set_color("#333355")

        axes[1,1].step(sessions, num_classes, where="mid", color="#88ddff", linewidth=2)
        axes[1,1].scatter(sessions, num_classes, color="#88ddff", s=40)
        axes[1,1].set_title("Clases Activas", color="white")
        axes[1,1].set_xlabel("Sesión", color="white")
        axes[1,1].set_ylabel("Nº Clases", color="white")
        axes[1,1].tick_params(colors="white")
        axes[1,1].set_facecolor("#0f0f23")
        axes[1,1].spines[:].set_color("#333355")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=plot_win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)


# ============================================================
# Función de arranque
# ============================================================
def main_gui():
    app = CoffeeApp()
    app.mainloop()

if __name__ == "__main__":
    main_gui()