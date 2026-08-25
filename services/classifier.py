# ============================================================
# services/classifier.py    --  Orquestador principal (con fusión CNN + botánicas)
# ============================================================
import os
import random
import shutil
import warnings
import joblib
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from config.settings import (
    ERROR_THRESHOLD_FOR_RETRAIN, PLANT_DESCRIPTIONS, get_display_name,
    CNN_IMAGE_SIZE, IMAGE_EXTENSIONS
)
from data.loader import load_images_from_folder, load_single_image
from training.trainer import ModelTrainer
from features.botanical_features import (
    get_contour_and_mask, extract_botanical_features,
    compute_vein_geometry,
)

# Por debajo de esta confianza, _infer() devuelve "DESCONOCIDO" en vez de la
# clase con mayor probabilidad: evita presentar como certera una predicción
# que en realidad es una elección débil entre varias especies.
CONFIDENCE_THRESHOLD = 0.35

class CoffeePlantClassifier:
    def __init__(self, train_folder: str, test_folder: str, results_folder: str):
        self.train_folder = os.path.abspath(train_folder)
        self.test_folder = os.path.abspath(test_folder)
        self.results_folder = os.path.abspath(results_folder)

        self.trainer = ModelTrainer(results_folder)
        self.error_count = 0
        self.retrain_count = 0
        self.idx_to_class = {}
        self._bot_feat_dim = 0

        # Constantes de normalización de ImageNet (mismas que en entrenamiento):
        # se precomputan una sola vez en vez de reconstruirlas en cada predicción.
        device = self.trainer.device
        self._norm_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

        self._setup_folders()
        self._load_counters()

    def _setup_folders(self):
        for base in (self.train_folder, self.test_folder, self.results_folder):
            os.makedirs(base, exist_ok=True)
        os.makedirs(os.path.join(self.results_folder, "logs"), exist_ok=True)
        for plant in PLANT_DESCRIPTIONS:
            os.makedirs(os.path.join(self.train_folder, plant), exist_ok=True)
            os.makedirs(os.path.join(self.test_folder, plant), exist_ok=True)

    def _load_counters(self):
        path = os.path.join(self.results_folder, "training_info.joblib")
        if os.path.exists(path):
            try:
                info = joblib.load(path)
                self.error_count = info.get("error_count", 0)
                self.retrain_count = info.get("retrain_count", 0)
                self._bot_feat_dim = info.get("bot_feat_dim", 0)
                print(f"Contadores: errores={self.error_count}, reentrenamientos={self.retrain_count}")
            except Exception:
                pass

    def _rebuild_idx_to_class(self):
        class_list = getattr(self.trainer, "class_list", None)
        if class_list:
            self.idx_to_class = {idx: cls for idx, cls in enumerate(class_list)}
        else:
            self.idx_to_class = {idx: cls for idx, cls in enumerate(PLANT_DESCRIPTIONS.keys())}

    def load_models(self) -> bool:
        """
        Carga el mejor modelo guardado (por precisión) si existe.
        Si no, carga el modelo principal (model_weights.pth) como fallback.
        """
        # Intentar cargar el mejor modelo de la lista de los 3 mejores
        if self.trainer.models_metadata:
            ok = self.trainer.load_best_model()
            if ok:
                self._rebuild_idx_to_class()
                if self.trainer.model is not None:
                    self._bot_feat_dim = self.trainer._bot_feat_dim
                print("✅ Cargado el mejor modelo global (por precisión).")
                return True

        # Fallback: cargar el modelo principal (model_weights.pth)
        ok, self.error_count, self.retrain_count = self.trainer.load()
        if ok:
            self._rebuild_idx_to_class()
            if self.trainer.model is not None:
                self._bot_feat_dim = self.trainer._bot_feat_dim
        return ok

    def train_and_save(self) -> bool:
        print("\nCargando imágenes de entrenamiento...")
        X, y = load_images_from_folder(self.train_folder, target_size=CNN_IMAGE_SIZE)
        fine_tune = self.trainer.model is not None
        if self.trainer.train(X, y, fine_tune=fine_tune):
            saved = self.trainer.save(self.error_count, self.retrain_count)
            if saved:
                self._rebuild_idx_to_class()
                self._bot_feat_dim = self.trainer._bot_feat_dim
            return saved
        return False

    def _extract_botanical_from_img(self, img: np.ndarray) -> np.ndarray:
        """Extrae características botánicas de una imagen float32 [0,1] RGB."""
        img_uint8 = (img * 255).astype(np.uint8)
        mask, contour = get_contour_and_mask(img_uint8)
        if contour is None or len(contour) == 0:
            return np.zeros(self._bot_feat_dim, dtype=np.float32)
        bot = extract_botanical_features(img_uint8, mask, contour)
        if len(bot) < self._bot_feat_dim:
            pad = np.zeros(self._bot_feat_dim - len(bot))
            bot = np.concatenate([bot, pad])
        elif len(bot) > self._bot_feat_dim:
            bot = bot[:self._bot_feat_dim]
        return bot.astype(np.float32)

    def _infer(self, img: np.ndarray) -> tuple[str, float]:
        """
        img: numpy array (H, W, C) float32 en [0,1]
        """
        self.trainer.model.eval()
        bot_feat = self._extract_botanical_from_img(img)
        bot_tensor = torch.from_numpy(bot_feat).unsqueeze(0).to(self.trainer.device)

        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.trainer.device)
        # Normalización con las mismas constantes de ImageNet usadas en entrenamiento
        tensor = (tensor - self._norm_mean) / self._norm_std

        with torch.no_grad():
            logits = self.trainer.model(tensor, bot_tensor)
            probs = torch.softmax(logits, dim=1)
            proba, idx = torch.max(probs, 1)

        conf = proba.item()
        if conf < CONFIDENCE_THRESHOLD:
            return "DESCONOCIDO", conf
        cls = self.idx_to_class.get(idx.item(), "DESCONOCIDO")
        return cls, conf

    def predict_single_image(self, img_path: str, true_label: str | None = None,
                             show_image: bool = True) -> tuple[str | None, float | None]:
        if self.trainer.model is None:
            print("No hay un modelo entrenado/cargado. Entrena o carga uno primero.")
            return None, None
        if not os.path.exists(img_path):
            print(f"Archivo no encontrado: {img_path}")
            return None, None
        img = load_single_image(img_path, target_size=CNN_IMAGE_SIZE)
        if img is None:
            return None, None
        pred, proba = self._infer(img)
        self._print_result(img_path, pred, proba, true_label)
        if show_image:
            self._display_result(img_path, pred, proba, true_label)
        if true_label is not None:
            if pred != true_label:
                self.learn_from_mistake(img_path, true_label)
            else:
                self.trainer.save(self.error_count, self.retrain_count)
        return pred, proba

    def analyze_for_explanation(self, img_path: str) -> dict | None:
        """
        Corre una predicción con todo lo necesario para explicarla: Grad-CAM
        sobre la CNN (qué mira la red en la imagen) y la geometría real detrás
        de cada rasgo botánico (contorno, venas detectadas, punto de
        estípulas), además de la probabilidad de las 5 clases. Pensado para
        la ventana interactiva de "Predecir una imagen (sin guardar)".
        """
        if self.trainer.model is None:
            print("No hay un modelo entrenado/cargado. Entrena o carga uno primero.")
            return None
        if not os.path.exists(img_path):
            print(f"Archivo no encontrado: {img_path}")
            return None
        img = load_single_image(img_path, target_size=CNN_IMAGE_SIZE)
        if img is None:
            return None
        img_uint8 = (img * 255).astype(np.uint8)

        bot_feat = self._extract_botanical_from_img(img)
        bot_tensor = torch.from_numpy(bot_feat).unsqueeze(0).to(self.trainer.device)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.trainer.device)
        tensor_n = (tensor - self._norm_mean) / self._norm_std

        cam, probs, pred_idx = self._compute_gradcam(tensor_n, bot_tensor)
        conf = float(probs[pred_idx])
        pred = self.idx_to_class.get(pred_idx, "DESCONOCIDO") if conf >= CONFIDENCE_THRESHOLD else "DESCONOCIDO"

        mask, contour = get_contour_and_mask(img_uint8)
        has_leaf = contour is not None and len(contour) > 0
        if has_leaf:
            edges_masked, ang_deg = compute_vein_geometry(img_uint8, mask)
            M = cv2.moments(contour)
            centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] else None
        else:
            centroid = None
            edges_masked = np.zeros(img_uint8.shape[:2], dtype=np.uint8)
            ang_deg = np.zeros(img_uint8.shape[:2], dtype=np.float32)

        names = [get_display_name(self.idx_to_class.get(i, "?")) for i in range(len(probs))]
        class_keys = [self.idx_to_class.get(i, "?") for i in range(len(probs))]

        return dict(
            img_filename=os.path.basename(img_path), img_uint8=img_uint8, cam=cam,
            mask=mask, contour=contour, centroid=centroid,
            edges=edges_masked, ang_deg=ang_deg, bot_feat=bot_feat,
            probs=probs, pred_idx=pred_idx, names=names, class_keys=class_keys,
            pred=pred, conf=conf,
        )

    def _compute_gradcam(self, tensor_n: torch.Tensor, bot_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray, int]:
        """Grad-CAM sobre la última capa convolucional del backbone (layer4):
        qué región de la imagen pesó más para llegar a la predicción."""
        model = self.trainer.model
        model.eval()
        activations, gradients = {}, {}
        h1 = model.backbone.layer4.register_forward_hook(lambda m, i, o: activations.__setitem__("v", o))
        h2 = model.backbone.layer4.register_full_backward_hook(lambda m, gi, go: gradients.__setitem__("v", go[0]))
        try:
            model.zero_grad()
            logits = model(tensor_n, bot_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(logits, dim=1).item())
            with warnings.catch_warnings():
                # El hook de layer4 no necesita el gradiente respecto a la
                # entrada (solo respecto a su salida, que es justo lo que
                # usa Grad-CAM); PyTorch avisa igual porque la imagen no
                # tiene requires_grad. El resultado es correcto, así que se
                # silencia solo esta advertencia puntual, no las demás.
                warnings.filterwarnings(
                    "ignore",
                    message="Full backward hook is firing when gradients are computed",
                    category=UserWarning,
                )
                logits[0, pred_idx].backward()
        finally:
            h1.remove()
            h2.remove()

        act = activations["v"][0]
        grad = gradients["v"][0]
        weights = grad.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * act).sum(0))
        cam = cam / (cam.max() + 1e-8)
        h, w = tensor_n.shape[-2:]
        cam_np = cv2.resize(cam.detach().cpu().numpy().astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        return np.clip(cam_np, 0, 1), probs.detach().cpu().numpy(), pred_idx

    def predict_frame(self, frame: np.ndarray) -> tuple[str | None, float | None]:
        if self.trainer.model is None:
            return None, None
        try:
            img = cv2.resize(frame, CNN_IMAGE_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            return self._infer(img)
        except Exception:
            return None, None

    def learn_from_mistake(self, img_path: str, correct_label: str | None = None):
        self.error_count += 1
        print(f"Error registrado ({self.error_count}/{ERROR_THRESHOLD_FOR_RETRAIN})")
        self.trainer.save(self.error_count, self.retrain_count)
        if self.error_count >= ERROR_THRESHOLD_FOR_RETRAIN:
            self._retrain()

    def _retrain(self):
        print("\nReentrenando por errores consecutivos...")
        X, y = load_images_from_folder(self.train_folder, target_size=CNN_IMAGE_SIZE)
        if self.trainer.train(X, y, fine_tune=True):
            self.retrain_count += 1
            self.error_count = 0
            self.trainer.save(self.error_count, self.retrain_count)
            self._rebuild_idx_to_class()
            self._bot_feat_dim = self.trainer._bot_feat_dim
            print(f"Reentrenamiento #{self.retrain_count} completado")

    def _move_to_folder(self, img_path: str, dest_dir: str) -> bool:
        os.makedirs(dest_dir, exist_ok=True)
        name, ext = os.path.splitext(os.path.basename(img_path))
        dest = os.path.join(dest_dir, os.path.basename(img_path))
        counter = 1
        while os.path.exists(dest):
            dest = os.path.join(dest_dir, f"{name}_{counter}{ext}")
            counter += 1
        try:
            shutil.move(img_path, dest)
            print(f"✅ Movida a: {dest}")
            return True
        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    def move_to_class_folder(self, img_path: str, plant_class: str, dataset: str = "train") -> bool:
        base = self.train_folder if dataset == "train" else self.test_folder
        return self._move_to_folder(img_path, os.path.join(base, plant_class))

    def move_to_mifoto(self, img_path: str, dataset: str = "train") -> bool:
        base = self.train_folder if dataset == "train" else self.test_folder
        return self._move_to_folder(img_path, os.path.join(base, "mifoto"))

    def show_learning_progress(self):
        print("\n" + "="*50)
        print(" ESTADO DEL SISTEMA")
        print("="*50)
        active = set(self.idx_to_class.values())
        for plant in PLANT_DESCRIPTIONS:
            if plant == "mifoto":
                continue
            folder = os.path.join(self.train_folder, plant)
            count = len([f for f in os.listdir(folder)
                         if f.lower().endswith(IMAGE_EXTENSIONS)]) if os.path.exists(folder) else 0
            status = "" if plant in active else " (sin modelo)"
            print(f"   {status} {get_display_name(plant)}: {count} imágenes")
        print(f"\n   Errores recientes:    {self.error_count}")
        print(f"   Reentrenamientos:     {self.retrain_count}")
        print(f"   Dispositivo:          {self.trainer.device}")
        print(f"   Clases en modelo:     {len(self.idx_to_class)}")

    @staticmethod
    def get_display_name(plant_key: str) -> str:
        return get_display_name(plant_key)

    def _print_result(self, img_path, pred, proba, true_label):
        print("\n=== Resultado ===")
        print(f"  Imagen:     {os.path.basename(img_path)}")
        print(f"  Predicción: {get_display_name(pred)} ({proba:.2%})")
        if true_label:
            status = " CORRECTO" if pred == true_label else " INCORRECTO"
            print(f"  Real:       {get_display_name(true_label)}    {status}")
        if pred in PLANT_DESCRIPTIONS:
            info = PLANT_DESCRIPTIONS[pred]
            print(f"\n  Nombre científico: {info['nombre_cientifico']}")
            print(f"  Altitud ideal:     {info['altitud_ideal']}")
            print(f"  Uso:               {info['uso']}")

    def _display_result(self, img_path, pred, proba, true_label):
        img_test = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        file_name = os.path.basename(img_path)
        first_letter = file_name[0] if file_name else "?"
        subtitle = f"\nReal: {get_display_name(true_label)}" if true_label else ""
        axes[0].imshow(img_test)
        axes[0].set_title(f"{first_letter}{subtitle}", pad=10)
        axes[0].axis("off")
        sample_path = self._get_sample_image(pred)
        if sample_path:
            axes[1].imshow(cv2.cvtColor(cv2.imread(sample_path), cv2.COLOR_BGR2RGB))
        else:
            axes[1].text(0.5, 0.5, "Sin imagen de ejemplo", ha="center", va="center")
        axes[1].set_title(f"Predicción: {get_display_name(pred)}\n{proba:.1%}", pad=10)
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()
        # Sin esto, cada predicción deja su figura viva en el registro interno
        # de matplotlib aunque cierres la ventana; tras muchas predicciones
        # seguidas en consola, los recursos de Tkinter acumulados terminan
        # rompiendo la creación de la siguiente ventana (o dejándola colgada).
        plt.close(fig)

    def _get_sample_image(self, plant: str) -> str | None:
        folder = os.path.join(self.train_folder, plant)
        if not os.path.exists(folder):
            return None
        files = [f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTENSIONS)]
        return os.path.join(folder, random.choice(files)) if files else None