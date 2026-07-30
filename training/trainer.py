# ============================================================
# training/trainer.py    --  Entrenamiento con fusión CNN + características botánicas
#                          + Gestión de múltiples modelos (mejores 3)
# ============================================================
import os
import json
import time
import threading
import traceback
import shutil
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import joblib

from config.settings import (
    BATCH_SIZE, CNN_IMAGE_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, PLANT_DESCRIPTIONS, DROPOUT_RATE
)
from models.fusion_model import FusionCoffeeModel
from features.botanical_features import (
    get_contour_and_mask, estimate_base_point, extract_botanical_features,
    BOTANICAL_FEATURE_DIM,
)
import cv2


class CoffeePlantDataset(Dataset):
    """Dataset para imágenes, características botánicas y etiquetas."""
    def __init__(self, images: np.ndarray, bot_feats: np.ndarray, labels: np.ndarray, transform=None):
        self.images = torch.from_numpy(images).float()
        self.bot_feats = torch.from_numpy(bot_feats).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].permute(2, 0, 1)
        if self.transform:
            img = self.transform(img)
        return img, self.bot_feats[idx], self.labels[idx]


class ModelTrainer:
    def __init__(self, results_folder: str):
        self.results_folder = results_folder
        self.models_folder = os.path.join(results_folder, "models")
        os.makedirs(self.models_folder, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: FusionCoffeeModel | None = None
        self.class_list: list[str] = []
        self._session_log = []
        self._bot_feat_dim = 0
        self._best_val_acc_global = 0.0   # mejor precisión de validación global

        # Transformaciones
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(CNN_IMAGE_SIZE[0], scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.01),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.val_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.plot_enabled = False
        self.epoch_metrics = []
        self._plot_keepalive_thread = None
        self._plot_keepalive_stop = None

        # Cargar metadatos de modelos existentes
        self._load_models_metadata()

    # ------------------------------------------------------------
    # Gestión de metadatos de modelos
    # ------------------------------------------------------------
    def _load_models_metadata(self):
        """Carga el archivo de metadatos de modelos guardados."""
        self.models_metadata = []
        meta_path = os.path.join(self.models_folder, "models_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.models_metadata = json.load(f)
            except:
                self.models_metadata = []
        # Ordenar por precisión descendente
        self.models_metadata.sort(key=lambda x: x.get("val_acc", 0), reverse=True)
        if self.models_metadata:
            self._best_val_acc_global = self.models_metadata[0].get("val_acc", 0.0)

    def _save_models_metadata(self):
        """Guarda el archivo de metadatos de modelos."""
        meta_path = os.path.join(self.models_folder, "models_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.models_metadata, f, indent=2)

    def _add_model_to_metadata(self, model_info: dict):
        """
        Añade un modelo a la lista de metadatos y mantiene solo los 3 mejores.
        model_info: diccionario con 'timestamp', 'val_acc', 'val_loss', 'epochs', 'class_list', 'file_name'
        """
        # Añadir el nuevo
        self.models_metadata.append(model_info)
        # Ordenar por val_acc descendente
        self.models_metadata.sort(key=lambda x: x.get("val_acc", 0), reverse=True)
        # Mantener solo los 3 mejores
        self.models_metadata = self.models_metadata[:3]
        self._save_models_metadata()
        # Actualizar mejor global
        if self.models_metadata:
            self._best_val_acc_global = self.models_metadata[0].get("val_acc", 0.0)

    def list_saved_models(self) -> list:
        """Devuelve la lista de modelos guardados (metadatos)."""
        return self.models_metadata

    def print_models_list(self):
        """Imprime en consola la lista de modelos guardados."""
        if not self.models_metadata:
            print("\n📭 No hay modelos guardados todavía.")
            return
        print("\n" + "="*80)
        print("   MODELOS GUARDADOS (mejores 3 por precisión)")
        print("="*80)
        for i, meta in enumerate(self.models_metadata, 1):
            print(f"{i}. {meta.get('timestamp', '')}  |  Val Acc: {meta.get('val_acc', 0):.1f}%  |  Val Loss: {meta.get('val_loss', 0):.4f}  |  Épocas: {meta.get('epochs', 0)}  |  Clases: {len(meta.get('class_list', []))}")
        print("="*80)

    def load_model_by_index(self, index: int) -> bool:
        """
        Carga un modelo específico de la lista de metadatos (por índice).
        Si index=0 carga el mejor, index=1 el segundo, etc.
        """
        if not self.models_metadata:
            print("❌ No hay modelos guardados.")
            return False
        if index < 0 or index >= len(self.models_metadata):
            print(f"❌ Índice inválido. Debe ser entre 0 y {len(self.models_metadata)-1}")
            return False
        meta = self.models_metadata[index]
        file_name = meta.get("file_name")
        if not file_name:
            print("❌ Metadato incompleto: falta file_name.")
            return False
        model_path = os.path.join(self.models_folder, file_name)
        if not os.path.exists(model_path):
            print(f"❌ Archivo de modelo no encontrado: {model_path}")
            return False
        try:
            num_classes = len(meta.get("class_list", []))
            bot_feat_dim = meta.get("bot_feat_dim", BOTANICAL_FEATURE_DIM)
            self._bot_feat_dim = bot_feat_dim
            self.class_list = meta.get("class_list", [])
            self.model = FusionCoffeeModel(num_classes=num_classes, bot_feat_dim=bot_feat_dim, pretrained=True).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ Modelo cargado: {meta.get('timestamp', '')} (Val Acc: {meta.get('val_acc', 0):.1f}%)")
            return True
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False

    def load_best_model(self) -> bool:
        """Carga el mejor modelo guardado (el primero de la lista)."""
        if not self.models_metadata:
            print("❌ No hay modelos guardados.")
            return False
        return self.load_model_by_index(0)

    # ------------------------------------------------------------
    # Métodos de entrenamiento
    # ------------------------------------------------------------
    def _start_plot_keepalive(self):
        """Sigue procesando los eventos de la ventana de entrenamiento tras
        terminar las épocas. Sin esto, nadie vuelve a bombear su cola de
        mensajes de Windows: el sistema operativo la marca "no responde" en
        cuanto la tocas, y si fuerzas el cierre desde ese diálogo, Windows
        termina TODO el proceso (no solo la ventana)."""
        if self.fig is None:
            return
        self._plot_keepalive_stop = threading.Event()
        fig, stop_event = self.fig, self._plot_keepalive_stop

        def _pump():
            while not stop_event.is_set() and plt.fignum_exists(fig.number):
                try:
                    fig.canvas.flush_events()
                except Exception:
                    break
                time.sleep(0.15)

        self._plot_keepalive_thread = threading.Thread(target=_pump, daemon=True)
        self._plot_keepalive_thread.start()

    def _stop_plot_keepalive(self):
        if self._plot_keepalive_stop is not None:
            self._plot_keepalive_stop.set()
        if self._plot_keepalive_thread is not None and self._plot_keepalive_thread.is_alive():
            self._plot_keepalive_thread.join(timeout=1.0)
        self._plot_keepalive_thread = None
        self._plot_keepalive_stop = None

    def _init_plot(self):
        # Nueva sesión de entrenamiento: si quedaba un hilo de la ronda
        # anterior bombeando eventos de esa figura, hay que pararlo antes de
        # reusar o reemplazar self.fig.
        self._stop_plot_keepalive()
        if self.fig is not None and plt.fignum_exists(self.fig.number):
            self.plot_enabled = True
            self.epoch_metrics = []
            return
        if self.fig is not None:
            plt.close(self.fig)
        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 4))
        self.fig.canvas.manager.set_window_title('Entrenamiento en tiempo real')
        self.plot_enabled = True
        self.epoch_metrics = []
        print("📊 Ventana de gráficas abierta.")

    def _update_plot(self, epoch, tr_loss, val_loss, tr_acc, val_acc, lr, avg_grad, class_acc):
        if not self.plot_enabled:
            return
        self.epoch_metrics.append((tr_loss, val_loss, tr_acc, val_acc, lr, avg_grad, class_acc))
        try:
            if self.fig is None or not plt.fignum_exists(self.fig.number):
                self.plot_enabled = False
                self.fig = None
                return
            epochs = range(1, len(self.epoch_metrics) + 1)
            train_loss = [m[0] for m in self.epoch_metrics]
            val_loss   = [m[1] for m in self.epoch_metrics]
            train_acc  = [m[2] for m in self.epoch_metrics]
            val_acc_vals = [m[3] for m in self.epoch_metrics]
            lr_vals    = [m[4] for m in self.epoch_metrics]
            avg_grad_vals = [m[5] for m in self.epoch_metrics]

            self.ax1.clear()
            self.ax1.plot(epochs, train_loss, label='Train Loss', color='#4488ff', lw=2)
            self.ax1.plot(epochs, val_loss, label='Val Loss', color='#ff6644', lw=2)
            self.ax1.legend()
            self.ax1.set_title('Pérdida (Loss)')
            self.ax1.set_xlabel('Época')
            self.ax1.grid(True, alpha=0.3)
            self.ax1.set_facecolor('#0f0f23')
            self.ax1.tick_params(colors='white')
            self.ax1.spines[:].set_color('#333355')

            self.ax2.clear()
            self.ax2.plot(epochs, train_acc, label='Train Acc', color='#4488ff', lw=2)
            self.ax2.plot(epochs, val_acc_vals, label='Val Acc', color='#00ff88', lw=2)
            best_val = max(val_acc_vals) if val_acc_vals else 0
            best_ep = val_acc_vals.index(best_val) + 1 if val_acc_vals else 0
            self.ax2.annotate(f'Mejor: {best_val:.1f}%', 
                            xy=(best_ep, best_val),
                            xytext=(best_ep + 0.5, best_val - 5),
                            color='white', fontsize=8,
                            arrowprops=dict(arrowstyle='->', color='white', lw=1))
            self.ax2.legend()
            self.ax2.set_title('Precisión (Accuracy %)')
            self.ax2.set_xlabel('Época')
            self.ax2.grid(True, alpha=0.3)
            self.ax2.set_facecolor('#0f0f23')
            self.ax2.tick_params(colors='white')
            self.ax2.spines[:].set_color('#333355')

            self.ax3.clear()
            self.ax3.plot(epochs, lr_vals, label='LR', color='#ffaa44', lw=2, marker='o', ms=3)
            ax3b = self.ax3.twinx()
            ax3b.plot(epochs, avg_grad_vals, label='Grad. Prom.', color='#ff66aa', lw=1.5, linestyle='--')
            ax3b.set_ylabel('Gradiente promedio', color='#ff66aa')
            ax3b.tick_params(axis='y', labelcolor='#ff66aa')
            ax3b.spines['right'].set_color('#ff66aa')
            self.ax3.set_yscale('log')
            self.ax3.set_ylabel('Learning Rate (log)', color='#ffaa44')
            self.ax3.set_xlabel('Época')
            self.ax3.legend(loc='upper left')
            self.ax3.grid(True, alpha=0.3)
            self.ax3.set_facecolor('#0f0f23')
            self.ax3.tick_params(colors='white')
            self.ax3.spines[:].set_color('#333355')

            if class_acc:
                worst_class_idx = min(class_acc, key=class_acc.get)
                worst_acc = class_acc[worst_class_idx]
                worst_name = self.class_list[worst_class_idx] if self.class_list else "?"
                self.ax2.set_title(f'Precisión - Peor: {worst_name} ({worst_acc:.1f}%)', color='#ff8888')

            self.fig.tight_layout()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception as e:
            self.plot_enabled = False
            if self.fig:
                plt.close(self.fig)
                self.fig = None

    def _train_epoch(self, loader, criterion, optimizer):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        total_grad = 0.0
        num_params = 0
        for X_b, bot_b, y_b in loader:
            X_b, bot_b, y_b = X_b.to(self.device), bot_b.to(self.device), y_b.to(self.device)
            optimizer.zero_grad()
            out = self.model(X_b, bot_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
            for param in self.model.parameters():
                if param.grad is not None:
                    total_grad += param.grad.abs().mean().item()
                    num_params += 1
        avg_grad = total_grad / num_params if num_params > 0 else 0.0
        return total_loss / len(loader), 100.0 * correct / total, avg_grad

    def _val_epoch(self, loader, criterion):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        class_correct = {}
        class_total = {}
        with torch.no_grad():
            for X_b, bot_b, y_b in loader:
                X_b, bot_b, y_b = X_b.to(self.device), bot_b.to(self.device), y_b.to(self.device)
                out = self.model(X_b, bot_b)
                loss = criterion(out, y_b)
                total_loss += loss.item()
                preds = out.argmax(1)
                correct += (preds == y_b).sum().item()
                total += y_b.size(0)
                for i in range(y_b.size(0)):
                    label = y_b[i].item()
                    if label not in class_total:
                        class_total[label] = 0
                        class_correct[label] = 0
                    class_total[label] += 1
                    if preds[i] == label:
                        class_correct[label] += 1
        class_acc = {}
        for label in class_total:
            class_acc[label] = 100.0 * class_correct[label] / class_total[label]
        return total_loss / len(loader), 100.0 * correct / total, class_acc

    # ------------------------------------------------------------
    # Entrenamiento principal (con guardado de mejores modelos)
    # ------------------------------------------------------------
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              validation_split: float = 0.2, fine_tune: bool = False,
              train_folder: str | None = None) -> bool:
        try:
            self._session_log = []
            unique, counts = np.unique(y_train, return_counts=True)
            valid = [c for c, n in zip(unique, counts) if n >= 3]
            if len(valid) < 2:
                raise ValueError("Se necesitan al menos 2 clases con mínimo 3 muestras.")
            mask = np.isin(y_train, valid)
            X_img = X_train[mask]
            y_raw = y_train[mask]
            self.class_list = sorted(set(valid))
            label_to_idx = {cls: idx for idx, cls in enumerate(self.class_list)}
            y = np.array([label_to_idx[lbl] for lbl in y_raw])

            # Extraer características botánicas
            print("Extrayendo características botánicas para entrenamiento...")
            sample_img = X_img[0]
            sample_img_uint8 = (sample_img * 255).astype(np.uint8)
            mask_s, contour_s = get_contour_and_mask(sample_img_uint8)
            if contour_s is not None and len(contour_s) > 0:
                base_point = estimate_base_point(contour_s)
                sample_bot = extract_botanical_features(sample_img_uint8, mask_s, contour_s, base_point)
                self._bot_feat_dim = len(sample_bot)
            else:
                self._bot_feat_dim = BOTANICAL_FEATURE_DIM

            bot_feats = []
            for img in X_img:
                img_uint8 = (img * 255).astype(np.uint8)
                mask, contour = get_contour_and_mask(img_uint8)
                if contour is None or len(contour) == 0:
                    bot_feats.append(np.zeros(self._bot_feat_dim, dtype=np.float32))
                    continue
                base_point = estimate_base_point(contour)
                bot = extract_botanical_features(img_uint8, mask, contour, base_point)
                if len(bot) < self._bot_feat_dim:
                    pad = np.zeros(self._bot_feat_dim - len(bot))
                    bot = np.concatenate([bot, pad])
                elif len(bot) > self._bot_feat_dim:
                    bot = bot[:self._bot_feat_dim]
                bot_feats.append(bot)
            bot_feats = np.array(bot_feats, dtype=np.float32)

            # Split
            X_train_img, X_val_img, y_train_labels, y_val_labels, bot_train, bot_val = train_test_split(
                X_img, y, bot_feats, test_size=0.3, stratify=y, random_state=42
            )

            train_dataset = CoffeePlantDataset(X_train_img, bot_train, y_train_labels, transform=self.train_transform)
            val_dataset   = CoffeePlantDataset(X_val_img,   bot_val,   y_val_labels,   transform=self.val_transform)
            # Si el último batch quedara con una sola muestra, el BatchNorm1d de
            # bot_norm no podría calcular varianza y fallaría en entrenamiento.
            # Solo se descarta ese batch cuando sobra exactamente 1 imagen y aun
            # así queda al menos un batch completo (nunca se queda en 0 batches).
            remainder = len(train_dataset) % BATCH_SIZE
            drop_last = remainder == 1 and len(train_dataset) > BATCH_SIZE
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=drop_last)
            val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

            num_classes = len(self.class_list)

            # El aprendizaje activo puede reentrenar sobre un conjunto de clases
            # o una dimensión de features botánicas distinta a la del modelo
            # cargado; hay que reconstruir classifier/bot_norm si no coinciden.
            if fine_tune and self.model is not None:
                print("\n📌 Fine‑tuning desde modelo existente...")
                expected_input_dim = 512 + self._bot_feat_dim
                current_input_dim = self.model.classifier[0].in_features
                current_output_dim = self.model.classifier[-1].out_features

                if current_input_dim != expected_input_dim or current_output_dim != num_classes:
                    print(f"   ⚠️ Reconstruyendo clasificador: entrada {current_input_dim} → {expected_input_dim}, "
                          f"clases {current_output_dim} → {num_classes}")
                    self.model.classifier = nn.Sequential(
                        nn.Linear(expected_input_dim, 256),
                        nn.ReLU(),
                        nn.Dropout(DROPOUT_RATE),
                        nn.Linear(256, num_classes)
                    )
                    if self.model.bot_norm.num_features != self._bot_feat_dim:
                        self.model.bot_norm = nn.BatchNorm1d(self._bot_feat_dim)
                    self.model = self.model.to(self.device)
                else:
                    if current_output_dim != num_classes:
                        self.model.classifier[-1] = nn.Linear(256, num_classes)
                        self.model = self.model.to(self.device)

                lr = LEARNING_RATE * 0.1
            else:
                print("\n🏗️ Construyendo modelo de fusión (ResNet18 + características botánicas)...")
                self.model = FusionCoffeeModel(num_classes=num_classes, bot_feat_dim=self._bot_feat_dim, pretrained=True).to(self.device)
                lr = LEARNING_RATE

            self._init_plot()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

            best_val_loss = float('inf')
            best_val_acc = 0.0
            patience = 0
            best_state = None

            print("\n" + "="*100)
            print("   🚀 ENTRENAMIENTO DEL MODELO DE FUSIÓN   ")
            print("="*100)
            print(f"  Dispositivo: {self.device}")
            print(f"  Clases: {num_classes}")
            print(f"  Épocas: {EPOCHS}")
            print(f"  Batch Size: {BATCH_SIZE}")
            print(f"  Learning Rate inicial: {lr:.6f}")
            print("="*100)
            print(f"{'Época':>6}  {'Train Loss':>9}  {'Train Acc':>8}  {'Val Loss':>9}  {'Val Acc':>8}  {'Mejor':>8}  {'LR':>10}  {'Grad Prom':>10}  {'Peor Clase':>15}")
            print("-"*110)

            start_time = time.time()
            for epoch in range(EPOCHS):
                tr_loss, tr_acc, avg_grad = self._train_epoch(train_loader, criterion, optimizer)
                val_loss, val_acc, class_acc = self._val_epoch(val_loader, criterion)
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    patience = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    best_marker = "⭐"
                else:
                    patience += 1
                    best_marker = ""
                if class_acc:
                    worst_idx = min(class_acc, key=class_acc.get)
                    worst_acc = class_acc[worst_idx]
                    worst_name = self.class_list[worst_idx] if self.class_list else "?"
                    worst_str = f"{worst_name}:{worst_acc:.1f}%"
                else:
                    worst_str = "N/A"
                print(f"{epoch+1:6}  {tr_loss:9.4f}  {tr_acc:8.1f}%  {val_loss:9.4f}  {val_acc:8.1f}%  {best_val_acc:8.1f}%  {current_lr:10.6f}  {avg_grad:10.6f}  {worst_str:>15}  {best_marker:>2}")
                self._session_log.append({
                    "epoch": epoch+1,
                    "train_loss": round(tr_loss, 4),
                    "train_acc": round(tr_acc, 2),
                    "val_loss": round(val_loss, 4),
                    "val_acc": round(val_acc, 2),
                    "lr": current_lr,
                    "avg_grad": avg_grad,
                    "class_acc": class_acc,
                })
                self._update_plot(epoch+1, tr_loss, val_loss, tr_acc, val_acc, current_lr, avg_grad, class_acc)
                if patience >= EARLY_STOPPING_PATIENCE:
                    print(f"\n⏹️ Early stopping en época {epoch+1}")
                    break
            if best_state:
                self.model.load_state_dict(best_state)

            total_time = time.time() - start_time
            print("\n" + "="*100)
            print("   📊 RESUMEN FINAL DEL ENTRENAMIENTO   ")
            print("="*100)
            print(f"  Mejor Precisión Validación: {best_val_acc:.1f}%")
            print(f"  Mejor Pérdida Validación:   {best_val_loss:.4f}")
            print(f"  Precisión Final Entrenamiento: {tr_acc:.1f}%")
            print(f"  Precisión Final Validación:   {val_acc:.1f}%")
            print(f"  Total Épocas:           {epoch+1}")
            print(f"  Tiempo Total:           {total_time:.0f}s")
            print("="*100)
            self._plot_confusion_matrix(val_loader)

            # ---- Guardar el modelo actual en la lista de mejores ----
            # Creamos un nombre único con timestamp y precisión
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"model_{timestamp}_acc{best_val_acc:.1f}.pth"
            model_path = os.path.join(self.models_folder, model_filename)
            torch.save(self.model.state_dict(), model_path)

            # Preparar metadatos
            model_info = {
                "timestamp": timestamp,
                "val_acc": round(best_val_acc, 2),
                "val_loss": round(best_val_loss, 4),
                "epochs": epoch+1,
                "class_list": self.class_list,
                "bot_feat_dim": self._bot_feat_dim,
                "file_name": model_filename,
                "train_acc": round(tr_acc, 2),
                "train_loss": round(tr_loss, 4),
            }
            self._add_model_to_metadata(model_info)

            # También guardar siempre el mejor modelo global en la raíz
            if best_val_acc >= self._best_val_acc_global:
                # Guardar como modelo principal
                torch.save(self.model.state_dict(), os.path.join(self.results_folder, "model_weights.pth"))
                meta = {
                    "class_list": self.class_list,
                    "error_count": 0,
                    "retrain_count": 0,
                    "device": str(self.device),
                    "bot_feat_dim": self._bot_feat_dim,
                    "best_val_acc_global": best_val_acc,
                }
                joblib.dump(meta, os.path.join(self.results_folder, "training_info.joblib"))
                print(f"⭐ Nuevo mejor modelo global guardado con {best_val_acc:.1f}% de precisión.")

            return True

        except Exception as e:
            print(f"\n❌ Error en entrenamiento: {e}")
            traceback.print_exc()
            return False

        finally:
            # Apagar el modo interactivo: si se deja encendido, plt.show()
            # en otras ventanas (evaluación de imágenes, monitor) deja de
            # bloquear y la ventana se abre y se cierra en el mismo instante.
            plt.ioff()
            # La ventana de entrenamiento queda abierta a propósito para
            # poder revisar las gráficas con calma; este hilo la mantiene
            # respondiendo hasta que el usuario la cierre.
            self._start_plot_keepalive()

    def _plot_confusion_matrix(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_b, bot_b, y_b in loader:
                X_b, bot_b = X_b.to(self.device), bot_b.to(self.device)
                out = self.model(X_b, bot_b)
                preds = out.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_b.numpy())
        if not all_labels:
            return
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_list,
                    yticklabels=self.class_list)
        plt.title('Matriz de Confusión - Validación')
        plt.tight_layout()
        out_path = os.path.join(self.results_folder, 'confusion_matrix.png')
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"📊 Matriz de confusión guardada en: {out_path}")

    # ------------------------------------------------------------
    # Persistencia del modelo global (guardar/cargar)
    # ------------------------------------------------------------
    def save(self, error_count: int = 0, retrain_count: int = 0) -> bool:
        """Guarda el modelo actual (el que está en memoria) como principal."""
        try:
            os.makedirs(self.results_folder, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(self.results_folder, "model_weights.pth"))
            meta = {
                "class_list": self.class_list,
                "error_count": error_count,
                "retrain_count": retrain_count,
                "device": str(self.device),
                "bot_feat_dim": self._bot_feat_dim,
                "best_val_acc_global": self._best_val_acc_global,
            }
            joblib.dump(meta, os.path.join(self.results_folder, "training_info.joblib"))

            # save() también se llama solo para persistir contadores (ej. al
            # confirmar una predicción en el aprendizaje activo), sin que haya
            # ocurrido un entrenamiento real. Si no hay épocas registradas en
            # esta sesión, no se agrega una entrada de sesión falsa (0%/0
            # épocas) al historial de logs/summary.json.
            if self._session_log:
                log_dir = os.path.join(self.results_folder, "logs")
                os.makedirs(log_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_data = {
                    "timestamp": timestamp,
                    "class_list": self.class_list,
                    "num_classes": len(self.class_list),
                    "epochs_run": len(self._session_log),
                    "best_val_acc": max((e["val_acc"] for e in self._session_log), default=0),
                    "best_val_loss": min((e["val_loss"] for e in self._session_log), default=0),
                    "error_count": error_count,
                    "retrain_count": retrain_count,
                    "history": self._session_log,
                }
                log_path = os.path.join(log_dir, f"session_{timestamp}.json")
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(session_data, f, indent=2)
                summary_path = os.path.join(log_dir, "summary.json")
                summary = []
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, encoding="utf-8") as f:
                            summary = json.load(f)
                    except:
                        pass
                summary.append({k: v for k, v in session_data.items() if k != "history"})
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)

                # Evita volver a registrar esta misma sesión si save() se
                # llama de nuevo más tarde (ej. al confirmar una predicción)
                # sin que haya habido un nuevo entrenamiento de por medio.
                self._session_log = []

            print(f"💾 Modelo guardado en: {self.results_folder}")
            return True
        except Exception as e:
            print(f"❌ Error guardando modelo: {e}")
            return False

    def load(self) -> tuple[bool, int, int]:
        """Carga el mejor modelo global (el que está en la raíz)."""
        model_path = os.path.join(self.results_folder, "model_weights.pth")
        info_path = os.path.join(self.results_folder, "training_info.joblib")
        if not os.path.exists(model_path) or not os.path.exists(info_path):
            print("Archivos de modelo no encontrados.")
            return False, 0, 0
        try:
            meta = joblib.load(info_path)
            self.class_list = meta.get("class_list", [])
            if not self.class_list:
                self.class_list = sorted(set(PLANT_DESCRIPTIONS.keys()) - {"mifoto"})
            num_classes = len(self.class_list)
            bot_feat_dim = meta.get("bot_feat_dim", BOTANICAL_FEATURE_DIM)
            self._bot_feat_dim = bot_feat_dim
            self._best_val_acc_global = meta.get("best_val_acc_global", 0.0)
            self.model = FusionCoffeeModel(num_classes=num_classes, bot_feat_dim=bot_feat_dim, pretrained=True).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"📦 Modelo cargado con {num_classes} clases y {bot_feat_dim} características botánicas.")
            return True, meta.get("error_count", 0), meta.get("retrain_count", 0)
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False, 0, 0

    # ------------------------------------------------------------
    # Función auxiliar para seleccionar modelo desde consola
    # ------------------------------------------------------------
    def select_model_interactive(self):
        """Permite al usuario ver la lista de modelos guardados y seleccionar uno."""
        if not self.models_metadata:
            print("\n📭 No hay modelos guardados. Entrena uno primero.")
            return False
        self.print_models_list()
        print("\nIngresa el número del modelo que deseas cargar (0 para cancelar):")
        try:
            choice = int(input("> ").strip())
            if choice == 0:
                print("Cancelado.")
                return False
            if choice < 1 or choice > len(self.models_metadata):
                print("Número inválido.")
                return False
            return self.load_model_by_index(choice - 1)
        except ValueError:
            print("Entrada inválida.")
            return False