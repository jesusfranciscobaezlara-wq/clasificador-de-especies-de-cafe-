# ============================================================
# training/evaluator.py  --  Métricas y reporte (adaptado para fusión)
# ============================================================
import os
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from config.settings import get_display_name

def evaluate(model, test_loader: DataLoader, device, results_folder: str, class_list: list[str]) -> None:
    """
    Evalúa el modelo de fusión usando un DataLoader que devuelve (imagen, bot_feats, etiqueta).
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, bot_feats, labels in test_loader:
            images = images.to(device)
            bot_feats = bot_feats.to(device)
            labels = labels.to(device)
            outputs = model(images, bot_feats)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    target_names = [get_display_name(cls) for cls in class_list]

    print("\n=== Reporte de Clasificación ===")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Matriz de Confusión - Evaluación", pad=20)
    plt.tight_layout()
    out_path = os.path.join(results_folder, "confusion_matrix_eval.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📊 Matriz guardada en: {out_path}")

    # Reporte CSV
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(results_folder, "evaluation_report.csv"), float_format="%.2f")
    print(f"📄 Reporte CSV guardado en: {os.path.join(results_folder, 'evaluation_report.csv')}")