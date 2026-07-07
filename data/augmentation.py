# ============================================================
# data/augmentation.py  —  Aumento de datos y balanceo
# ============================================================
import numpy as np
import albumentations as A
from imblearn.over_sampling import SMOTE, RandomOverSampler

from config.settings import AUGMENTATION_FACTOR, SMOTE_K_NEIGHBORS, USE_SMOTE


def build_aug_transform() -> A.Compose:
    return A.Compose([
        A.Rotate(limit=25, p=0.5),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
        A.Affine(scale=(0.9, 1.1), translate_percent=0.05,
                 rotate=(-25, 25), shear=(-5, 5), p=0.4),
        A.CLAHE(p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.1),
    ])


def augment_images(
    X: np.ndarray,
    y: np.ndarray,
    aug_transform: A.Compose,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera imágenes aumentadas para equilibrar las clases."""
    augmented_images, augmented_labels = [], []
    unique, counts = np.unique(y, return_counts=True)
    max_samples = max(counts) if len(counts) > 0 else 0

    for img, label in zip(X, y):
        img_uint8 = (img * 255).astype(np.uint8)
        current_count = counts[np.where(unique == label)[0][0]] if label in unique else 0
        needed = max(0, max_samples - current_count)
        per_image = max(1, min(AUGMENTATION_FACTOR, needed // max(current_count, 1)))
        for _ in range(per_image):
            aug_img = aug_transform(image=img_uint8)["image"].astype(np.float32) / 255.0
            augmented_images.append(aug_img)
            augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)


def balance_with_smote(
    X_features: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica SMOTE (o RandomOverSampler si hay pocas muestras)."""
    if not USE_SMOTE:
        return X_features, y

    print("\nBalanceando clases con SMOTE...")
    _, counts = np.unique(y, return_counts=True)
    min_samples = min(counts)
    k = min(SMOTE_K_NEIGHBORS, min_samples - 1) if min_samples > 1 else 1

    if k < 1:
        sampler = RandomOverSampler(random_state=42)
    else:
        sampler = SMOTE(k_neighbors=k, random_state=42)

    return sampler.fit_resample(X_features, y)