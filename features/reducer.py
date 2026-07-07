# ============================================================
# features/reducer.py  —  StandardScaler + PCA
# ============================================================
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config.settings import MAX_PCA_COMPONENTS, PCA_VARIANCE
from features.extractor import extract_features_with_measurements


class FeatureReducer:
    """Encapsula el pipeline scaler → PCA. Stateful: guarda parámetros al fit."""

    def __init__(self):
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.expected_feature_dim: int | None = None
        self.expected_leaf_dim: int = 4   # número de mediciones de hoja
        self._use_leaf_features: bool = False

    # ------------------------------------------------------------------
    def fit_transform(
        self,
        images: np.ndarray,
        leaf_measurements: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Extrae, escala y reduce features. Ajusta los transformadores.
        
        Parámetros
        ----------
        images : np.ndarray
            Lote de imágenes.
        leaf_measurements : np.ndarray o None
            Array de shape (n_images, 4) con mediciones de hoja.
            Si es None, se usan ceros.
        """
        # Extraer características visuales y concatenar mediciones
        raw_visual = self._extract_all_visual(images)
        if leaf_measurements is not None:
            leaf_meas = np.atleast_2d(leaf_measurements).astype(np.float32)
            if leaf_meas.shape[0] != raw_visual.shape[0]:
                raise ValueError("Número de mediciones no coincide con imágenes")
            # Asegurar 4 columnas
            if leaf_meas.shape[1] < self.expected_leaf_dim:
                pad = np.zeros((leaf_meas.shape[0], self.expected_leaf_dim - leaf_meas.shape[1]))
                leaf_meas = np.hstack([leaf_meas, pad])
            elif leaf_meas.shape[1] > self.expected_leaf_dim:
                leaf_meas = leaf_meas[:, :self.expected_leaf_dim]
            self._use_leaf_features = True
        else:
            leaf_meas = np.zeros((raw_visual.shape[0], self.expected_leaf_dim), dtype=np.float32)
            self._use_leaf_features = False

        raw = np.hstack([raw_visual, leaf_meas])
        self.expected_feature_dim = raw.shape[1]

        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(raw)

        # Elegir n_components automáticamente
        pca_temp = PCA().fit(scaled)
        cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
        n = min(int(np.argmax(cumsum >= PCA_VARIANCE)) + 1, MAX_PCA_COMPONENTS)
        print(f"  PCA: {n} componentes → varianza explicada {cumsum[n-1]:.4f}")

        self.pca = PCA(n_components=n)
        return self.pca.fit_transform(scaled)

    def transform(
        self,
        images: np.ndarray,
        leaf_measurements: np.ndarray | None = None
    ) -> np.ndarray:
        """Extrae, escala y reduce con los transformadores ya ajustados."""
        assert self.scaler and self.pca, "Llama a fit_transform primero"

        raw_visual = self._extract_all_visual(images)

        if leaf_measurements is not None:
            leaf_meas = np.atleast_2d(leaf_measurements).astype(np.float32)
            if leaf_meas.shape[0] != raw_visual.shape[0]:
                raise ValueError("Número de mediciones no coincide con imágenes")
            if leaf_meas.shape[1] < self.expected_leaf_dim:
                pad = np.zeros((leaf_meas.shape[0], self.expected_leaf_dim - leaf_meas.shape[1]))
                leaf_meas = np.hstack([leaf_meas, pad])
            elif leaf_meas.shape[1] > self.expected_leaf_dim:
                leaf_meas = leaf_meas[:, :self.expected_leaf_dim]
        else:
            leaf_meas = np.zeros((raw_visual.shape[0], self.expected_leaf_dim), dtype=np.float32)

        raw = np.hstack([raw_visual, leaf_meas])

        # Alinear dimensión si hay mismatch
        if raw.shape[1] != self.expected_feature_dim:
            if raw.shape[1] > self.expected_feature_dim:
                raw = raw[:, : self.expected_feature_dim]
            else:
                pad = np.zeros((raw.shape[0], self.expected_feature_dim - raw.shape[1]))
                raw = np.hstack([raw, pad])

        return self.pca.transform(self.scaler.transform(raw))

    # ------------------------------------------------------------------
    #  Metodos para trabajar con features visuales pre‑calculadas
    # ------------------------------------------------------------------
    def fit_transform_from_visual(
        self,
        visual_features: np.ndarray,
        leaf_measurements: np.ndarray | None = None
    ) -> np.ndarray:
        """Ajusta scaler + PCA usando features visuales ya extraídas."""
        leaf_meas = np.zeros((visual_features.shape[0], self.expected_leaf_dim),
                             dtype=np.float32)
        if leaf_measurements is not None:
            lm = np.atleast_2d(leaf_measurements).astype(np.float32)
            if lm.shape[0] != visual_features.shape[0]:
                raise ValueError("Número de mediciones no coincide con features")
            leaf_meas[:, : min(lm.shape[1], self.expected_leaf_dim)] = lm[:, :self.expected_leaf_dim]

        raw = np.hstack([visual_features, leaf_meas])
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(raw)
        self.expected_feature_dim = raw.shape[1]

        pca_temp = PCA().fit(scaled)
        cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
        n = min(int(np.argmax(cumsum >= PCA_VARIANCE)) + 1, MAX_PCA_COMPONENTS)
        print(f"  PCA: {n} componentes → varianza explicada {cumsum[n-1]:.4f}")

        self.pca = PCA(n_components=n)
        return self.pca.fit_transform(scaled)

    def transform_from_visual(
        self,
        visual_features: np.ndarray,
        leaf_measurements: np.ndarray | None = None
    ) -> np.ndarray:
        """Aplica scaler + PCA sobre features visuales ya extraídas."""
        assert self.scaler and self.pca, "Llama a fit_transform primero"
        leaf_meas = np.zeros((visual_features.shape[0], self.expected_leaf_dim),
                             dtype=np.float32)
        if leaf_measurements is not None:
            lm = np.atleast_2d(leaf_measurements).astype(np.float32)
            if lm.shape[0] != visual_features.shape[0]:
                raise ValueError("Número de mediciones no coincide con features")
            leaf_meas[:, : min(lm.shape[1], self.expected_leaf_dim)] = lm[:, :self.expected_leaf_dim]

        raw = np.hstack([visual_features, leaf_meas])
        if raw.shape[1] != self.expected_feature_dim:
            if raw.shape[1] > self.expected_feature_dim:
                raw = raw[:, :self.expected_feature_dim]
            else:
                pad = np.zeros((raw.shape[0], self.expected_feature_dim - raw.shape[1]))
                raw = np.hstack([raw, pad])
        return self.pca.transform(self.scaler.transform(raw))

    # ------------------------------------------------------------------
    def _extract_all_visual(self, images: np.ndarray) -> np.ndarray:
        """Extrae solo características visuales (sin mediciones)."""
        return np.array([extract_features_with_measurements(img, None) for img in images])

    def uses_leaf_features(self) -> bool:
        return self._use_leaf_features