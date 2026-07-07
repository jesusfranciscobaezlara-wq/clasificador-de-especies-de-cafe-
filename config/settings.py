# ============================================================
# config/settings.py  --  Parámetros centrales del sistema
# ============================================================
IMAGE_SIZE = (640, 640)
CNN_IMAGE_SIZE = (224, 224)
PCA_VARIANCE = 0.95
MAX_PCA_COMPONENTS = 300
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.005
DROPOUT_RATE = 0.5
USE_SMOTE = False
SMOTE_K_NEIGHBORS = 3
AUGMENTATION_FACTOR = 3
EARLY_STOPPING_PATIENCE = 25
ERROR_THRESHOLD_FOR_RETRAIN = 20
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

PLANT_DESCRIPTIONS = {
    "liberica": {
        "nombre_cientifico": "Coffea liberica",
        "descripcion": "Granos grandes y aroma intenso con notas ahumadas y florales características.",
        "altitud_ideal": "200-600 msnm",
        "uso": "Cafés especiales de origen único"
    },
    "marcelleza": {
        "nombre_cientifico": "Coffea arabica (variedad Marsellesa®)",
        "descripcion": "Variedad de café híbrido resistente a roya, con buena productividad y notas frutales y florales.",
        "altitud_ideal": "700-1300 msnm",
        "uso": "Café comercial de volumen y calidad equilibrada"
    },
    "geisha": {
        "nombre_cientifico": "Coffea arabica (variedad Geisha/Gesha)",
        "descripcion": "Café arábica de alta gama, conocido por sus notas florales, dulces y frutales.",
        "altitud_ideal": "1600-2000+ msnm",
        "uso": "Cafés de especialidad de alta gama"
    },
    "robusta_comun": {
        "nombre_cientifico": "Coffea canephora",
        "descripcion": "Café fuerte y amargo con alto contenido de cafeína y mayor resistencia a plagas.",
        "altitud_ideal": "0-800 msnm",
        "uso": "Café comercial, instantáneo y mezclas"
    },
    "robusta_nemaya": {
        "nombre_cientifico": "Coffea canephora var. Nemaya",
        "descripcion": "Variedad robusta resistente a nemátodos, utilizada como portainjerto.",
        "altitud_ideal": "400-900 msnm",
        "uso": "Portainjerto y producción en zonas bajas y medias"
    }
}

DISPLAY_NAMES = {
    "liberica":       "Liberica",
    "marcelleza":     "Marcelleza",
    "geisha":         "Geisha",
    "robusta_comun":  "Robusta Común",
    "robusta_nemaya": "Robusta Nemaya",
}

def get_display_name(plant_key: str) -> str:
    return DISPLAY_NAMES.get(plant_key, plant_key)