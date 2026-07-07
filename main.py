# ============================================================
# main.py  --  Punto de entrada del sistema 
# ============================================================
import os
import sys
import traceback

# El proyecto usa emojis en muchos print() (✅ ❌ ⚠️ etc.). Si la salida
# estándar no está en UTF-8 (cmd.exe clásico, salida redirigida a archivo,
# algunas consolas de CI), esos print() lanzan UnicodeEncodeError y crashean
# el programa justo dentro de un bloque except. Forzamos UTF-8 al iniciar.
if sys.stdout.encoding is not None and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import cv2

from services.classifier import CoffeePlantClassifier
from vision.camera import CameraSession
from vision.measurement import calcular_medidas_imagen
from config.settings import get_display_name

# Por defecto usa datasets/ junto al proyecto (portátil entre máquinas);
# COFFEE_BASE_DIR permite apuntar a otra ubicación sin tocar el código.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.environ.get("COFFEE_BASE_DIR", os.path.join(_PROJECT_DIR, "datasets"))


def build_classifier():
    return CoffeePlantClassifier(
        train_folder   = os.path.join(BASE_DIR, "Training"),
        test_folder    = os.path.join(BASE_DIR, "Test"),
        results_folder = os.path.join(BASE_DIR, "Results"),
    )


def detect_available_cameras(max_cameras=5):
    available = []
    print("\nDetectando cámaras...")
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened() and cap.read()[0]:
            available.append(i)
            print(f"   Cámara {i} detectada")
        cap.release()
    return available


def select_camera(cameras):
    """Permite al usuario elegir una cámara de la lista."""
    print(f"\nCámaras disponibles: {cameras}")
    try:
        choice = input(f"Número de cámara (Enter = {cameras[0]}): ").strip()
        camera_id = int(choice) if choice else cameras[0]
        if camera_id in cameras:
            return camera_id
        else:
            print("ID no disponible, usando cámara 0.")
            return cameras[0]
    except (ValueError, IndexError):
        return cameras[0]


# ============================================================
# Modo Entrenamiento
# ============================================================
def modo_entrenamiento(classifier):
    while True:
        print("\n" + "="*40)
        print("  MODO ENTRENAMIENTO")
        print("="*40)
        print("  1. Entrenar modelo")
        print("  2. Evaluar imágenes (aprendizaje activo)")
        print("  3. Volver al menú principal")
        op = input("Opción (1-3): ").strip()

        if op == "1":
            print("\nIniciando entrenamiento...")
            classifier.train_and_save()
        elif op == "2":
            evaluar_imagenes_entrenamiento(classifier)
        elif op == "3":
            break
        else:
            print("Opción inválida")


def evaluar_imagenes_entrenamiento(classifier):
    """Ciclo de evaluación con feedback (modo entrenamiento)."""
    mifoto_dir = os.path.join(classifier.test_folder, "mifoto")
    if not os.path.exists(mifoto_dir):
        os.makedirs(mifoto_dir, exist_ok=True)

    while True:
        try:
            print("\n--- Evaluación de imágenes (Enter para salir) ---")
            imgs = [f for f in os.listdir(mifoto_dir)
                    if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
            if not imgs:
                print("No hay imágenes en Test/mifoto.")
                return

            for i, name in enumerate(imgs, 1):
                print(f"  {i}. {name}")
            print("  0. Volver")
            choice = input("Seleccione número: ").strip()
            if choice == "0" or choice == "":
                return

            try:
                idx = int(choice) - 1
                if idx < 0 or idx >= len(imgs):
                    print("Número inválido.")
                    continue
            except ValueError:
                print("Entrada inválida.")
                continue

            img_path = os.path.join(mifoto_dir, imgs[idx])
            pred, proba = classifier.predict_single_image(img_path, show_image=True)
            if pred is None:
                print("Error al procesar la imagen.")
                continue

            print(f"\nPredicción: {get_display_name(pred)} ({proba:.2%})")
            print(calcular_medidas_imagen(img_path))

            while True:
                resp = input("¿Fue correcta la predicción? (s/n): ").strip().lower()
                if resp == "s":
                    classifier.move_to_class_folder(img_path, pred, dataset="train")
                    classifier.trainer.save(classifier.error_count, classifier.retrain_count)
                    print("✅ Imagen movida a entrenamiento (acierto).")
                    break
                elif resp == "n":
                    classifier.move_to_mifoto(img_path, dataset="test")
                    classifier.learn_from_mistake(img_path)
                    print("❌ Imagen movida a Test/mifoto (error registrado).")
                    break
                else:
                    print("Respuesta no válida. Escriba 's' o 'n'.")

        except KeyboardInterrupt:
            print("\n⚠️ Interrupción manual detectada. Saliendo del modo evaluación...")
            return
        except Exception as e:
            print(f"\n❌ Error inesperado en el bucle: {e}")
            import traceback
            traceback.print_exc()
            input("Presiona Enter para continuar...")
            return


# ============================================================
# Modo Prueba
# ============================================================
def modo_prueba(classifier):
    while True:
        print("\n" + "="*40)
        print("  MODO PRUEBA")
        print("="*40)
        print("  1. Predecir una imagen (sin guardar)")
        print("  2. Usar cámara en tiempo real")
        print("  3. Ver progreso del sistema")
        print("  4. Volver al menú principal")
        op = input("Opción (1-4): ").strip()

        if op == "1":
            predecir_imagen_prueba(classifier)
        elif op == "2":
            cameras = detect_available_cameras()
            if not cameras:
                print("\nNo hay cámaras disponibles.")
                continue
            cam_id = select_camera(cameras)
            session = CameraSession(classifier)
            session.run(cam_id)
        elif op == "3":
            classifier.show_learning_progress()
        elif op == "4":
            break
        else:
            print("Opción inválida")


def predecir_imagen_prueba(classifier):
    """Predice una imagen de Test/mifoto sin guardar feedback."""
    mifoto_dir = os.path.join(classifier.test_folder, "mifoto")
    if not os.path.exists(mifoto_dir):
        os.makedirs(mifoto_dir, exist_ok=True)

    while True:
        imgs = [f for f in os.listdir(mifoto_dir)
                if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
        if not imgs:
            print("No hay imágenes en Test/mifoto.")
            return

        print("\n--- Predicción de prueba ---")
        for i, name in enumerate(imgs, 1):
            print(f"  {i}. {name}")
        print("  0. Volver")

        choice = input("Seleccione número: ").strip()
        if choice == "0" or choice == "":
            return
        try:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(imgs):
                print("Número inválido.")
                continue
        except ValueError:
            print("Entrada inválida.")
            continue

        img_path = os.path.join(mifoto_dir, imgs[idx])
        pred, proba = classifier.predict_single_image(img_path, show_image=True)
        if pred is None:
            print("Error al procesar la imagen.")
            continue

        print(f"\nPredicción: {get_display_name(pred)} ({proba:.2%})")
        print(calcular_medidas_imagen(img_path))
        # Sin feedback


# ============================================================
# Gestión de modelos guardados
# ============================================================
def gestionar_modelos(classifier):
    """Muestra la lista de modelos guardados y permite cargar uno."""
    while True:
        print("\n" + "="*40)
        print("  GESTIÓN DE MODELOS GUARDADOS")
        print("="*40)
        classifier.trainer.print_models_list()
        print("\n  1. Cargar modelo seleccionado")
        print("  2. Cargar el mejor modelo (por precisión)")
        print("  3. Volver al menú principal")
        op = input("Opción (1-3): ").strip()

        if op == "1":
            classifier.trainer.select_model_interactive()
            # Actualizar idx_to_class después de cargar
            classifier._rebuild_idx_to_class()
            if classifier.trainer.model is not None:
                classifier._bot_feat_dim = classifier.trainer._bot_feat_dim
        elif op == "2":
            if classifier.trainer.load_best_model():
                classifier._rebuild_idx_to_class()
                if classifier.trainer.model is not None:
                    classifier._bot_feat_dim = classifier.trainer._bot_feat_dim
                print("✅ Modelo cargado.")
            else:
                print("❌ No se pudo cargar el mejor modelo.")
        elif op == "3":
            break
        else:
            print("Opción inválida")


# ============================================================
# Menú principal
# ============================================================
def menu_principal(classifier):
    while True:
        print("\n" + "="*50)
        print("  SISTEMA DE CLASIFICACIÓN DE CAFÉ")
        print("="*50)
        print("  1. Modo Entrenamiento")
        print("  2. Modo Prueba")
        print("  3. Ver progreso del sistema")
        print("  4. Monitor de entrenamientos")
        print("  5. Abrir interfaz gráfica")
        print("  6. Gestionar modelos guardados")
        print("  0. Salir")

        op = input("Opción: ").strip().upper()
        if op == "1":
            modo_entrenamiento(classifier)
        elif op == "2":
            modo_prueba(classifier)
        elif op == "3":
            classifier.show_learning_progress()
        elif op == "4":
            try:
                from tools.training_monitor import load_summary, menu
                summary = load_summary()
                menu(summary)
            except ImportError:
                print("No se encontró tools/training_monitor.py")
            except SystemExit:
                pass
        elif op == "5":
            try:
                from vision.gui_app import main_gui
                main_gui()
            except ImportError as e:
                print(f"No se pudo cargar la GUI: {e}")
        elif op == "6":
            gestionar_modelos(classifier)
        elif op == "0":
            print("\n¡Hasta luego!")
            sys.exit(0)
        else:
            print("Opción inválida")


# ============================================================
# Punto de entrada
# ============================================================
def main():
    try:
        classifier = build_classifier()
        classifier.load_models()
        menu_principal(classifier)
    except Exception as e:
        print(f"\nError inesperado: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()