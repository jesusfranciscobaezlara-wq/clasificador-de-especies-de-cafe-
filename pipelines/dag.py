# ============================================================
# pipelines/dag.py  —  DAG de Airflow para reentrenamiento
# ============================================================
import os
from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

# Mismo criterio que main.py y api/routes.py: datasets/ junto a la raíz del
# proyecto por defecto, salvo que COFFEE_BASE_DIR indique otra ubicación.
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.environ.get("COFFEE_BASE_DIR", os.path.join(_PROJECT_DIR, "datasets"))

default_args = {
    "owner": "coffee_classifier",
    "depends_on_past": False,
    "start_date": datetime(2023, 1, 1),
    "retries": 1,
}

dag = DAG(
    "coffee_plant_retraining",
    default_args=default_args,
    description="Reentrenamiento semanal del clasificador de café",
    schedule="0 0 * * 0",   # cada domingo a medianoche
)


def _retrain_model():
    from services.classifier import CoffeePlantClassifier
    print("\n🔄 Reentrenamiento programado iniciado...")
    clf = CoffeePlantClassifier(
        train_folder  = os.path.join(BASE_DIR, "Training"),
        test_folder   = os.path.join(BASE_DIR, "Test"),
        results_folder= os.path.join(BASE_DIR, "Results"),
    )
    if not clf.train_and_save():
        raise RuntimeError("Reentrenamiento fallido")
    print("✅ Reentrenamiento completado")


def _run_api():
    from api.routes import run_api
    run_api()


retrain_task = PythonOperator(
    task_id="retrain_coffee_classifier",
    python_callable=_retrain_model,
    dag=dag,
)

api_task = PythonOperator(
    task_id="run_classifier_api",
    python_callable=_run_api,
    dag=dag,
)

retrain_task >> api_task