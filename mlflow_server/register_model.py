import mlflow
import pandas as pd
import joblib
import json
import os
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv

load_dotenv()

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID") 
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY") 

base_path_v1 = "/home/mle-user/mle_projects/mle-project-sprint-1-v001/part2_dvc"
model_path = os.path.join(base_path_v1, "models/fitted_model.pkl")
data_path = os.path.join(base_path_v1, "data/init_data.csv")
metrics_path = os.path.join(base_path_v1, "cv_results/cv_res.json")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("baseline_real_estate")

def register_baseline():
    if not os.path.exists(metrics_path):
        print(f"ОШИБКА: Файл метрик не найден: {metrics_path}")
        return
    
    with open(metrics_path, "r") as f:
        old_metrics = json.load(f)
    
    model = joblib.load(model_path)
    df = pd.read_csv(data_path).head(5)
    
    target_col = 'price'
    X_sample = df.drop(columns=[target_col])
    y_sample = df[target_col]
    signature = infer_signature(X_sample, y_sample)

    with mlflow.start_run(run_name="initial_baseline_run") as run:
        mlflow.log_metric("mae", old_metrics.get("mae", 0))
        mlflow.log_metric("r2", old_metrics.get("r2", 0))
        mlflow.log_metric("fit_time", old_metrics["fit_time"])

        mlflow.log_param("model_type", "sk_pipeline_baseline")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="baseline_model",
            signature=signature,
            registered_model_name="RealEstate_Baseline"
        )
        
        print(f"Модель зарегистрирована!")
        print(f"Использованы метрики из прошлого проекта: MAE={old_metrics.get('mae')}, R2={old_metrics.get('r2')}")

if __name__ == "__main__":
    register_baseline()