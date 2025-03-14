import os
import pandas as pd
import mlflow
import mlflow.keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.utils.utils import save_json
from pathlib import Path
import tensorflow as tf
from src.entity.config_entity import ModelEvaluationConfig
import joblib

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        return {
            "accuracy": accuracy_score(actual, pred),
            "precision": precision_score(actual, pred, average="weighted"),
            "recall": recall_score(actual, pred, average="weighted"),
            "f1_score": f1_score(actual, pred, average="weighted"),
            "classification_report": classification_report(actual, pred, output_dict=True)
        }

    def save_results(self):
       
        test_data = pd.read_csv(self.config.test_data_path)
        model_path = os.path.join(self.config.root_dir, "model.keras")
        
        model = tf.keras.models.load_model("artifacts/model_trainer/model.keras", safe_mode=False)

        
        scaler_path ="artifacts/model_trainer/scaler.pkl"
        scaler = joblib.load(scaler_path)

        
        test_x = scaler.transform(test_data.drop(columns=[self.config.target_column]).values)
        test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))  
        test_y = test_data[self.config.target_column].values

        predicted_probs = model.predict(test_x)
        predicted_labels = (predicted_probs > 0.6).astype(int).flatten()

       
        scores = self.eval_metrics(test_y, predicted_labels)
        save_json(path=Path(self.config.metric_file_name), data=scores)

        print("Model evaluation completed! Metrics saved successfully.")
        print("Classification Report:\n", classification_report(test_y, predicted_labels))

       
        mlflow.set_experiment("SQL Injection Detection")

        with mlflow.start_run():
            mlflow.log_param("model_path", self.config.model_path)
            mlflow.log_metric("accuracy", scores["accuracy"])
            mlflow.log_metric("precision_weighted", scores["precision"])
            mlflow.log_metric("recall_weighted", scores["recall"])
            mlflow.log_metric("f1_score_weighted", scores["f1_score"])
            
           
            for class_label in scores["classification_report"].keys():
                if class_label.isdigit():  
                    mlflow.log_metric(f"precision_class_{class_label}", scores["classification_report"][class_label]["precision"])
                    mlflow.log_metric(f"recall_class_{class_label}", scores["classification_report"][class_label]["recall"])

            report_path = Path(self.config.root_dir) / "classification_report.txt"
            with open(report_path, "w") as f:
                f.write(classification_report(test_y, predicted_labels))

            mlflow.log_artifact(str(report_path))
            mlflow.keras.log_model(model, "model")
            

            

        print("Metrics and model logged in MLflow!")
