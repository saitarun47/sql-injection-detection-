import mlflow
import mlflow.sklearn
import os

class MLFlowTracking:
    def __init__(self, experiment_name="SQL_Injection_Detection", script_paths=None):
        self.experiment_name = experiment_name

        # Ensure paths are absolute and correctly formatted
        base_dir = os.path.abspath(os.path.join(os.getcwd(), "src", "components"))
        self.script_paths = script_paths or [
            os.path.join(base_dir, "model_trainer.py"),
            os.path.join(base_dir, "data_transformation.py"),
            os.path.join(base_dir, "model_evaluation.py"),
        ]

        mlflow.set_experiment(self.experiment_name)

    def log_metrics(self, params, metrics, model):
        """Logs parameters, metrics, model, and scripts to MLflow."""
        with mlflow.start_run():
            # Log hyperparameters
            for param, value in params.items():
                mlflow.log_param(param, value)

            # Log performance metrics
            for metric, value in metrics.items():
                mlflow.log_metric(metric, value)

            # Log trained model
            mlflow.sklearn.log_model(model, "model")

            # Manually check and log scripts
            for script in self.script_paths:
                abs_script_path = os.path.abspath(script)  # Ensure absolute path
                if os.path.exists(abs_script_path):
                    print(f"Logging script: {abs_script_path}")  # Debugging
                    mlflow.log_artifact(abs_script_path)  # Manually log
                    print(f"Successfully logged: {abs_script_path}")
                else:
                    print(f"Script not found: {abs_script_path}")

            print("MLflow tracking complete with scripts saved!")

    def log_scripts_separately(self):
        """Manually log scripts in a separate MLflow run if needed."""
        with mlflow.start_run():
            for script in self.script_paths:
                abs_script_path = os.path.abspath(script)
                if os.path.exists(abs_script_path):
                    print(f"Manually logging script: {abs_script_path}")
                    mlflow.log_artifact(abs_script_path)
                    print(f"Successfully logged manually: {abs_script_path}")
                else:
                    print(f" Script not found: {abs_script_path}")

    def load_best_model(self, run_id):
        """Loads the best model from MLflow given a run ID."""
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)
        return model
