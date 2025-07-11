import os
import pandas as pd
import mlflow
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.utils.utils import save_json
from pathlib import Path
from src.entity.config_entity import ModelEvaluationConfig

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return text, label

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim=1, dropout=0.5, padding_idx=0):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        logits = self.fc(out).squeeze(1)
        return logits

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        
        
        vocab_path = os.path.join(os.path.dirname(self.config.test_data_path), "vocab.pkl")
        params_path = os.path.join(os.path.dirname(self.config.test_data_path), "params.pkl")
        
        vocab = joblib.load(vocab_path)
        params = joblib.load(params_path)
        
        
        model_params_path = os.path.join(os.path.dirname(self.config.root_dir), "model_trainer", "model_params.pkl")
        model_params = joblib.load(model_params_path)
        
       
        X_test = test_data['encoded_text'].apply(eval).tolist()
        y_test = test_data['Label'].tolist()

       
        test_dataset = TextDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=64)

      
        model = LSTMClassifier(
            vocab_size=model_params['vocab_size'],
            embed_dim=model_params['embed_dim'],
            hidden_dim=model_params['hidden_dim'],
            padding_idx=model_params['pad_idx']
        ).to(self.device)

       
        model_path = os.path.join(os.path.dirname(self.config.root_dir), "model_trainer", "model.pt")
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        predicted_labels = []
        actual_labels = []
        
        with torch.no_grad():
            for texts_batch, labels_batch in test_loader:
                texts_batch = texts_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                outputs = model(texts_batch)
                preds = torch.sigmoid(outputs).round().cpu().numpy()
                predicted_labels.extend(preds)
                actual_labels.extend(labels_batch.cpu().numpy())

       
        scores = self.eval_metrics(actual_labels, predicted_labels)
        save_json(path=Path(self.config.metric_file_name), data=scores)

        print("Model evaluation completed! Metrics saved successfully.")
        print("Classification Report:\n", classification_report(actual_labels, predicted_labels))

        # MLflow logging
        try:
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
                    f.write(classification_report(actual_labels, predicted_labels))

                mlflow.log_artifact(str(report_path))
                mlflow.pytorch.log_model(model, "model")

            print("Metrics and model logged in MLflow!")
        except Exception as e:
            print(f"MLflow tracking not available: {e}")
            print("Saving metrics locally only...")
            
           
            report_path = Path(self.config.root_dir) / "classification_report.txt"
            with open(report_path, "w") as f:
                f.write(classification_report(actual_labels, predicted_labels))
            
            print("Metrics saved locally.")
