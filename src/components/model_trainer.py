import os
import pandas as pd
import numpy as np
import mlflow
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
from src import logger
from src.entity.config_entity import ModelTrainerConfig

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

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self):
        
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        
        
        vocab_path = os.path.join(os.path.dirname(self.config.train_data_path), "vocab.pkl")
        params_path = os.path.join(os.path.dirname(self.config.train_data_path), "params.pkl")
        
        vocab = joblib.load(vocab_path)
        params = joblib.load(params_path)
        
       
        X_train = train_data['encoded_text'].apply(eval).tolist()
        X_test = test_data['encoded_text'].apply(eval).tolist()
        y_train = train_data['Label'].tolist()
        y_test = test_data['Label'].tolist()

        
        train_dataset = TextDataset(X_train, y_train)
        test_dataset = TextDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)

       
        vocab_size = params['vocab_size']
        embed_dim = self.config.embed_dim
        hidden_dim = self.config.hidden_dim
        padding_idx = params['pad_idx']

       
        model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, padding_idx=padding_idx).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)

       
        num_epochs = self.config.epochs
        patience = 2
        best_val_loss = float('inf')
        epochs_no_improve = 0

        train_losses = []
        val_losses = []

       
        mlflow.set_tracking_uri("http://localhost:5000")

        with mlflow.start_run():
            
            for epoch in range(num_epochs):
                model.train()
                running_train_loss = 0.0
                train_preds, train_labels = [], []

                for texts_batch, labels_batch in train_loader:
                    texts_batch = texts_batch.to(self.device)
                    labels_batch = labels_batch.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(texts_batch)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    optimizer.step()

                    running_train_loss += loss.item()
                    preds = torch.sigmoid(outputs).round().cpu().detach().numpy()
                    train_preds.extend(preds)
                    train_labels.extend(labels_batch.cpu().numpy())

                avg_train_loss = running_train_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                train_acc = accuracy_score(train_labels, train_preds)
                print(f"Epoch {epoch}/{num_epochs} – Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")

               
                model.eval()
                running_val_loss = 0.0
                val_preds, val_labels = [], []
                
                with torch.no_grad():
                    for texts_batch, labels_batch in test_loader:
                        texts_batch = texts_batch.to(self.device)
                        labels_batch = labels_batch.to(self.device)
                        outputs = model(texts_batch)
                        loss = criterion(outputs, labels_batch)
                        running_val_loss += loss.item()

                        preds = torch.sigmoid(outputs).round().cpu().numpy()
                        val_preds.extend(preds)
                        val_labels.extend(labels_batch.cpu().numpy())

                    avg_val_loss = running_val_loss / len(test_loader)
                    val_losses.append(avg_val_loss)

                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    
                    model_path = os.path.join(self.config.root_dir, "model.pt")
                    torch.save(model.state_dict(), model_path)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"No improvement in val loss for {patience} epochs. Stopping early.")
                        break

           
            model.eval()
            final_val_preds, final_val_labels = [], []
            with torch.no_grad():
                for texts_batch, labels_batch in test_loader:
                    texts_batch = texts_batch.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    outputs = model(texts_batch)
                    preds = torch.sigmoid(outputs).round().cpu().numpy()
                    final_val_preds.extend(preds)
                    final_val_labels.extend(labels_batch.cpu().numpy())

            final_accuracy = accuracy_score(final_val_labels, final_val_preds)

           
            model_params = {
                'vocab_size': vocab_size,
                'embed_dim': embed_dim,
                'hidden_dim': hidden_dim,
                'max_len': params['max_len'],
                'pad_idx': padding_idx
            }
            params_save_path = os.path.join(self.config.root_dir, "model_params.pkl")
            joblib.dump(model_params, params_save_path)

            
            mlflow.log_param("vocab_size", vocab_size)
            mlflow.log_param("embed_dim", embed_dim)
            mlflow.log_param("hidden_dim", hidden_dim)
            mlflow.log_param("epochs", num_epochs)
            mlflow.log_param("batch_size", self.config.batch_size)
            mlflow.log_metric("final_accuracy", final_accuracy)
            mlflow.log_metric("best_val_loss", best_val_loss)

            
            mlflow.pytorch.log_model(model, "model")

            logger.info(f"Model trained with accuracy: {final_accuracy:.4f}")
            print(f"Model trained successfully with accuracy: {final_accuracy:.4f}")

        print("MLflow tracking complete!")
