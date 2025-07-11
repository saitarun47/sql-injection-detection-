import joblib 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

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

class PredictionPipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        vocab_path = Path('artifacts/data_transformation/vocab.pkl')
        params_path = Path('artifacts/data_transformation/params.pkl')
        model_params_path = Path('artifacts/model_trainer/model_params.pkl')
        
        self.vocab = joblib.load(vocab_path)
        self.params = joblib.load(params_path)
        self.model_params = joblib.load(model_params_path)
        
       
        self.model = LSTMClassifier(
            vocab_size=self.model_params['vocab_size'],
            embed_dim=self.model_params['embed_dim'],
            hidden_dim=self.model_params['hidden_dim'],
            padding_idx=self.model_params['pad_idx']
        ).to(self.device)
        
        
        model_path = Path('artifacts/model_trainer/model.pt')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def tokenize(self, text):
        """Tokenizes text using NLTK word_tokenize."""
        return word_tokenize(text.lower())

    def encode_and_pad(self, text, max_len):
        """Encodes text to indices and pads to max_len."""
        tokens = self.tokenize(text)
        pad_idx = self.vocab['<pad>']
        unk_idx = self.vocab['<unk>']
        
        ids = [self.vocab.get(token, unk_idx) for token in tokens]
        if len(ids) < max_len:
            ids = ids + [pad_idx] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        return ids

    def predict(self, text):
        """Predicts whether the input text is SQL injection or not."""
       
        encoded_text = self.encode_and_pad(text, self.params['max_len'])
        
        
        input_tensor = torch.tensor([encoded_text], dtype=torch.long).to(self.device)
        
       
        with torch.no_grad():
            output = self.model(input_tensor)
            probability = torch.sigmoid(output).cpu().numpy()[0]
            prediction = (probability > 0.5).astype(int)
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'is_sql_injection': bool(prediction)
        }