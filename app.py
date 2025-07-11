import gradio as gr
import torch
import torch.nn as nn
import numpy as np
import joblib
import re
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# --- Model Loader (same as Flask logic) ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim=1, dropout=0.5, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        logits = self.fc(out).squeeze(1)
        return logits

class SQLInjectionDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vocab = None
        self.params = None
        self.load_model()
    def load_model(self):
        model_params = joblib.load("artifacts/model_trainer/model_params.pkl")
        self.vocab = joblib.load("artifacts/data_transformation/vocab.pkl")
        self.params = joblib.load("artifacts/data_transformation/params.pkl")
        self.model = LSTMClassifier(
            vocab_size=model_params['vocab_size'],
            embed_dim=model_params['embed_dim'],
            hidden_dim=model_params['hidden_dim'],
            padding_idx=model_params['pad_idx']
        ).to(self.device)
        self.model.load_state_dict(torch.load("artifacts/model_trainer/model.pt", map_location=self.device))
        self.model.eval()
    def clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s<>=\-\+\*\/\(\)\[\]\{\}\.\,\;\:\'\"]', '', text)
        return text
    def convert_word_to_int(self, sentence, vocab):
        return [vocab.get(word, vocab['<unk>']) for word in sentence]
    def pad_features(self, reviews_int, seq_length):
        features = np.zeros((len(reviews_int), seq_length), dtype=int)
        for i, row in enumerate(reviews_int):
            if len(row) != 0:
                features[i, -len(row):] = np.array(row)[:seq_length]
        return features
    def detect_sql_injection(self, text):
        if self.model is None or self.params is None:
            return False, 0.0
        try:
            cleaned_text = self.clean_text(text)
            tokens = word_tokenize(cleaned_text)
            encoded = self.convert_word_to_int(tokens, self.vocab)
            max_len = self.params['max_len']
            padded = self.pad_features([encoded], max_len)
            input_tensor = torch.tensor(padded, dtype=torch.long).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
                probability = torch.sigmoid(output).cpu().numpy()[0]
                prediction = (probability > 0.5).astype(int)
            is_sql_injection = bool(prediction)
            confidence = float(probability)
            return is_sql_injection, confidence
        except Exception as e:
            return False, 0.0

detector = SQLInjectionDetector()

def gradio_login(username, password):
    user_inj, user_conf = detector.detect_sql_injection(username)
    pass_inj, pass_conf = detector.detect_sql_injection(password)
    if user_inj or pass_inj:
        gr.Warning(f"SQL Injection Detected!")
    else:
        gr.Info(f"Safe input. No SQL injection detected.")
    return None

with gr.Blocks(css=".gradio-container {max-width: 350px !important; margin: 40px auto !important;}") as demo:
    gr.Markdown("""
    # Login 
    <div style='text-align:center;'>Enter your username and password</div>
    """)
    with gr.Row():
        with gr.Column():
            username = gr.Textbox(label="Username", placeholder="Enter username")
            password = gr.Textbox(label="Password", type="password", placeholder="Enter password")
            login_btn = gr.Button("Login")
            login_btn.click(fn=gradio_login, inputs=[username, password], outputs=None)

demo.launch() 