# SQL Injection Detection System

A robust machine learning based SQL injection detection system built with PyTorch, using a LSTM neural network architecture and a Gradio web interface.

## Project Overview

This project implements an SQL injection detection system that leverages deep learning techniques to identify malicious SQL queries in real time. The system uses a LSTM neural network trained on a comprehensive dataset of SQL injection patterns and benign queries.

## Key Features

- **Deep Learning Model**: LSTM-based neural network with embedding layers
- **Real-time Detection**: Instant analysis of user inputs
- **Web Interface**: Gradio-based UI for easy interaction
- **Comprehensive Pipeline**: End-to-end ML pipeline with data ingestion, validation, transformation, training, and evaluation
- **Production Ready**: Docker containerization and deployment ready
- **MLOps Integration**: MLflow tracking and experiment management

## Architecture

### Model Architecture
- **Embedding Layer**: 64-dimensional word embeddings
- **LSTM Layers**: 2-layer LSTM with 32 hidden units
- **Dropout**: 0.5 dropout rate for regularization
- **Output Layer**: Sigmoid activation for binary classification

### Pipeline Components
```
src/
├── components/
│   ├── data_ingestion.py      # Data collection and preprocessing
│   ├── data_validation.py     # Data validation
│   ├── data_transformation.py # Feature engineering and tokenization
│   ├── model_trainer.py       # LSTM model training
│   └── model_evaluation.py    # Model performance evaluation
├── pipeline/                  # Pipeline orchestration
├── config/                    # Configuration management
└── utils/                     # Utility functions
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
PyTorch
Gradio
MLflow
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd SQL-Injection-Detection

# Install dependencies
pip install -r requirements.txt

# Run the training pipeline
python main.py

# Launch the web interface
python app.py
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

## Model Performance

The LSTM model achieves excellent performance on SQL injection detection:

- **Accuracy**: High classification accuracy on test data (~96%)
- **F1-score**: ~92 %
- **Real-time Processing**: Sub-second response times
- **Low False Positives**: Minimal impact on legitimate queries

## Configuration

Key model parameters in `params.yaml`:
```yaml
model_trainer:
  embed_dim: 64
  hidden_dim: 32
  dropout_rate: 0.5
  epochs: 30
  batch_size: 32
  learning_rate: 0.001
```

## Project Structure

```
├── app.py                     # Gradio web interface
├── main.py                    # Training pipeline orchestration
├── src/                       # Source code
│   ├── components/           # ML pipeline components
│   ├── pipeline/            # Pipeline stages
│   ├── config/              # Configuration management
│   └── utils/               # Utility functions
├── artifacts/               # Model artifacts and data
├── config/                  # Configuration files
├── mlruns/                  # MLflow experiment tracking

```



##  Detection Capabilities

The system detects various SQL injection patterns:
- **Union-based attacks**: `UNION SELECT * FROM users`
- **Boolean-based attacks**: `OR 1=1`, `AND 1=1`
- **Time-based attacks**: `WAITFOR DELAY '0:0:5'`
- **Error-based attacks**: `CONVERT(int,@@version)`
- **Stacked queries**: `; DROP TABLE users`

## Training Pipeline

The complete ML pipeline includes:

1. **Data Ingestion**: Downloads and processes SQL injection dataset
2. **Data Validation**: Ensures data quality and consistency
3. **Data Transformation**: Tokenization and feature engineering
4. **Model Training**: LSTM model training with hyperparameter optimization
5. **Model Evaluation**: Performance metrics and validation

## Technologies Used

- **PyTorch**: Deep learning framework
- **Gradio**: Web interface framework
- **MLflow**: Experiment tracking and model management
- **NLTK**: Natural language processing
- **Docker**: Containerization


