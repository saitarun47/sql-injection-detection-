import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from src import logger
from src.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
       
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1).values
        test_x = test_data.drop([self.config.target_column], axis=1).values
        train_y = train_data[[self.config.target_column]].values
        test_y = test_data[[self.config.target_column]].values

        scaler = MinMaxScaler()
        train_x = scaler.fit_transform(train_x)
        test_x = scaler.transform(test_x)

        
        train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
        test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

       
        scaler_path = os.path.join(self.config.root_dir, "scaler.pkl")
        joblib.dump(scaler, scaler_path)

    
        mlflow.set_tracking_uri("http://localhost:5000")

        with mlflow.start_run():
            
            model = Sequential([
                LSTM(self.config.lstm_units, activation="relu", return_sequences=True, input_shape=(1, train_x.shape[2])),
                Dropout(self.config.dropout_rate),
                LSTM(self.config.lstm_units, activation="relu"),
                Dropout(self.config.dropout_rate),
                Dense(1, activation="sigmoid")
            ])

            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            class_weights = {0: 1.0, 1: 3.0}  

            model.fit(train_x, train_y, epochs=self.config.epochs, batch_size=self.config.batch_size, validation_data=(test_x, test_y), verbose=1,class_weight=class_weights)

            loss, accuracy = model.evaluate(test_x, test_y, verbose=0)

           
            model_path = os.path.join(self.config.root_dir, "model.keras")
            model.save(model_path)

            
            mlflow.keras.log_model(model, "model")
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("loss", loss)

            logger.info(f"Model trained with accuracy: {accuracy:.4f}")
            print(f"Model trained successfully with accuracy: {accuracy:.4f}")

        print("MLflow tracking complete!")
