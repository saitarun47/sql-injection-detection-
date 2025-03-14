import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from keras.models import load_model


class PredictionPipeline:
    def __init__(self):
        self.model = load_model(Path('artifacts/model_trainer/model.keras'))

    
    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction