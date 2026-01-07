# src/inference.py

import torch
import pandas as pd
import numpy as np
from model import BinaryClassifier

class DropoutInferenceModel:
    def __init__(self, model_state_dict, input_features, imputer, scaler, feature_columns):
        self.model = BinaryClassifier(input_features)
        self.model.load_state_dict(model_state_dict)
        self.model.eval()

        self.imputer = imputer
        self.scaler = scaler
        self.feature_columns = feature_columns

    def predict(self, data):
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        df = df.reindex(columns=self.feature_columns, fill_value=np.nan)
        X = self.scaler.transform(self.imputer.transform(df.values))
        X = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            probs = torch.sigmoid(self.model(X))
            preds = (probs > 0.5).float()

        return preds.cpu().numpy().flatten()
