# src/inference.py

import pickle
import torch
import pandas as pd

from src.model import BinaryClassifier

ARTIFACT_DIR = "models"


class InferencePipeline:
    def __init__(self):
        with open(f"{ARTIFACT_DIR}/features.pkl", "rb") as f:
            self.feature_columns = pickle.load(f)

        with open(f"{ARTIFACT_DIR}/imputer.pkl", "rb") as f:
            self.imputer = pickle.load(f)

        with open(f"{ARTIFACT_DIR}/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        self.model = BinaryClassifier(len(self.feature_columns))
        self.model.load_state_dict(
            torch.load(f"{ARTIFACT_DIR}/model.pt", map_location="cpu")
        )
        self.model.eval()

    def predict_proba(self, encoded_features: dict) -> float:
        """
        encoded_features: must match X_train_new columns (175 features)
        """
        X = pd.DataFrame([encoded_features])

        # enforce training-time feature order
        X = X.reindex(columns=self.feature_columns, fill_value=0)

        X = self.imputer.transform(X)
        X = self.scaler.transform(X)

        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            prob = torch.sigmoid(self.model(X_tensor)).item()

        return prob
