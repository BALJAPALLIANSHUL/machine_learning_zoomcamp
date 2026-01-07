# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.preprocessing import build_features
from src.model import BinaryClassifier


def train(df: pd.DataFrame):
    X = build_features(df)
    y = (df["Target"] == "Dropout").astype(int)
    
    X = X.replace([float("inf"), float("-inf")], pd.NA)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)

    input_features = X_train.shape[1]
    model = BinaryClassifier(input_features)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    for _ in range(50):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = loss_fn(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    return model, imputer, scaler, X.columns.tolist()


def save_artifacts(model, imputer, scaler, features):
    torch.save(model.state_dict(), "models/model.pt")

    with open("models/imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open("models/features.pkl", "wb") as f:
        pickle.dump(features, f)
