# End-to-End Paddy Yield Prediction API

This project is a complete, production-ready machine learning service to predict paddy yield (in kilograms) based on environmental, agricultural, and weather data.
The service is built around a final, optimized XGBoost Regressor model wrapped inside an sklearn Pipeline, which handles all preprocessing steps automatically.

The pipeline is deployed using FastAPI and fully containerized with Docker for production use.

The final model achieves a Test RMSE of 9101.64 kg (≈ 9.1 tonnes), which we determined to be the maximum predictive power achievable given the cleaned and engineered features.

## Problem Description & Data Analysis

The core challenge of this project was not the model itself, but the extreme data cleaning and feature engineering required.
Several major issues were identified in the raw dataset:

### Target Leakage
11 features (including hectares, seedrate, urea_40days) had a correlation of 0.99 with paddy_yield.

Solution: All 11 leaky features were removed from training.

### High Multicollinearity
Many feature groups (e.g., all drain features, all ai features) were perfectly correlated.

Solution: Each correlated group was reduced using PCA into single components (e.g., rain_pca, ai_pca).

### Skewed / Non-Normal Distributions
Most numerical features had skewed or multimodal distributions, making StandardScaler unsuitable.

Solution: RobustScaler (median and IQR based) was used for all numeric scaling.

### Useless Categorical Features
After one-hot encoding and Mutual Information analysis, many categorical columns had 0.0 importance.

Solution: All zero-score categorical columns were dropped to reduce noise.

The final XGBoost model outperformed RandomForestRegressor and MLPRegressor in both stability and accuracy.

## Running the Project

You can run the project using uv (for development) or Docker (for production).

## Option 1: Run Locally (with uv)

### 1. Create and Activate Environment
uv venv
source .venv/bin/activate

### 2. Install Dependencies
uv pip sync requirements.txt

### 3. Train the Model
python train_paddy_model.py

### 4. Run the API Server
uvicorn predict_api:app --host 0.0.0.0 --port 8000

### 5. Test the API
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @sample_payload.json

## Option 2: Run with Docker (Production)

### 1. Build Docker Image
docker build -t paddy-yield-api .

### 2. Run Docker Container
docker run -d -p 8000:8000 --name paddy-api paddy-yield-api

### 3. Test the API
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @sample_payload.json

## Deployment

This API is deployed on render.

Live Service URL:
https://paddy-yield-api.onrender.com/docs

## Folder Structure
.
├── neural_network_experiment.ipynb
├── training.ipynb
├── preprocessing.ipynb
├── paddy_yield_model_v1.pkl
├── train_pipeline.py
├── predict.py
├── requirements.txt
├── Dockerfile
└── README.md
