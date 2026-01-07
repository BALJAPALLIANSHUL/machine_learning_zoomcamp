# Student Dropout Prediction – ML Zoomcamp Capstone 1

## 1. Problem Description

Student dropout is a major challenge for higher education institutions. Early prediction of students at risk allows universities to intervene through academic support, counseling, or financial aid.

The goal of this project is to build a **binary classification model** that predicts whether a student will **drop out (1)** or **continue their studies (0)** based on academic, demographic, and socio-economic data.

This project was developed as **Capstone Project 1** for **Machine Learning Zoomcamp 2025**.

---

## 2. Dataset

- **Source**: UCI – Predict Students Dropout and Academic Success
- **Samples**: ~4,400 students
- **Target Variable**:
  - `Dropout` → 1
  - `Graduate / Enrolled` → 0

### Feature Categories
- Academic performance (grades, approvals, enrollment counts)
- Demographics (age, gender, nationality)
- Financial indicators (debtor status, scholarship holder)
- Engineered features (approval rates, financial risk, course load pressure)
- One-hot encoded categorical features

Final model input size: **175 features**.

---

## 3. Project Structure

```
capstone-1/
├── Dockerfile
├── README.md
├── requirements.txt
├── pyproject.toml
├── uv.lock
├── data/
├── notebooks/
│   └── notebook.ipynb
├── models/
│   ├── model.pt
│   ├── imputer.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   └── metadata.pkl
└── src/
    ├── app.py
    ├── inference.py
    ├── preprocessing.py
    ├── model.py
    └── predict.py
```

---

## 4. Model

- **Algorithm**: Feed-forward Neural Network (PyTorch)
- **Loss Function**: BCEWithLogitsLoss
- **Optimizer**: Adam
- **Regularization**:
  - Dropout layers
  - L2 weight decay
- **Early Stopping** based on validation loss

The model is trained **only in a notebook environment (Google Colab)** and exported for inference.

---

## 5. Training Pipeline

1. Load raw CSV data
2. Decode categorical variables using mapping dictionaries
3. Feature engineering:
   - Approval rates
   - Financial risk indicators
   - Course load pressure
   - Age groups
   - Parental education indicators
4. One-hot encoding (`pd.get_dummies`, `drop_first=True`)
5. Train/validation/test split
6. Median imputation
7. Standard scaling
8. PyTorch training with early stopping
9. Save trained artifacts

---

## 6. Inference Pipeline

At inference time:

1. Raw JSON input is converted to a DataFrame
2. Identical feature engineering logic is applied
3. One-hot encoding
4. Feature alignment using `features.pkl`
5. Imputation and scaling using saved artifacts
6. PyTorch model inference
7. Sigmoid activation to obtain probability

All preprocessing steps are **identical to training**.

---

## 7. API Design (FastAPI)

### Health Check

```
GET /health
```

Response:
```json
{ "status": "ok" }
```

### Prediction Endpoint

```
POST /predict
```

Example request:
```json
{
  "Age at enrollment": 19,
  "Admission grade": 150,
  "Curricular units 1st sem (enrolled)": 6,
  "Curricular units 1st sem (approved)": 6,
  "Curricular units 2nd sem (enrolled)": 6,
  "Curricular units 2nd sem (approved)": 6,
  "Debtor": 0,
  "Scholarship holder": 1,
  "Gender": 1,
  "Daytime/evening attendance": 1
}
```

Example response:
```json
{
  "dropout_probability": 0.22,
  "dropout": false
}
```

---

## 8. Deployment (AWS EC2)

- **Platform**: AWS EC2 (Ubuntu)
- **Instance Type**: Free-tier compatible
- **Serving**: FastAPI + Uvicorn
- **Containerization**: Docker
- **Model**: CPU-only PyTorch

Run locally or on EC2:

```bash
docker build -t dropout-api .
docker run -p 8000:8000 dropout-api
```

API available at: (not ec2 but render because free tier is expired)
```
 https://student-dropout-prediction-ke7m.onrender.com/docs
```

---

## 9. Reproducibility

- Fixed random seeds
- Saved preprocessing artifacts
- Deterministic inference pipeline
- Dependency versions pinned

---

## 10. Limitations

- Model trained on historical data from a single institution
- Predictions are probabilistic, not deterministic outcomes
- Requires periodic retraining for long-term use

---

## 11. Future Improvements

- Model explainability (SHAP)
- Batch prediction support
- Monitoring and drift detection
- CI/CD pipeline

---

## 12. Author

**Anshul Patel**  
Machine Learning Zoomcamp 2025  
Capstone Project 1
