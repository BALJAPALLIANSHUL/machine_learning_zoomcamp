from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import pickle
import fastapi as fapi
from pydantic import BaseModel, Field, field_validator

MODEL_PATH = Path(__file__).parent / "model.pkl"

app = fapi.FastAPI(title="Churn prediction service")


# categorical fields only (avoid mangling numeric fields)
CATEGORICAL_FIELDS = [
    "gender", "partner", "dependents", "phoneservice", "multiplelines",
    "internetservice", "onlinesecurity", "onlinebackup", "deviceprotection",
    "techsupport", "streamingtv", "streamingmovies", "contract",
    "paperlessbilling", "paymentmethod"
]


class Customer(BaseModel):
    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    tenure: int = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0.0)
    totalcharges: float = Field(..., ge=0.0)

    # normalize only categorical fields before parsing
    @field_validator(
        "gender","partner","dependents","phoneservice","multiplelines",
        "internetservice","onlinesecurity","onlinebackup","deviceprotection",
        "techsupport","streamingtv","streamingmovies","contract",
        "paperlessbilling","paymentmethod",
        mode="before"
    )
    @classmethod
    def normalize_categorical(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return v.strip().lower().replace(" ", "_")
        return v


class PredictionResponse(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
    churn: bool
    mail: bool


_model = None  # will hold the loaded pipeline


@asynccontextmanager
async def lifespan(app: fapi.FastAPI):
    global _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    # load once on startup
    with open(MODEL_PATH, "rb") as f_in:
        _model = pickle.load(f_in)
    yield


app = fapi.FastAPI(lifespan=lifespan)


@app.get("/", tags=["health"])
def health():
    return {"status": "ok"}

def _positive_index_from_model():
    classes = getattr(_model, "classes_", None)
    if classes is not None:
        try:
            return list(classes).index(1)
        except ValueError:
            return 1
    return 1


@app.post("/predict", response_model=PredictionResponse, tags=["predict"])
def predict(customer: Customer):
    if _model is None:
        raise fapi.HTTPException(status_code=500, detail="Model not loaded")
    rec = customer.model_dump()  # pydantic v2; use .dict() if v1
    probs = _model.predict_proba([rec])
    pos_idx = _positive_index_from_model()
    prob = float(probs[0, pos_idx])
    churn_flag = prob >= 0.5
    return PredictionResponse(probability=prob, churn=churn_flag, mail=churn_flag)


@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["predict"])
def predict_batch(customers: List[Customer]):
    if _model is None:
        raise fapi.HTTPException(status_code=500, detail="Model not loaded")
    dicts = [c.model_dump() for c in customers]
    probs = _model.predict_proba(dicts)
    pos_idx = _positive_index_from_model()
    results: List[PredictionResponse] = []
    for p in probs:
        prob = float(p[pos_idx])
        churn_flag = prob >= 0.5
        results.append(PredictionResponse(probability=prob, churn=churn_flag, mail=churn_flag))
    return results