import pandas as pd
import joblib
import fastapi as fapi
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

# --- Configuration ---

# Set the path to your saved model file
MODEL_PATH = Path(__file__).parent / "paddy_yield_model_v1.pkl"

# List of all categorical fields that need text normalization
# This must match the list used in your training pipeline
CATEGORICAL_FIELDS = [
    'agriblock', 'variety', 'soil_types', 'nursery', 
    'wind_direction_d1_d30', 'wind_direction_d31_d60', 
    'wind_direction_d61_d90', 'wind_direction_d91_d120'
]

# This global variable will hold the loaded model pipeline
_model = None

# --- Pydantic Data Models ---

class FarmData(BaseModel):
    """Pydantic model for a single raw data entry from a farm."""
    
    # We must define ALL 47 raw features that the pipeline expects.
    # --- Leaky Features (will be dropped by preprocessor) ---
    hectares: float = Field(..., ge=0)
    seedrate_in_kg: float = Field(..., ge=0, alias='seedrate(in_kg)')
    lp_mainfield_in_tonnes: float = Field(..., ge=0, alias='lp_mainfield(in_tonnes)')
    nursery_area_cents: float = Field(..., ge=0, alias='nursery_area_(cents)')
    lp_nurseryarea_in_tonnes: float = Field(..., ge=0, alias='lp_nurseryarea(in_tonnes)')
    dap_20days: float = Field(..., ge=0)
    weed28d_thiobencarb: float = Field(..., ge=0)
    urea_40days: float = Field(..., ge=0)
    potassh_50days: float = Field(..., ge=0)
    micronutrients_70days: float = Field(..., ge=0)
    pest_60day_in_ml: float = Field(..., ge=0, alias='pest_60day(in_ml)')
    trash_in_bundles: float = Field(..., ge=0, alias='trash(in_bundles)')

    # --- PCA Rain/AI Features ---
    drain_30_mm: float = Field(..., ge=0, alias='30drain(_in_mm)')
    dai_30_mm: float = Field(..., ge=0, alias='30dai(in_mm)')
    drain_30_50_mm: float = Field(..., ge=0, alias='30_50drain(_in_mm)')
    dai_30_50_mm: float = Field(..., ge=0, alias='30_50dai(in_mm)')
    drain_51_70_mm: float = Field(..., ge=0, alias='51_70drain(in_mm)')
    ai_51_70_mm: float = Field(..., ge=0, alias='51_70ai(in_mm)')
    drain_71_105_mm: float = Field(..., ge=0, alias='71_105drain(in_mm)')
    ai_71_105_mm: float = Field(..., ge=0, alias='71_105dai(in_mm)')

    # --- Temp Features ---
    min_temp_d1_d30: float
    max_temp_d1_d30: float
    min_temp_d31_d60: float
    max_temp_d31_d60: float
    min_temp_d61_d90: float
    max_temp_d61_d90: float
    min_temp_d91_d120: float
    max_temp_d91_d120: float

    # --- Other Numeric Features ---
    inst_wind_speed_d1_d30_knots: float = Field(..., ge=0, alias='inst_wind_speed_d1_d30(in_knots)')
    inst_wind_speed_d31_d60_knots: float = Field(..., ge=0, alias='inst_wind_speed_d31_d60(in_knots)')
    inst_wind_speed_d61_d90_knots: float = Field(..., ge=0, alias='inst_wind_speed_d61_d90(in_knots)')
    inst_wind_speed_d91_d120_knots: float = Field(..., ge=0, alias='inst_wind_speed_d91_d120(in_knots)')
    relative_humidity_d1_d30: float = Field(..., ge=0)
    relative_humidity_d31_d60: float = Field(..., ge=0)
    relative_humidity_d61_d90: float = Field(..., ge=0)
    relative_humidity_d91_d120: float = Field(..., ge=0)

    # --- Categorical Features ---
    agriblock: str
    variety: str
    soil_types: str
    nursery: str
    wind_direction_d1_d30: str
    wind_direction_d31_d60: str
    wind_direction_d61_d90: str
    wind_direction_d91_d120: str
    
    # This validator cleans all incoming categorical text data
    @field_validator(*CATEGORICAL_FIELDS, mode="before")
    @classmethod
    def normalize_categorical(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            # Apply the same cleaning as in training: lower, strip, replace space
            return v.strip().lower().replace(" ", "_")
        return v
    
    # Use Pydantic's config to allow aliases
    class Config:
        populate_by_name = True

class YieldPredictionResponse(BaseModel):
    """Pydantic model for the prediction response."""
    predicted_yield_kg: float = Field(..., ge=0.0)

# --- FastAPI App Lifespan (Model Loading) ---

@asynccontextmanager
async def lifespan(app: fapi.FastAPI):
    """Loads the model pipeline on app startup."""
    global _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    print(f"Loading model from {MODEL_PATH}...")
    _model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
    yield
    # (No cleanup needed on shutdown)

app = fapi.FastAPI(
    title="Paddy Yield Prediction Service",
    description="API for predicting paddy yield based on farm data.",
    lifespan=lifespan
)

# --- API Endpoints ---

@app.get("/", tags=["Health Check"])
def health():
    """Health check endpoint to ensure the service is running."""
    return {"status": "ok", "model_loaded": _model is not None}

@app.post("/predict", response_model=YieldPredictionResponse, tags=["Predictions"])
def predict(farm_data: FarmData):
    """
    Predicts the paddy yield for a single farm entry.
    
    The input must be a JSON object matching the `FarmData` schema.
    """
    if _model is None:
        raise fapi.HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 1. Convert the Pydantic model to a dictionary.
        #    *** FIX: Use by_alias=True to get the clean column names ***
        #    that the pipeline was trained on.
        raw_data_dict = farm_data.model_dump(by_alias=True)

        # 2. Convert the single dict to a pandas DataFrame (1 row)
        raw_data_df = pd.DataFrame([raw_data_dict])
        
        # 3. Get the prediction
        prediction = _model.predict(raw_data_df)
        
        # 4. Extract the single prediction value
        yield_kg = float(prediction[0])
        
        return YieldPredictionResponse(predicted_yield_kg=yield_kg)

    except Exception as e:
        print(f"Error during prediction: {e}")
        raise fapi.HTTPException(status_code=400, detail=f"Error processing input: {e}")

@app.post("/predict/batch", response_model=List[YieldPredictionResponse], tags=["Predictions"])
def predict_batch(farm_data_list: List[FarmData]):
    """
    Predicts the paddy yield for a batch of farm entries.
    
    The input must be a JSON array of objects, each matching the `FarmData` schema.
    """
    if _model is None:
        raise fapi.HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 1. Convert the list of Pydantic models to a list of dicts
        #    *** FIX: Use by_alias=True to get the clean column names ***
        dicts = [farm.model_dump(by_alias=True) for farm in farm_data_list]
        
        # 2. Convert the list of dicts to a DataFrame
        raw_data_df = pd.DataFrame(dicts)
        
        # 3. Get all predictions at once
        predictions = _model.predict(raw_data_df)
        
        # 4. Format the response
        results = [
            YieldPredictionResponse(predicted_yield_kg=float(p)) for p in predictions
        ]
        return results

    except Exception as e:
        print(f"Error during batch prediction: {e}")
        raise fapi.HTTPException(status_code=400, detail=f"Error processing input: {e}")