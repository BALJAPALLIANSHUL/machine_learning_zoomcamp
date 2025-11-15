import pandas as pd
import joblib
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

# This dictionary maps the original, messy column names to clean, consistent ones.
# This is the most critical part of the preprocessing.
RAW_COLUMN_MAP = {
    'Hectares ': 'hectares',
    'Agriblock': 'agriblock',
    'Variety': 'variety',
    'Soil Types': 'soil_types',
    'Seedrate(in Kg)': 'seedrate(in_kg)',
    'LP_Mainfield(in Tonnes)': 'lp_mainfield(in_tonnes)',
    'Nursery': 'nursery',
    'Nursery area (Cents)': 'nursery_area_(cents)',
    'LP_nurseryarea(in Tonnes)': 'lp_nurseryarea(in_tonnes)',
    'DAP_20days': 'dap_20days',
    'Weed28D_thiobencarb': 'weed28d_thiobencarb',
    'Urea_40Days': 'urea_40days',
    'Potassh_50Days': 'potassh_50days',
    'Micronutrients_70Days': 'micronutrients_70days',
    'Pest_60Day(in ml)': 'pest_60day(in_ml)',
    '30DRain( in mm)': '30drain(_in_mm)',
    '30DAI(in mm)': '30dai(in_mm)',
    '30_50DRain( in mm)': '30_50drain(_in_mm)',
    '30_50DAI(in mm)': '30_50dai(in_mm)',
    '51_70DRain(in mm)': '51_70drain(in_mm)',
    '51_70AI(in mm)': '51_70ai(in_mm)',
    '71_105DRain(in mm)': '71_105drain(in_mm)',
    '71_105DAI(in mm)': '71_105dai(in_mm)',
    'Min temp_D1_D30': 'min_temp_d1_d30',
    'Max temp_D1_D30': 'max_temp_d1_d30',
    'Min temp_D31_D60': 'min_temp_d31_d60',
    'Max temp_D31_D60': 'max_temp_d31_d60',
    'Min temp_D61_D90': 'min_temp_d61_d90',
    'Max temp_D61_D90': 'max_temp_d61_d90',
    'Min temp_D91_D120': 'min_temp_d91_d120',
    'Max temp_D91_D120': 'max_temp_d91_d120',
    'Inst Wind Speed_D1_D30(in Knots)': 'inst_wind_speed_d1_d30(in_knots)',
    'Inst Wind Speed_D31_D60(in Knots)': 'inst_wind_speed_d31_d60(in_knots)',
    'Inst Wind Speed_D61_D90(in Knots)': 'inst_wind_speed_d61_d90(in_knots)',
    'Inst Wind Speed_D91_D120(in Knots)': 'inst_wind_speed_d91_d120(in_knots)',
    'Wind Direction_D1_D30': 'wind_direction_d1_d30',
    'Wind Direction_D31_D60': 'wind_direction_d31_d60',
    'Wind Direction_D61_D90': 'wind_direction_d61_d90',
    'Wind Direction_D91_D120': 'wind_direction_d91_d120',
    'Relative Humidity_D1_D30': 'relative_humidity_d1_d30',
    'Relative Humidity_D31_D60': 'relative_humidity_d31_d60',
    'Relative Humidity_D61_D90': 'relative_humidity_d61_d90',
    'Relative Humidity_D91_D120': 'relative_humidity_d91_d120',
    'Trash(in bundles)': 'trash(in_bundles)',
    'Paddy yield(in Kg)': 'paddy_yield(in_kg)'
}

def load_and_process_data(path, column_map):
    """
    Loads raw data, renames columns, and cleans categorical text.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {path}")
        return None

    # Rename columns to a clean, consistent format
    df = df.rename(columns=column_map)

    # Standardize text data in categorical columns
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')
    
    print("Data loaded and columns standardized.")
    return df

def train_model(df, target_column):
    """
    Defines the full preprocessing pipeline, defines the model,
    and fits the final pipeline on the data.
    """
    print("Defining feature groups and pipeline...")
    
    # 1. Define Column Groups
    LEAKY_FEATURES = [
        'hectares', 'seedrate(in_kg)', 'lp_mainfield(in_tonnes)', 'nursery_area_(cents)', 
        'lp_nurseryarea(in_tonnes)', 'dap_20days', 'weed28d_thiobencarb', 
        'urea_40days', 'potassh_50days', 'micronutrients_70days', 'pest_60day(in_ml)',
        'trash(in_bundles)' # Added this based on our 0.957 correlation finding
    ]

    PCA_RAIN_FEATURES = ['30drain(_in_mm)', '30_50drain(_in_mm)', '51_70drain(in_mm)', '71_105drain(in_mm)']
    PCA_AI_FEATURES = ['30dai(in_mm)', '30_50dai(in_mm)', '51_70ai(in_mm)', '71_105dai(in_mm)']

    TEMP_FEATURES = [
        'min_temp_d1_d30', 'min_temp_d31_d60', 'min_temp_d61_d90', 'min_temp_d91_d120',
        'max_temp_d1_d30', 'max_temp_d31_d60', 'max_temp_d61_d90', 'max_temp_d91_d120'
    ]

    OTHER_NUMERIC_FEATURES = [
        'inst_wind_speed_d1_d30(in_knots)', 'inst_wind_speed_d31_d60(in_knots)',
        'inst_wind_speed_d61_d90(in_knots)', 'inst_wind_speed_d91_d120(in_knots)',
        'relative_humidity_d1_d30', 'relative_humidity_d31_d60',
        'relative_humidity_d61_d90', 'relative_humidity_d91_d120'
    ]

    CATEGORICAL_FEATURES = [
        'agriblock', 'variety', 'soil_types', 'nursery', 
        'wind_direction_d1_d30', 'wind_direction_d31_d60', 
        'wind_direction_d61_d90', 'wind_direction_d91_d120'
    ]
    
    # 2. Define the transformation pipelines
    rain_pca_pipe = Pipeline(steps=[
        ('scaler', RobustScaler()),
        ('pca', PCA(n_components=1, random_state=42)) # Added random_state
    ])

    ai_pca_pipe = Pipeline(steps=[
        ('scaler', RobustScaler()),
        ('pca', PCA(n_components=1, random_state=42)) # Added random_state
    ])

    numeric_pipe = Pipeline(steps=[
        ('scaler', RobustScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # 3. Create the master preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('rain_pca', rain_pca_pipe, PCA_RAIN_FEATURES),
            ('ai_pca', ai_pca_pipe, PCA_AI_FEATURES),
            ('scale_temps', numeric_pipe, TEMP_FEATURES),
            ('scale_other_numeric', numeric_pipe, OTHER_NUMERIC_FEATURES),
            ('encode_cats', categorical_pipe, CATEGORICAL_FEATURES),
            ('drop_leaky', 'drop', LEAKY_FEATURES)
        ],
        remainder='drop' # Drops any columns not explicitly mentioned
    )

    # 4. Define the best XGBoost model
    xgb = XGBRegressor(n_estimators=100,
                max_depth=5,
                learning_rate=0.01,
                min_child_weight=7,
                subsample=1,
                colsample_bytree=0.9,
                objective='reg:squarederror',
                verbosity=2,
                seed=42)

    # 5. Create the Final Production Pipeline
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', xgb)
    ])
    
    # 6. Separate features and target
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # 7. Fit the entire pipeline
    print("Fitting the final pipeline...")
    final_pipeline.fit(X, y)
    print("Pipeline fitting complete.")
    
    return final_pipeline

def save_model(pipeline, path):
    """
    Saves the fitted pipeline to a file using joblib.
    """
    joblib.dump(pipeline, path)
    print(f"Model pipeline saved to {path}")

if __name__ == '__main__':
    # Define file paths and target
    RAW_DATA_PATH = 'paddydataset.csv'
    MODEL_OUTPUT_PATH = 'paddy_yield_model_v1.pkl'
    TARGET_COLUMN = 'paddy_yield(in_kg)'
    
    # Run the full pipeline
    df = load_and_process_data(RAW_DATA_PATH, RAW_COLUMN_MAP)
    
    if df is not None:
        pipeline = train_model(df, TARGET_COLUMN)
        save_model(pipeline, MODEL_OUTPUT_PATH)
        
        print("\n--- All Done ---")
        print('pandas version:', pd.__version__)
        print('scikit-learn version:', sklearn.__version__)
        print('joblib version:', joblib.__version__)