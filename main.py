'''
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI(t)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model
model = joblib.load("best_model.pkl")

# Input data structure
class PredictionInput(BaseModel):
    feature1: float = Field(..., ge=0, le=100)
    feature2: float = Field(..., ge=0, le=100)
    # Add more fields as needed...

@app.post("/predict")
def predict(data: PredictionInput):
    input_data = [[data.feature1, data.feature2]]
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}
'''



from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Flight Delay Prediction API", description="Predict flight delay minutes using ML", version="1.0")

# 📦 Load trained model
model = joblib.load("best_model.pkl")

# 🧾 Define request schema
class PredictionRequest(BaseModel):
    Feature1: float
    Feature2: float
    Feature3: float
    # Add all other required features here...

@app.post("/predict")
def predict_delay(data: PredictionRequest):
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)
        return {"predicted_delay_minutes": round(prediction[0], 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
