from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import List, Dict
from src.models.train_model import BoutiqueModel

app = FastAPI(
    title="Boutique Sales Forecast API",
    description="API for predicting boutique sales and analyzing historical data",
    version="1.0.0"
)

# Load the trained model
model = BoutiqueModel()
try:
    model.load_model('models/sales_forecast_model.joblib')
except:
    raise Exception("Model file not found. Please train the model first.")

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    date: datetime
    
class PredictionResponse(BaseModel):
    date: datetime
    predicted_sales: float
    
class ModelMetrics(BaseModel):
    metric_name: str
    value: float
    
class FeatureImportance(BaseModel):
    feature_name: str
    importance_score: float

@app.get("/")
async def root():
    return {
        "message": "Welcome to Boutique Sales Forecast API",
        "status": "active",
        "version": "1.0.0"
    }

@app.post("/predict/", response_model=PredictionResponse)
async def predict_sales(request: PredictionRequest):
    try:
        # Create features for the prediction date
        features = pd.DataFrame([{
            'date': request.date,
            'year': request.date.year,
            'month': request.date.month,
            'day_of_month': request.date.day,
            'day_of_week': request.date.weekday(),
            'week_of_year': request.date.isocalendar()[1],
            'is_weekend': 1 if request.date.weekday() >= 5 else 0,
        }])
        
        # Load historical data for calculating rolling features
        historical_data = pd.read_csv('data/processed/daily_metrics.csv')
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        
        # Calculate rolling features
        features['sales_ma_7d'] = historical_data['total_amount'].rolling(window=7).mean().iloc[-1]
        features['sales_ma_30d'] = historical_data['total_amount'].rolling(window=30).mean().iloc[-1]
        features['sales_std_7d'] = historical_data['total_amount'].rolling(window=7).std().iloc[-1]
        
        # Add lag features
        for i in range(1, 8):
            features[f'sales_lag_{i}d'] = historical_data['total_amount'].iloc[-i]
        
        # Make prediction
        prediction = model.predict_sales(features)[0]
        
        return PredictionResponse(
            date=request.date,
            predicted_sales=float(prediction)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/metrics", response_model=List[ModelMetrics])
async def get_model_metrics():
    try:
        # Load historical data
        historical_data = pd.read_csv('data/processed/daily_metrics.csv')
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        
        # Prepare features
        X, y = model.prepare_time_series_data(historical_data)
        predictions = model.predict_sales(historical_data)
        
        # Calculate metrics
        metrics = model.evaluate_model(historical_data['total_amount'].values, predictions)
        
        return [
            ModelMetrics(metric_name=name, value=float(value))
            for name, value in metrics.items()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/feature-importance", response_model=List[FeatureImportance])
async def get_feature_importance():
    try:
        importance = pd.DataFrame({
            'feature': model.feature_columns,
            'importance': model.sales_model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        
        return [
            FeatureImportance(
                feature_name=row['feature'],
                importance_score=float(row['importance'])
            )
            for _, row in importance.iterrows()
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model.sales_model is not None,
        "timestamp": datetime.now().isoformat()
    }
