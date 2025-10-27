import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd
from src.logger import logging 
from src.exception import MyException
from src.pipline.prediction_pipeline import HeartPatientData, HeartRiskClassifier
from fastapi.responses import FileResponse

# Initialize FastAPI app
app = FastAPI(
    title="Heart Risk Prediction API",
    description="API for predicting heart disease risk in patients",
    version="1.0.0"
)

# Add this line after creating app
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add this route
@app.get("/")
async def serve_home():
    return FileResponse("static/index.html")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates (optional)
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# Initialize prediction pipeline
classifier = HeartRiskClassifier()

# Pydantic model for request validation
class HeartPatientRequest(BaseModel):
    gender: str = Field(..., description="Patient gender (M/F)")
    anchor_age: int = Field(..., ge=0, le=120, description="Patient age")
    admission_type: str = Field(..., description="Type of admission")
    insurance: str = Field(..., description="Insurance type")
    race: str = Field(..., description="Patient race")
    marital_status: str = Field(..., description="Marital status")
    creatinine: float = Field(..., ge=0, description="Creatinine level (mg/dL)")
    glucose: float = Field(..., ge=0, description="Glucose level (mg/dL)")
    sodium: float = Field(..., ge=0, description="Sodium level (mEq/L)")
    potassium: float = Field(..., ge=0, description="Potassium level (mEq/L)")
    troponin_t: float = Field(..., ge=0, description="Troponin T level (ng/mL)")
    creatine_kinase_mb: float = Field(..., ge=0, description="Creatine Kinase MB (ng/mL)")
    hemoglobin: float = Field(..., ge=0, description="Hemoglobin (g/dL)")
    white_blood_cells: float = Field(..., ge=0, description="White blood cell count (K/uL)")
    heart_rate: float = Field(..., ge=0, description="Heart rate (bpm)")
    bp_systolic: float = Field(..., ge=0, description="Systolic blood pressure (mmHg)")
    bp_diastolic: float = Field(..., ge=0, description="Diastolic blood pressure (mmHg)")
    spo2: float = Field(..., ge=0, le=100, description="Oxygen saturation (%)")
    respiratory_rate: float = Field(..., ge=0, description="Respiratory rate (breaths/min)")
    temperature: float = Field(..., ge=0, description="Body temperature (°C)")

    class Config:
        schema_extra = {
            "example": {
                "gender": "M",
                "anchor_age": 65,
                "admission_type": "EMERGENCY",
                "insurance": "Medicare",
                "race": "WHITE",
                "marital_status": "MARRIED",
                "creatinine": 1.2,
                "glucose": 120.0,
                "sodium": 140.0,
                "potassium": 4.0,
                "troponin_t": 0.15,
                "creatine_kinase_mb": 6.5,
                "hemoglobin": 13.5,
                "white_blood_cells": 8.5,
                "heart_rate": 85.0,
                "bp_systolic": 130.0,
                "bp_diastolic": 80.0,
                "spo2": 97.0,
                "respiratory_rate": 16.0,
                "temperature": 37.0
            }
        }

class PredictionResponse(BaseModel):
    prediction: str
    risk_status: str
    message: str

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint"""
    return """
    <html>
        <head>
            <title>Heart Risk Prediction API</title>
        </head>
        <body>
            <h1>Heart Risk Prediction API</h1>
            <p>Welcome to the Heart Risk Prediction API</p>
            <p>Visit <a href="/docs">/docs</a> for API documentation</p>
            <p>Visit <a href="/health">/health</a> to check API health</p>
        </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        logging.info("Health check requested")
        return {
            "status": "healthy",
            "message": "Heart Risk Prediction API is running",
            "version": "1.0.0"
        }
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_request: HeartPatientRequest):
    try:
        logging.info("Received prediction request")
        logging.info(f"Patient data: Age={patient_request.anchor_age}, Gender={patient_request.gender}")
        
        # Create HeartPatientData object
        patient_data = HeartPatientData(
            gender=patient_request.gender,
            anchor_age=patient_request.anchor_age,
            admission_type=patient_request.admission_type,
            insurance=patient_request.insurance,
            race=patient_request.race,
            marital_status=patient_request.marital_status,
            creatinine=patient_request.creatinine,
            glucose=patient_request.glucose,
            sodium=patient_request.sodium,
            potassium=patient_request.potassium,
            troponin_t=patient_request.troponin_t,
            creatine_kinase_mb=patient_request.creatine_kinase_mb,
            hemoglobin=patient_request.hemoglobin,
            white_blood_cells=patient_request.white_blood_cells,
            heart_rate=patient_request.heart_rate,
            bp_systolic=patient_request.bp_systolic,
            bp_diastolic=patient_request.bp_diastolic,
            spo2=patient_request.spo2,
            respiratory_rate=patient_request.respiratory_rate,
            temperature=patient_request.temperature
        )
        
        logging.info("Patient data object created successfully")
        
        # Get dataframe
        patient_df = patient_data.get_heart_input_data_frame()
        logging.info(f"Patient dataframe shape: {patient_df.shape}")
        
        # Make prediction
        prediction_result = classifier.predict(patient_df)
        logging.info(f"Prediction completed: {prediction_result}")
        
        # Determine risk status
        risk_status = "HIGH RISK" if "has heart risk" in prediction_result else "LOW RISK"
        
        return PredictionResponse(
            prediction=prediction_result,
            risk_status=risk_status,
            message="Prediction completed successfully"
        )
        
    except MyException as e:
        logging.error(f"MyException in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    except Exception as e:
        logging.error(f"Unexpected error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(patients: list[HeartPatientRequest]):
    """
    Predict heart disease risk for multiple patients
    
    Returns:
        List of predictions
    """
    try:
        logging.info(f"Received batch prediction request for {len(patients)} patients")
        
        results = []
        for idx, patient_request in enumerate(patients):
            try:
                # Create HeartPatientData object
                patient_data = HeartPatientData(
                    gender=patient_request.gender,
                    anchor_age=patient_request.anchor_age,
                    admission_type=patient_request.admission_type,
                    insurance=patient_request.insurance,
                    race=patient_request.race,
                    marital_status=patient_request.marital_status,
                    creatinine=patient_request.creatinine,
                    glucose=patient_request.glucose,
                    sodium=patient_request.sodium,
                    potassium=patient_request.potassium,
                    troponin_t=patient_request.troponin_t,
                    creatine_kinase_mb=patient_request.creatine_kinase_mb,
                    hemoglobin=patient_request.hemoglobin,
                    white_blood_cells=patient_request.white_blood_cells,
                    heart_rate=patient_request.heart_rate,
                    bp_systolic=patient_request.bp_systolic,
                    bp_diastolic=patient_request.bp_diastolic,
                    spo2=patient_request.spo2,
                    respiratory_rate=patient_request.respiratory_rate,
                    temperature=patient_request.temperature
                )
                
                # Get dataframe and predict
                patient_df = patient_data.get_heart_input_data_frame()
                prediction_result = classifier.predict(patient_df)
                risk_status = "HIGH RISK" if "has heart risk" in prediction_result else "LOW RISK"
                
                results.append({
                    "patient_index": idx,
                    "prediction": prediction_result,
                    "risk_status": risk_status,
                    "status": "success"
                })
                
            except Exception as e:
                logging.error(f"Error processing patient {idx}: {str(e)}")
                results.append({
                    "patient_index": idx,
                    "prediction": None,
                    "risk_status": None,
                    "status": "failed",
                    "error": str(e)
                })
        
        logging.info(f"Batch prediction completed: {len(results)} results")
        return {
            "total_patients": len(patients),
            "successful_predictions": sum(1 for r in results if r["status"] == "success"),
            "failed_predictions": sum(1 for r in results if r["status"] == "failed"),
            "results": results
        }
        
    except Exception as e:
        logging.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logging.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "message": "An internal error occurred",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )