import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd
from src.logger import logging 
from src.exception import MyException
from src.pipline.prediction_pipeline import HeartPatientData, HeartRiskClassifier
import os

# Initialize FastAPI app
app = FastAPI(
    title="Heart Risk Prediction API",
    description="API for predicting heart disease risk in patients",
    version="1.0.0"
)

# Add CORS middleware - MUST BE BEFORE ROUTES
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (if static folder exists)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

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
    temperature: float = Field(..., ge=0, description="Body temperature (°F)")

    class Config:
        # Fixed for Pydantic V2
        json_schema_extra = {
            "example": {
                "gender": "M",
                "anchor_age": 86,
                "admission_type": "URGENT",
                "insurance": "Medicare",
                "race": "WHITE",
                "marital_status": "SINGLE",
                "creatinine": 1.516,
                "glucose": 122.166,
                "sodium": 141.666,
                "potassium": 4.016,
                "troponin_t": 1.007,
                "creatine_kinase_mb": 13.0,
                "hemoglobin": 8.2,
                "white_blood_cells": 5.15,
                "heart_rate": 73.0,
                "bp_systolic": 142.0,
                "bp_diastolic": 83.0,
                "spo2": 98.0,
                "respiratory_rate": 20.0,
                "temperature": 98.0
            }
        }

class PredictionResponse(BaseModel):
    prediction: str
    risk_status: str
    message: str = "Prediction completed successfully"

# Add this route to your app
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico") if os.path.exists("static/favicon.ico") else None

# Root endpoint - Serve HTML or default message
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - serves static HTML if available"""
    try:
        # Try to serve index.html from static folder
        if os.path.exists("static/index.html"):
            return FileResponse("static/index.html")
        else:
            # Return default HTML
            return """
            <html>
                <head>
                    <title>Heart Risk Prediction API</title>
                    <style>
                        body { 
                            font-family: Arial, sans-serif; 
                            max-width: 800px; 
                            margin: 50px auto; 
                            padding: 20px;
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            color: white;
                        }
                        .container {
                            background: white;
                            color: #333;
                            padding: 40px;
                            border-radius: 10px;
                            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                        }
                        h1 { color: #667eea; }
                        a { color: #667eea; text-decoration: none; font-weight: bold; }
                        a:hover { text-decoration: underline; }
                        .status { 
                            display: inline-block;
                            background: #51cf66;
                            color: white;
                            padding: 5px 15px;
                            border-radius: 20px;
                            font-size: 0.9em;
                        }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>❤️ Heart Risk Prediction API <span class="status">ONLINE</span></h1>
                        <p><strong>Version:</strong> 1.0.0</p>
                        <p>Welcome to the Heart Risk Prediction API. This service uses machine learning to assess cardiac risk.</p>
                        <hr>
                        <h3>📚 Available Endpoints:</h3>
                        <ul>
                            <li><a href="/docs">📖 API Documentation (Swagger UI)</a></li>
                            <li><a href="/redoc">📘 ReDoc Documentation</a></li>
                            <li><a href="/health">💚 Health Check</a></li>
                            <li><strong>POST /predict</strong> - Single patient prediction</li>
                            <li><strong>POST /batch_predict</strong> - Batch prediction</li>
                        </ul>
                        <hr>
                        <h3>🧪 Test the API:</h3>
                        <p>Use the interactive documentation at <a href="/docs">/docs</a> to test predictions</p>
                    </div>
                </body>
            </html>
            """
    except Exception as e:
        logging.error(f"Error serving root: {str(e)}")
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        logging.info("Health check requested")
        return {
            "status": "healthy",
            "message": "Heart Risk Prediction API is running",
            "version": "1.0.0",
            "model_loaded": classifier is not None
        }
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

@app.post("/predict", response_model=PredictionResponse)
async def predict(patient_request: HeartPatientRequest):
    """
    Predict heart disease risk for a single patient
    
    Args:
        patient_request: Patient data
        
    Returns:
        Prediction result with risk status
    """
    try:
        logging.info("=" * 50)
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
        logging.info(f"Patient dataframe columns: {patient_df.columns.tolist()}")
        
        # Make prediction
        prediction_result = classifier.predict(patient_df)
        logging.info(f"Prediction completed: {prediction_result}")
        
        # Determine risk status
        risk_status = "HIGH RISK" if "has heart risk" in prediction_result.lower() else "LOW RISK"
        
        response = PredictionResponse(
            prediction=prediction_result,
            risk_status=risk_status,
            message="Prediction completed successfully"
        )
        
        logging.info(f"Response: {response}")
        logging.info("=" * 50)
        
        return response
        
    except MyException as e:
        logging.error(f"MyException in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    except Exception as e:
        logging.error(f"Unexpected error in prediction: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(patients: list[HeartPatientRequest]):
    """
    Predict heart disease risk for multiple patients
    
    Args:
        patients: List of patient data
        
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
                risk_status = "HIGH RISK" if "has heart risk" in prediction_result.lower() else "LOW RISK"
                
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
    import traceback
    logging.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "message": "An internal error occurred",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        # reload=False,  # Set to False for production
        log_level="info"
    )