# 🫀 HEART-FAILURE RISK PREDICTION IN ICU PATIENTS

### **A machine learning system to predict heart-attack risk in ICU patients using clinical data to provide early warnings for doctors.**

---

## 📌 Overview

Early detection of heart attacks in ICU patients is challenging because clinical data is massive, complex, and continuously changing.  
This project builds an automated **early-warning ML system** that predicts the risk of a heart attack using patient vitals, lab values, and demographics.

The solution integrates **full MLOps pipeline** including:
- Data ingestion  
- Data validation & transformation  
- ML model training & tuning  
- MLflow experiment tracking  
- DVC for data versioning  
- AWS S3 for storage  
- Docker containerization  
- CI/CD with GitHub Actions  
- Kubernetes (EKS) deployment  
- Monitoring with Prometheus + Grafana  

---

## 📚 Dataset Used

**Source:** MIMIC-IV v3.1 (PhysioNet)  
**Raw data size:** ~60 GB across tables  
**Tables used:**  
- `patients`  
- `admissions`  
- `diagnoses_icd`  
- `labevents`  
- `chartevents`

**Final curated dataset:**
- **33,018 patients**  
- 16,509 heart-attack cases  
- 16,509 matched controls  
- **23 clinical features** (labs, vitals, demographics)

---

## 🔬 Core Features

### **Lab Features**
- Creatinine  
- Glucose  
- Sodium  
- Potassium  
- Troponin-T  
- CK-MB  
- Hemoglobin  
- WBC

### **Vital Features**
- Heart Rate  
- BP Systolic  
- BP Diastolic  
- SpO2  
- Respiratory Rate  
- Temperature

### **Demographics**
- Age  
- Gender  
- Race  
- Admission Type  
- Insurance  
- Marital Status  

---

## 🧪 Machine Learning Models

All major ML models were tested:
- Logistic Regression  
- Random Forest  
- Extra Trees  
- KNN  
- SVM  
- XGBoost  
- LightGBM  
- CatBoost  
- Gradient Boosting  
- AdaBoost  
- Naive Bayes  

### ⭐ Best Models (After Hyperparameter Tuning)
| Model        | Accuracy | AUC    | F1     | Precision | Recall |
|--------------|----------|--------|--------|-----------|--------|
| **XGBoost**  | **0.9412** | 0.9825 | 0.9401 | 0.9588    | 0.9222 |
| **CatBoost** | 0.9356   | **0.9830** | 0.9344 | 0.9534    | 0.9161 |

---

## ⚙️ Tech Stack

### **Languages & Tools**
- Python  
- Pandas, NumPy, Scikit-learn  
- XGBoost, LightGBM, CatBoost  
- DuckDB  
- MLflow & Dagshub  
- FastAPI  
- Docker  
- DVC  
- GitHub Actions  
- AWS S3, ECR, EC2, EKS  
- Prometheus, Grafana  

---

## 🔄 Project Workflow (MLOps Pipeline)

### **1. Project Setup**
- Created project template using `template.py`  
- Added editable install via `-e .`  
- Configured local packages in `setup.py`  

### **2. Data Ingestion**
- Fetch data from MongoDB / DuckDB queries  
- Convert key-value documents into DataFrame  
- Store ingestion artifacts  

### **3. Data Validation**
- Schema defined in `schema.yaml`  
- Check missing values, columns, datatypes  

### **4. Data Transformation**
- Handle missing values  
- Outlier removal (IQR / winsorization)  
- Log / Yeo-Johnson transforms  
- Scaling (Standard / Robust / MinMax)  

### **5. Model Training**
- 240+ stratified combinations  
- Multiple ML models trained  
- MLflow logs metrics, parameters, artifacts  

### **6. Model Evaluation & Registry**
- Compare with evaluation threshold  
- Push best model to AWS S3 Model Registry  

### **7. Deployment**
- Build Docker image  
- Push to AWS ECR  
- Deploy via Kubernetes (EKS)  
- Expose FastAPI endpoints  

### **8. Monitoring**
- Prometheus scrapes FastAPI metrics  
- Grafana dashboards visualize health & predictions  

---

## 🚀 Deployment Architecture

**Tech used:**
- Docker  
- AWS ECR (image storage)  
- GitHub Actions CI/CD  
- AWS EC2 (self-hosted runner optional)  
- AWS EKS (Kubernetes cluster)  
- LoadBalancer service for FastAPI  

---

## 🧪 API Endpoints (FastAPI)

### **POST /predict**
Input patient values → returns risk score + prediction.

### **POST /train**
Triggers full training pipeline (optional).

---

## 💻 Running Locally

```bash
# 1. Create virtual environment
python -m venv heart-risk

# 2. Activate environment
heart-risk\Scripts\activate  # Windows
# source heart-risk/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run API
python app.py
