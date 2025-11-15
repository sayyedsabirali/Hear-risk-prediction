# 🫀 HEART-FAILURE RISK PREDICTION IN ICU PATIENTS

### **A machine learning system to predict heart-attack risk in ICU patients using clinical data to provide early warnings for doctors.**

---

## 📌 Overview

Early detection of heart attacks in ICU patients is challenging because clinical data is massive, complex, and continuously changing.  
This project builds an automated **early-warning ML system** that predicts the risk of a heart attack using patient vitals, lab values, and demographics.

The solution integrates a **complete end-to-end MLOps pipeline**, including:
- Data ingestion  
- Data validation & transformation  
- ML model training  
- Local MLflow experiment tracking  
- DVC for data versioning  
- Docker containerization  
- CI/CD with GitHub Actions  
- AWS (S3, ECR, EC2, EKS) for deployment  

---

## 🎬 Project Demo Video

👉 **Watch the full project demonstration:**  
https://drive.google.com/file/d/1vfAtL3qrUr5U6ToNtJ9Jf6OEGZf4CABv/view?usp=drive_link

---

## 📚 Dataset Used

The **MIMIC-IV v3.1** dataset from PhysioNet is a **very large clinical database (~500GB+)** containing ICU patient data such as demographics, labs, vitals, medications, and diagnoses.

Because the full dataset is extremely large, we extracted only the tables relevant to predicting heart-failure risk.

### 📦 Extracted Tables (60GB from 500GB+)
We selected **5 key tables** important for clinical prediction:

- `patients`  
- `admissions`  
- `diagnoses_icd`  
- `labevents`  
- `chartevents`

These combined were approximately **60GB** in size.

### 🧩 Building the Final Dataset

From these 60GB tables:

1. **Filtered patients diagnosed with heart failure / heart attack**  
   using ICD-9 (`410%`) and ICD-10 (`I21%`) codes.

2. **Selected an equal number of non-heart-failure patients**  
   to create a **balanced dataset** with a 1:1 ratio.  
   - Prevents bias  
   - Improves model training stability  
   - Gives more reliable evaluation

3. **Selected only clinically meaningful features**, including:  
   - Lab values  
   - Vital signs  
   - Demographics  
   - Admission details

### ✅ Final Curated Dataset
- **33,018 ICU patients**  
- 16,509 heart-failure cases  
- 16,509 matched non-heart-failure controls  
- **23 clinical features** (labs, vitals, demographics)

This dataset was cleaned, balanced, and optimized for machine learning.

---

## 🔬 Core Features Used

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

All major ML models were trained and evaluated:
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

Both models performed extremely well and were used for final evaluation.

---

## ⚙️ Tech Stack

### **Languages & Libraries**
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost, LightGBM, CatBoost  
- DuckDB  
- Matplotlib / Seaborn  

### **MLOps & Infrastructure**
- MLflow (local tracking)  
- DVC (data versioning)  
- Docker  
- AWS S3 (artifact storage)  
- AWS ECR (container registry)  
- AWS EC2  
- Kubernetes (EKS)  
- GitHub Actions (CI/CD)  

---

## 🔄 Project Workflow (MLOps Pipeline)

### **1. Project Setup**
- Created project structure using `template.py`  
- Added editable install (`-e .`)  
- Added local package configuration in `setup.py`  

### **2. Data Ingestion**
- Pulled 5 major MIMIC-IV tables  
- Performed filtering using ICD codes  
- Created balanced heart-failure vs non-heart-failure dataset  

### **3. Data Validation**
- Schema in `schema.yaml`  
- Check datatypes, missing values, required columns  

### **4. Data Transformation**
- Missing value handling  
- Outlier treatment  
- Skewness correction  
- Standardization / normalization  

### **5. Model Training**
- Multiple ML models tested  
- Hyperparameter tuning  
- Tracked experiments using **local MLflow**  

### **6. Model Evaluation & Registry**
- Compared metrics using a threshold score  
- Best model pushed to AWS S3 model registry  

### **7. Deployment**
- Built Docker image  
- Pushed to AWS ECR  
- Deployed to AWS EKS cluster with:
  - `deployment.yaml`
  - `service.yaml`

### **8. FastAPI Endpoints**
- `/predict` — returns heart-failure risk score  

---

## 💻 Running Locally

```bash
# Create virtual environment
python -m venv heart-risk

# Activate (Windows)
heart-risk\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run API
python app.py
