🏥 Adaptive Diabetes Diagnostic Decision Support System (DDSS)

An AI-powered Clinical Decision Support System built using Streamlit, Machine Learning, and Rule-Based Clinical Logic to predict diabetes risk and provide intelligent, explainable clinical recommendations.

📌 Project Overview

The Adaptive Diabetes DDSS is designed to assist healthcare professionals by:

Predicting the probability of diabetes

Routing patients to appropriate clinical pathways

Providing Explainable AI (XAI) insights

Supporting batch patient analysis

Ensuring reliability with a rule-based fallback system

The system combines machine learning intelligence with medical domain knowledge to enhance clinical decision-making.

🚀 Key Features
🤖 Dual Prediction Engine

Machine Learning Model (Random Forest)

Rule-Based System as fallback if ML model is unavailable

🎯 Adaptive Clinical Routing

Automatically recommends:

Endocrinologist consultation

Lifestyle intervention

Preventive care

Emergency priority pathways

🔍 Explainable AI (XAI)

Feature impact analysis (SHAP-inspired)

Visual explanation of key risk factors

Transparent decision logic

📊 Data Visualization & Analytics

Risk probability visualization

Feature importance plots

Glucose & BMI distribution analysis

📈 Batch Processing

Upload CSV files for multi-patient analysis

Automated risk stratification

Downloadable results

💾 Data Export

Save individual patient records

Export complete datasets in CSV format

🧠 Technologies Used

Frontend: Streamlit

Backend: Python

Machine Learning: Scikit-learn (Random Forest)

Data Processing: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Model Persistence: Joblib

🩺 Clinical Parameters Used

Parameter	Description

Pregnancies	Number of pregnancies

Glucose	Plasma glucose concentration

Blood Pressure	Systolic blood pressure

Skin Thickness	Triceps skinfold thickness

Insulin	Serum insulin

BMI	Body Mass Index

Diabetes Pedigree Function	Family history score

Age	Patient age

🧪 Risk Prediction Logic
Machine Learning Mode

Input data is standardized

Random Forest model predicts probability of diabetes

Rule-Based Fallback Mode

Risk score is computed using:

Glucose (50%)

BMI (20%)

Age (15%)

Blood Pressure (15%)

This ensures system reliability even if ML model files are missing.

🏥 Clinical Decision Pathways

Based on risk probability and patient factors:

🚨 Senior Endocrinologist (High-risk elderly)

⚠️ Endocrinologist Consultation

📋 GP + Additional Tests

💪 Lifestyle Intervention

🛡️ Preventive Care

✅ Routine Annual Screening

📁 Project Structure

📦 Diabetes-DDSS

 ┣ 📜 app.py
 
 ┣ 📜 diabetes_model.pkl
 
 ┣ 📜 scaler.pkl
 
 ┣ 📜 requirements.txt
 
 ┣ 📜 README.md

📊 Batch Processing CSV Format

Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigree,Age
2,180,85,25,95,28.5,0.62,45
1,95,70,20,80,22.0,0.25,35
3,150,90,30,120,32.0,1.20,55
0,110,75,22,65,24.0,0.35,28

▶️ How to Run the Project

1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the Application
streamlit run app.py

3️⃣ Open Browser
http://localhost:8501

⚠️ Important Notes

This system is for clinical decision support only

Predictions should always be validated by medical professionals

High-risk results must be confirmed with laboratory tests

🎓 Academic & Resume Value

This project demonstrates:

Machine Learning application in healthcare

Explainable AI concepts

Clinical rule-based systems

Real-world Streamlit dashboard design

Fault-tolerant system architecture

👤 Author

Pawandeep Singh

Computer Science & Engineering

Final Year Project – AI in Healthcare
