# MLproject
## Live Demo
Deployed on Render: <https://mlproject-nwbw.onrender.com/>

# 🎓 Student Performance Prediction — End-to-End ML + Flask

An **end-to-end Machine Learning project** that predicts a student’s **Math Score** using demographic + academic features.  
It includes a complete **training pipeline**, a **prediction pipeline**, an **interactive Flask web app**, and a **REST API**.

✅ EDA + Model Training (Notebooks)  
✅ Modular ML Pipeline (Ingestion → Transformation → Training)  
✅ Prediction Pipeline  
✅ Flask Web UI + JSON API  
✅ Deployed on Render  

---

## 🌐 Live Demo

➡️ https://mlproject-nwbw.onrender.com/

---

## 🧠 Problem Statement

Predict **Math Score** based on:

- gender  
- race/ethnicity  
- parental education  
- lunch type  
- test preparation course  
- reading score  
- writing score  

---

## 🚀 Features

- **Interactive Web UI** → Enter student details and get prediction instantly  
- **REST API** → `POST /predict` returns JSON prediction  
- **Reusable pipeline code** inside `src/` (industry-style structure)  
- **Model artifacts** saved for deployment (`model.pkl`, `preprocessor.pkl`)  
- **Logs** stored for debugging and tracking pipeline runs  

---

## 🏗️ Project Structure

MLproject/
│── app.py
│── templates/
│ └── index.html
│── notebooks/
│ ├── 1 . EDA STUDENT PERFORMANCE .ipynb
│ ├── 2. MODEL TRAINING.ipynb
│ └── data/stud.csv
│── src/
│ └── mlproject/
│ ├── components/
│ │ ├── data_ingestion.py
│ │ ├── data_transformation.py
│ │ └── model_trainer.py
│ ├── pipeline/
│ │ ├── train_pipeline.py
│ │ └── predict_pipeline.py
│ ├── utils/
│ │ └── common.py
│ ├── logger.py
│ └── exception.py
│── artifacts/
│── logs/
│── requirements.txt
│── setup.py
│── Procfile
│── .gitignore


---

## ⚙️ How It Works

### ✅ Training Pipeline (`train_pipeline.py`)
The training pipeline performs:

1. **Data Ingestion**
   - Reads dataset from `notebooks/data/stud.csv`
   - Saves raw data into: `artifacts/raw.csv`

2. **Data Transformation**
   - Applies preprocessing:
     - Standard scaling for numerical columns
     - One-hot encoding for categorical columns
   - Saves preprocessor into: `artifacts/preprocessor.pkl`

3. **Model Training**
   - Trains multiple regression models
   - Selects best model using **R2 score**
   - Saves trained model into: `artifacts/model.pkl`

---

### ✅ Prediction Pipeline (`predict_pipeline.py`)
- Loads `preprocessor.pkl` and `model.pkl`
- Converts user input into DataFrame
- Returns predicted math score

---

## ▶️ Run Locally

### 1) Clone Repository
`bash
git clone https://github.com/yuvicann/MLproject.git
cd MLproject
2) Create & Activate Environment
conda create -p .venv python=3.11 -y
conda activate ./.venv

3) Install Dependencies
pip install -r requirements.txt
pip install -e .

4) Train Model (Creates Artifacts)
python src/mlproject/pipeline/train_pipeline.py

5) Run Flask App
python app.py


Open in browser:
➡️ http://127.0.0.1:5000/

🔌 API Usage
✅ Endpoint
POST /predict

✅ Sample Request (JSON)
{
  "gender": "female",
  "race_ethnicity": "group B",
  "parental_level_of_education": "bachelor's degree",
  "lunch": "standard",
  "test_preparation_course": "none",
  "reading_score": 72,
  "writing_score": 74
}

✅ Sample Response
{
  "predicted_math_score": 68.23
}

🧰 Tech Stack

Python 3.11

Pandas, NumPy

Scikit-learn

XGBoost, CatBoost

Flask (UI + API)

Render (Deployment)

👨‍💻 Author

Yuvraj Singh
📩 yurajsingh22@gmail.com
