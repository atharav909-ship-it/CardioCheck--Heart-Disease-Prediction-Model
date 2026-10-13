# ❤️ CardioCheck — Heart Disease Prediction System

**Machine Learning | FastAPI | Scikit-learn | Healthcare Analytics**

CardioCheck is a machine learning-powered web application designed to assess patterns associated with heart disease using cardiovascular health indicators. It combines a Logistic Regression classification model with a modern, interactive dashboard to provide accessible, educational health assessments.

The application demonstrates an end-to-end machine learning workflow, from exploratory data analysis and model training to backend integration, interactive visualization, and cloud deployment.

> **Disclaimer:** CardioCheck is an educational project and is not intended for clinical diagnosis, treatment decisions, or professional medical advice.

---

## Project Overview

Heart disease is a major global health concern. Cardiovascular indicators such as cholesterol levels, blood pressure, exercise response, and blood glucose measurements can provide useful information for understanding heart health.

CardioCheck explores how supervised machine learning can identify patterns associated with the presence of heart disease.

The system accepts health measurements, processes them through a trained classification pipeline, and presents the model's estimated probability and classification through an interactive interface.

### Project Objectives

- Develop a binary classification model for heart disease prediction.
- Analyze cardiovascular health indicators through exploratory data analysis.
- Evaluate model performance using multiple classification metrics.
- Integrate the trained model with a FastAPI backend.
- Build a responsive frontend for entering measurements and viewing results.
- Deploy the application as a publicly accessible web service.

## Key Features

### Interactive Health Assessment

A structured assessment form accepts cardiovascular and lifestyle measurements, including cholesterol levels, blood pressure, physical activity, and exercise-related indicators.

### Machine Learning Predictions

A trained Logistic Regression model processes the submitted measurements and returns:

- Heart Disease Indicated / Heart Disease Not Indicated
- Estimated likelihood of the positive classification
- A visual probability indicator

### Health Analytics Dashboard

A multi-panel interface organizes the application into:

| Panel | Purpose |
|---|---|
| Dashboard | Enter health measurements and initiate assessments |
| Vitals | Review submitted measurements and visualizations |
| Predictions | View classification results and estimated likelihood |
| Settings | Review model information, performance, and limitations |

### Backend Integration

FastAPI connects the frontend to the trained machine learning model through a prediction API, allowing assessments to be processed without retraining the model for every request.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Classification Algorithm | Logistic Regression |
| Preprocessing | RobustScaler |
| Model Serialization | Joblib |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Model Development | Kaggle Notebooks |
| Version Control | Git, GitHub |
| Deployment | Render |

## Dataset and Input Features

The model was trained using the **Heart Disease Risk 2026** dataset available on Kaggle, containing approximately 9,000 records.

**Target variable:** `has_heart_disease`

- `0` — Heart disease absent
- `1` — Heart disease present

The final model uses the following ten input features:

| Feature | Description |
|---|---|
| `max_heart_rate_achieved` | Maximum heart rate recorded during exercise |
| `exercise_induced_angina` | Presence of exercise-induced chest discomfort |
| `st_depression` | ST-segment depression associated with exercise |
| `age` | Patient age |
| `ldl` | Low-density lipoprotein cholesterol |
| `hdl` | High-density lipoprotein cholesterol |
| `resting_bp_systolic` | Resting systolic blood pressure |
| `exercise_minutes_per_week` | Weekly physical activity duration |
| `hba1c` | Glycated hemoglobin measurement |
| `bmi` | Body mass index |

The application must preserve the feature names, order, encodings, and measurement units expected by the trained model.

## Machine Learning Pipeline

The development process involved:

1. **Data Exploration:** Examining dataset structure, feature distributions, class balance, and relationships between variables.
2. **Feature Selection:** Selecting ten cardiovascular indicators informed by exploratory analysis.
3. **Train-Test Split:** Using an 80:20 stratified split to preserve the target-class distribution.
4. **Preprocessing:** Applying RobustScaler to reduce sensitivity to extreme feature values.
5. **Model Training:** Training Logistic Regression with balanced class weights.
6. **Evaluation:** Measuring classification accuracy, precision, recall, F1-score, and ROC-AUC.
7. **Model Export:** Saving the complete preprocessing and classification pipeline using Joblib.
8. **Deployment:** Loading the serialized model within a FastAPI application.

### Model Configuration

```python
model = Pipeline([
    ("scaler", RobustScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=60
    ))
])
```

## Model Performance

The following results were obtained from the held-out test split during model development.

| Metric | Result |
|---|---:|
| Accuracy | 88.78% |
| Precision — Heart Disease | 77% |
| Recall — Heart Disease | 89.36% |
| F1-Score — Heart Disease | 82.82% |
| ROC-AUC | 94.79% |

### Confusion Matrix

| Actual / Predicted | No Heart Disease | Heart Disease |
|---|---:|---:|
| No Heart Disease | 1,111 | 144 |
| Heart Disease | 58 | 487 |

The model correctly identified 487 of 545 positive cases in the test split.

**Evaluation limitations:** These results reflect performance on the development dataset, not clinical validation. Feature selection was informed by analysis of the dataset before splitting, which may lead to optimistic estimates. Independent external validation and probability calibration would be required before considering clinical use.

## Application Architecture

```text
User
  |
  v
Interactive Web Dashboard
  |
  | Health measurements
  v
JavaScript Frontend
  |
  | POST /api/predict
  v
FastAPI Backend
  |
  | Input validation and feature preparation
  v
Scikit-learn Pipeline
  |
  +-- RobustScaler
  |
  +-- Logistic Regression
  |
  v
Prediction + Estimated Probability
  |
  v
Frontend Result Visualization
```

### Prediction Workflow

1. The user enters health measurements in the assessment form.
2. JavaScript collects the submitted values.
3. The frontend sends the values to the FastAPI prediction endpoint.
4. The backend prepares the model's expected input features.
5. The saved Scikit-learn pipeline applies preprocessing and classification.
6. The backend returns the predicted class and estimated probability.
7. The frontend displays the assessment outcome and visualizations.

## Project Structure

```text
CardioCheck/
├── app.py
├── best_heart_disease_model.joblib
├── requirements.txt
├── render.yaml
├── README.md
├── .gitignore
└── static/
    ├── index.html
    ├── style.css
    └── script.js

```

*The structure above assumes the standard project layout used for deployment.*

## Installation and Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/atharav909-ship-it/CardioCheck--Heart-Disease-Prediction-Model
cd CardioCheck-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the trained model

Ensure `best_heart_disease_model.joblib` is present in the project root and was created using a compatible version of Scikit-learn.

### 5. Run the application

```bash
uvicorn app:app --reload
```

Open:

`http://127.0.0.1:8000`

## Deployment

CardioCheck is configured for deployment on Render using GitHub.

**Build Command**

```bash
pip install -r requirements.txt
```

**Start Command**

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

The application can be deployed by connecting the GitHub repository to a Render Web Service.

**Live Application:** Add the Render URL after deployment.

## Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- The model does not consider every possible cardiovascular risk factor.
- Model probabilities have not been independently calibrated.
- Performance on new populations may differ from test-set results.
- The application is not a substitute for professional cardiovascular assessment.
- The current system is intended for educational demonstration and machine learning experimentation.

## Future Improvements

- Compare additional classification algorithms and tuned models.
- Evaluate performance on independent external datasets.
- Add model explainability using SHAP or similar techniques.
- Introduce probability calibration and additional validation.
- Improve input validation and accessibility.
- Add automated backend and frontend testing.
- Develop privacy-preserving assessment history functionality.

## Learning Outcomes

This project demonstrates practical experience in:

- Exploratory data analysis and feature selection
- Supervised machine learning and binary classification
- Model evaluation and interpretation
- Scikit-learn pipelines and model serialization
- FastAPI backend development
- Frontend-to-backend API integration
- Interactive dashboard development
- Git-based version control
- Cloud deployment

## License and Usage

This repository is intended for educational and portfolio purposes. Dataset usage remains subject to the original dataset provider's terms.

---

**CardioCheck — Applying Machine Learning to Cardiovascular Health Analytics**

*Educational demonstration only. Not a medical device or diagnostic tool.*