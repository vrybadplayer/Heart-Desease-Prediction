# 💓 Heart Disease Prediction System

A machine learning pipeline and interactive web application for predicting cardiovascular disease risk based on clinical patient biomarkers. This project integrates statistical data analysis, automated feature preprocessing, multi-model benchmarking with a custom clinical evaluation metric, and a lightweight Flask web interface for real-time risk assessment.

---

## 📌 Table of Contents
1. [Overview & Project Objective](#-overview--project-objective)
2. [Dataset & Exploratory Data Analysis (EDA) Insights](#-dataset--exploratory-data-analysis-eda-insights)
3. [Machine Learning Pipeline & Model Comparison](#-machine-learning-pipeline--model-comparison)
4. [Composite Evaluation Metric (CEM) Selection](#-composite-evaluation-metric-cem-selection)
5. [Local Deployment & Installation Guide](#-local-deployment--installation-guide)
6. [Sandbox Verification & Execution Outputs](#-sandbox-verification--execution-outputs)
7. [Repository File Structure](#-repository-file-structure)
8. [Key Clinical & Engineering Takeaways](#-key-clinical--engineering-takeaways)

---

## 🎯 Overview & Project Objective

Cardiovascular diseases (CVDs) remain the leading cause of mortality worldwide. Early identification of individuals at elevated risk allows for timely lifestyle modifications, diagnostic angiography, and preventative pharmacological interventions.

This project delivers:
- **Comprehensive Data Hygiene & EDA:** Rigorous audit of patient records, duplicate detection, and correlation analysis.
- **Robust Feature Preprocessing:** Dual-stream column transformation (standardization for continuous variables, one-hot encoding for categorical variables) with SMOTE oversampling for class balance.
- **Clinically Grounded Benchmarking:** Training and comparing three distinct architectures (Random Forest, Support Vector Classifier, Artificial Neural Network) evaluated on Recall, F2 Score, and ROC AUC to minimize dangerous false negatives.
- **Production Web Deployment:** A Flask application allowing healthcare practitioners or individuals to input diagnostic parameters and obtain instant risk classification with calibrated prediction probabilities.

---

## 📊 Dataset & Exploratory Data Analysis (EDA) Insights

The model is trained on `Heart_Disease_Prediction.csv`, a merged dataset derived from five recognized clinical cardiovascular cohorts (Cleveland, Hungarian, Switzerland, Long Beach V, and Statlog).

### 1. Data Cleaning & Distribution
- **Raw Volume:** 1,190 records across 12 clinical attributes.
- **Missing Values:** `0` missing or null entries detected across all features.
- **Duplicate Removal:** Exactly **272 duplicate records** were detected and removed, yielding **918 clean, unique patient observations**.
- **Class Balance (Target):**
  - **No Heart Disease (`target = 0`):** 410 patients (44.66%)
  - **Heart Disease Present (`target = 1`):** 508 patients (55.34%)

### 2. Feature Correlation & Clinical Insights
Feature correlations with the target variable (`target`) revealed critical diagnostic patterns:

| Feature | Type | Correlation with Target | Clinical Takeaway & Significance |
| :--- | :--- | :---: | :--- |
| **`ST_slope`** | Categorical | **+0.5535** | **Strongest overall predictor.** Flat or downsloping ST segments during peak exercise strongly correlate with myocardial ischemia. |
| **`exercise_angina`** | Binary | **+0.4943** | Patients experiencing exercise-induced angina are substantially more prone to underlying coronary blockages. |
| **`chest_pain_type`** | Categorical | **+0.4714** | Asymptomatic chest pain (Type 4) paradoxically correlates highest with severe disease (indicative of silent ischemia), surpassing atypical or non-anginal pain. |
| **`oldpeak`** | Continuous | **+0.4040** | ST depression induced by exercise relative to rest. Greater depression corresponds directly to severity of coronary artery disease. |
| **`max_heart_rate`** | Continuous | **-0.4004** | **Strongest negative correlation.** Inability to achieve normal age-predicted maximum heart rate (chronotropic incompetence) indicates cardiac impairment. |
| **`sex`** | Binary | **+0.3054** | Male patients (coded as 1) present higher disease frequency in this demographic cohort. |
| **`age`** | Continuous | **+0.2820** | Risk monotonically ascends with advancing age due to cumulative arterial plaque burden. |
| **`fasting_blood_sugar`** | Binary | **+0.2673** | Fasting blood sugar > 120 mg/dL serves as an independent risk modifier linked to metabolic dysfunction. |
| **`cholesterol`** | Continuous | **-0.2327** | Exhibits an inverse artifact in this combined dataset because several inpatient cohort records had recorded values of `0 mg/dL` prior to standardization. |
| **`resting_bp_s`** | Continuous | **+0.1076** | Modest positive correlation; elevated systolic blood pressure contributes to long-term vascular resistance. |
| **`resting_ecg`** | Categorical | **+0.0610** | Lowest correlation in this cohort; safely omitted from final feature selection to minimize noise. |

### 3. Feature Selection & Pipeline Setup
The 10 selected features are processed through a Scikit-Learn `ColumnTransformer`:
- **Continuous (StandardScaler):** `resting_bp_s`, `max_heart_rate`, `cholesterol`, `oldpeak`, `age`
- **Categorical (OneHotEncoder):** `sex`, `chest_pain_type`, `fasting_blood_sugar`, `exercise_angina`, `ST_slope`

---

## 🤖 Machine Learning Pipeline & Model Comparison

Data was split into an **80% training set** and a **20% holdout test set** (184 test records: 82 non-disease, 102 disease) using **stratified sampling** to preserve the class ratio. Synthetic Minority Over-sampling Technique (**SMOTE**) was applied to the training split.

Three classification architectures were trained and evaluated:
1. **Random Forest Classifier** (`RandomForestClassifier(random_state=42)`)
2. **Support Vector Classifier** (`SVC(probability=True, random_state=42)`)
3. **Artificial Neural Network** (`MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)`)

### Evaluation Results on Test Set (184 Patients)

| Model Architecture | Accuracy | Precision | Recall | F1 Score | F2 Score | ROC AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 88.59% | 88.57% | 91.18% | 89.86% | 90.64% | 93.39% |
| **Support Vector Machine (SVM)** | **90.22%** | **89.62%** | **93.14%** | **91.35%** | **92.41%** | **94.49%** |
| **Artificial Neural Network (ANN)**| 86.41% | 88.89% | 86.27% | 87.56% | 86.79% | 90.83% |

---

## 🏆 Composite Evaluation Metric (CEM) Selection

In medical diagnostics, **False Negatives are catastrophic**—failing to identify an individual with heart disease can lead to preventable cardiac events or fatality. In contrast, False Positives lead to non-invasive confirmatory testing.

To reflect clinical priorities, model selection utilized a **Composite Evaluation Metric (CEM)** that heavily weights sensitivity and discrimination:

$$\text{CEM} = (0.50 \times \text{Recall}) + (0.30 \times \text{F2 Score}) + (0.20 \times \text{ROC AUC})$$

### Computed CEM Scores:
- **Support Vector Machine (SVM):** $(0.50 \times 0.9314) + (0.30 \times 0.9241) + (0.20 \times 0.9449) =$ **`0.9319`** 🥇 *(Winner)*
- **Random Forest:** $(0.50 \times 0.9118) + (0.30 \times 0.9064) + (0.20 \times 0.9339) =$ **`0.9146`**
- **Artificial Neural Network:** $(0.50 \times 0.8627) + (0.30 \times 0.8679) + (0.20 \times 0.9083) =$ **`0.8734`**

Because the **Support Vector Classifier (SVC)** dominated every single metric (highest Recall of 93.14%, highest Accuracy of 90.22%, and highest ROC AUC of 94.49%), it was exported as `heart_model.pkl` along with `preprocessor.pkl`.

---

## 💻 Local Deployment & Installation Guide

Follow these steps to deploy and execute the application in your local environment.

### 1. Prerequisites
- **Python:** Version 3.10 or 3.11 recommended.
- **Git** installed on your system.

> ⚠️ **CRITICAL DEPENDENCY NOTE:**  
> The exported `heart_model.pkl` and `preprocessor.pkl` files were serialized using **`scikit-learn==1.5.1`**. In `scikit-learn >= 1.6`, internal changes to `ColumnTransformer` (`_RemainderColsList`) will raise an unpickling `AttributeError`. **You must install `scikit-learn==1.5.1`** to run the pre-trained model directly without retraining.

### 2. Clone the Repository
```bash
git clone https://github.com/vrybadplayer/Heart-Desease-Prediction.git
cd Heart-Desease-Prediction
```

### 3. Create and Activate a Virtual Environment
- **On Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **On Windows (Command Prompt / PowerShell):**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```

### 4. Install Required Packages
Install the verified working versions of the dependencies:
```bash
pip install flask==3.1.0 pandas==2.2.2 "scikit-learn==1.5.1" joblib==1.4.2
```
*(Alternatively, create a `requirements.txt` containing the lines above and run `pip install -r requirements.txt`.)*

### 5. Launch the Web Application
Execute the Flask server from within the root repository directory (where `heart_model.pkl` and `preprocessor.pkl` are located):
```bash
python app.py
```

You should see terminal output confirming the server has started:
```text
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🧪 Sandbox Verification & Execution Outputs

The application was executed and validated directly in a local sandbox environment. Both API endpoints and the inference engine were tested against boundary clinical scenarios.

### 1. High-Risk Patient Scenario
- **Input Parameters:**
  - Age: `61`
  - Sex: `Male` (`1`)
  - Chest Pain Type: `4` (Asymptomatic)
  - Resting Blood Pressure: `140 mm Hg`
  - Serum Cholesterol: `207 mg/dL`
  - Fasting Blood Sugar > 120: `No` (`0`)
  - Maximum Heart Rate: `138 bpm`
  - Exercise-Induced Angina: `Yes` (`1`)
  - Oldpeak (ST Depression): `1.9`
  - ST Slope: `1` (Downsloping)
- **Model Output:**
  ```text
  Prediction : High Chance of Heart Disease ❌
  Probability: 82.97%
  HTTP Status: 200 OK
  ```

### 2. Low-Risk Patient Scenario
- **Input Parameters:**
  - Age: `40`
  - Sex: `Female` (`0`)
  - Chest Pain Type: `2` (Atypical Angina)
  - Resting Blood Pressure: `120 mm Hg`
  - Serum Cholesterol: `180 mg/dL`
  - Fasting Blood Sugar > 120: `No` (`0`)
  - Maximum Heart Rate: `172 bpm`
  - Exercise-Induced Angina: `No` (`0`)
  - Oldpeak (ST Depression): `0.0`
  - ST Slope: `3` (Upsloping)
- **Model Output:**
  ```text
  Prediction : Low Chance of Heart Disease ✅
  Probability: 12.85%
  HTTP Status: 200 OK
  ```

### 3. Web UI Interactive Features
- **🔁 Random Input Generator:** The `fillRandom()` client script dynamically populates physiological ranges within realistic human thresholds for rapid stress testing.
- **💾 Session Persistence:** JavaScript `localStorage` automatically saves form inputs on submission and reloads them on page refresh, preserving user data during iterative assessments.

---

## 📂 Repository File Structure

```text
Heart-Desease-Prediction/
├── app.py                         # Flask web server and prediction routing logic
├── Heart_Disease_Prediction.csv   # Multicenter cardiovascular patient dataset
├── HeartDiseasePrediction.ipynb   # Jupyter Notebook: EDA, SMOTE, training & CEM benchmark
├── heart_model.pkl                # Serialized best model (Support Vector Classifier, SVC)
├── preprocessor.pkl               # Serialized ColumnTransformer (StandardScaler + OneHotEncoder)
├── templates/
│   └── index.html                 # HTML5/CSS3 frontend template with form validation
└── HeartDiseasePrediction_CEM/    # Auxiliary project directory with model checkpoints
```

---

## 💡 Key Clinical & Engineering Takeaways

1. **Electrocardiographic Stress Signals Dominate Diagnostic Weight:**
   `ST_slope` (+0.5535) and exercise-induced `oldpeak` depression (+0.4040) proved far more predictive than resting metrics like blood pressure. Stress testing provides the clearest physiological signal of coronary insufficiency.
2. **Watch for the "Silent Ischemia" Paradox:**
   Patients with Type 4 (asymptomatic) chest pain had higher instances of positive diagnoses than those reporting typical discomfort, highlighting why screening shouldn't solely rely on subjective patient complaints.
3. **Metric Alignment with Clinical Reality:**
   Standard accuracy (90.22%) is insufficient on its own in healthcare. The custom CEM formulation ensured that the chosen SVM architecture achieved **93.14% Recall**, capturing the maximum number of true cardiac cases while keeping overall accuracy high.
4. **Serialization Environment Consistency:**
   Always pin scikit-learn minor versions when distributing pickled machine learning pipelines across teams or cloud deployments to avoid serialization format mismatches.
