# 💳 Credit Card Fraud Detection System

An end-to-end Machine Learning project that identifies fraudulent credit card transactions using **Random Forest** and **AdaBoost** classification algorithms. The system includes data preprocessing, class imbalance handling, model evaluation, REST APIs, and an interactive React dashboard.

## 🚀 Project Overview

Credit card fraud is a major challenge in digital banking and financial transactions. This project uses machine learning techniques to identify suspicious transactions and distinguish fraudulent transactions from legitimate ones.

The system trains and compares multiple classification algorithms and provides a prediction interface through a FastAPI backend and React frontend.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Algorithms | Random Forest, AdaBoost |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Imbalanced Data | SMOTE |
| Backend | FastAPI |
| Frontend | React.js, Vite |
| API Communication | Axios |
| Model Persistence | Joblib |
| Development | Jupyter Notebook, VS Code |

## 🏗️ Project Architecture

```text
Credit-Card-Fraud-Detection/
│
├── dataset/
│   └── creditcard.csv
│
├── backend/
│   └── app/
│       ├── config.py
│       ├── data_loader.py
│       ├── preprocessing.py
│       ├── model_training.py
│       ├── evaluation.py
│       ├── prediction.py
│       ├── schemas.py
│       └── main.py
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── style.css
│   ├── package.json
│   └── vite.config.js
│
├── models/
├── reports/
├── notebooks/
├── tests/
├── requirements.txt
└── README.md
```

## 📊 Dataset

**Dataset:** Credit Card Fraud Detection – Kaggle

Dataset Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset characteristics:

- 284,807 transactions
- 492 fraudulent transactions
- 30 input features
- Target variable: `Class`

Target classification:

- `0` – Legitimate Transaction
- `1` – Fraudulent Transaction

**Note:** Download `creditcard.csv` from Kaggle and place it inside the `dataset/` directory.

## ⚙️ Machine Learning Workflow

1. Data Collection – Load transaction data from the Kaggle dataset.
2. Data Preprocessing – Validate data, handle duplicates and prepare features.
3. Feature Scaling – Standardize Time and Amount features.
4. Class Imbalance Handling – Apply SMOTE to training data.
5. Model Training – Train Random Forest and AdaBoost classifiers.
6. Model Evaluation – Compare classification performance.
7. Model Selection – Select the model based on Precision-Recall AUC.
8. Prediction – Classify new transactions using the trained model.

## 🤖 Machine Learning Algorithms

### 1. Random Forest

- Ensemble learning algorithm.
- Uses multiple decision trees.
- Supports class weighting.
- Identifies complex transaction patterns.

### 2. AdaBoost

- Boosting-based ensemble algorithm.
- Combines multiple weak learners.
- Improves classification through sequential learning.

## 📈 Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Precision-Recall AUC
- Confusion Matrix

Precision, Recall and F1-score are particularly important because the dataset contains significantly fewer fraudulent transactions than legitimate transactions.

Actual evaluation metrics are generated after running the training pipeline.

## 🔌 Backend API – FastAPI

The backend provides REST APIs for model health, evaluation metrics and fraud prediction.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health`
