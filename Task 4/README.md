# Credit Card Fraud Detection using Advanced Machine Learning

## Project Description

Credit card fraud is one of the major challenges faced by financial institutions. Fraudulent transactions cause significant financial losses and reduce customer trust. This project aims to develop an advanced machine learning system capable of accurately detecting fraudulent credit card transactions.

The project demonstrates the complete machine learning workflow, including data preprocessing, feature engineering, handling imbalanced datasets, model training, hyperparameter tuning, cross-validation, and deployment through a Streamlit web application.

---

## Problem Statement

The objective of this project is to build a machine learning model that can identify fraudulent credit card transactions with high accuracy while minimizing false positives and false negatives.

Since fraudulent transactions represent only a very small percentage of all transactions, the dataset is highly imbalanced. Special techniques are required to ensure effective fraud detection.

---

## Dataset Information

Dataset: Credit Card Fraud Detection Dataset

### Features

* Time
* Amount
* V1 to V28 (PCA transformed features)
* Class (Target Variable)

### Target Variable

| Class | Description            |
| ----- | ---------------------- |
| 0     | Legitimate Transaction |
| 1     | Fraudulent Transaction |

---

## Project Objectives

* Explore and analyze transaction data.
* Perform advanced feature engineering.
* Handle imbalanced data using SMOTE.
* Train and compare multiple machine learning models.
* Optimize model performance using hyperparameter tuning.
* Validate model stability using cross-validation.
* Build an interactive Streamlit application.
* Generate real-time fraud predictions.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Streamlit
* Joblib
* GitHub

---

## Exploratory Data Analysis (EDA)

The following analyses were performed:

* Dataset shape and structure analysis
* Statistical summary of features
* Missing value analysis
* Duplicate record detection
* Class distribution visualization
* Correlation analysis
* Outlier analysis
* Feature importance evaluation

### Key Findings

* The dataset contains highly imbalanced classes.
* Fraudulent transactions represent a very small percentage of total transactions.
* Most features are already transformed and normalized.
* The Amount feature required scaling.
* Duplicate transactions were identified and removed.

---

## Feature Engineering

The following preprocessing techniques were applied:

### Data Cleaning

* Removed duplicate records.
* Verified missing values.

### Feature Scaling

* StandardScaler applied on:

  * Time
  * Amount

### Feature Selection

* Correlation analysis
* Identification of important predictors

### Imbalanced Data Handling

SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the minority class and improve model performance.

---

## Machine Learning Models

The following classification algorithms were trained and evaluated:

| Model                        |
| ---------------------------- |
| Logistic Regression          |
| Decision Tree Classifier     |
| Random Forest Classifier     |
| K-Nearest Neighbors (KNN)    |
| Support Vector Machine (SVM) |

---

## Model Evaluation Metrics

Models were evaluated using:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve
* AUC Score

These metrics provide a comprehensive assessment of fraud detection performance.

---

## Hyperparameter Tuning

The best-performing model was optimized using:

### GridSearchCV

Parameters tuned:

* n_estimators
* max_depth
* min_samples_split

### RandomizedSearchCV

Random parameter combinations were tested to improve model efficiency and performance.

---

## Cross Validation

K-Fold Cross Validation was applied to:

* Evaluate model stability
* Reduce overfitting risk
* Ensure consistent performance across different data splits

---

## Best Model Selection

After comparing all models based on evaluation metrics, the best-performing model was selected and saved for deployment.

Selection criteria included:

* High Recall
* High F1-Score
* Strong ROC-AUC Performance
* Cross-validation consistency

---

## Streamlit Web Application

The project includes an interactive Streamlit application with the following features:

### Dataset Overview

* Dataset preview
* Summary statistics

### Model Comparison

* Performance comparison of all trained models

### Prediction System

* User input interface
* Real-time fraud prediction

### Visualizations

* Class distribution
* Confusion matrix
* ROC curve
* Model comparison charts

---

## Project Structure

Week4-Advanced-ML-Optimization/

├── data/

│ └── creditcard.csv

├── models/

│ ├── best_model.pkl

│ └── scaler.pkl

├── notebooks/

│ └── credit_card_fraud_detection.ipynb

├── feature_engineering/

│ └── preprocessing.py

├── screenshots/

├── app.py

├── requirements.txt

├── README.md

└── model_comparison.ipynb

---

## Installation

### Clone Repository

git clone <repository-link>

cd Week4-Advanced-ML-Optimization

### Install Dependencies

pip install -r requirements.txt

---

## Running the Project

### Run Jupyter Notebook

Open:

credit_card_fraud_detection.ipynb

### Run Streamlit Application

streamlit run app.py

---

## Project Outcomes

* Successfully handled imbalanced transaction data.
* Applied advanced feature engineering techniques.
* Compared multiple machine learning algorithms.
* Improved performance through model optimization.
* Evaluated models using advanced classification metrics.
* Built a deployable fraud detection system.
* Developed a production-ready Streamlit application.

---

## Future Enhancements

* Ensemble stacking techniques
* Automated model selection
* Cloud deployment using Streamlit Cloud or Render
* End-to-end ML pipeline implementation
* Real-time transaction monitoring system

---

## Screenshots

The repository includes screenshots of:

* Dataset Overview
* EDA Results
* Class Distribution
* SMOTE Output
* Model Training Results
* Confusion Matrix
* ROC Curve
* Hyperparameter Tuning Results
* Cross Validation Results
* Streamlit Application

---

## Author

Advanced Machine Learning Optimization Project

Credit Card Fraud Detection System

Machine Learning Internship – Week 4
