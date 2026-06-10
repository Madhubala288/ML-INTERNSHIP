# 📊 Student Performance Analysis using Machine Learning

## 📌 Project Overview
This project demonstrates an end-to-end Machine Learning workflow on a student performance dataset. The goal is to analyze student data, understand patterns, and build predictive models to classify student grades based on academic and behavioral factors.
## 🎯 Objectives
* Perform data preprocessing and cleaning
* Conduct exploratory data analysis (EDA)
* Visualize patterns using graphs
* Train machine learning models
* Evaluate model performance
* Document complete workflow
## 📂 Dataset Information
The dataset includes the following features:
* weekly_self_study_hours
* attendance_percentage
* class_participation
* total_score
* grade (Target variable)
## ⚙️ Technologies Used
* Python 🐍
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn
* Jupyter Notebook
## 🧹 Data Preprocessing
* Removed unnecessary columns (e.g., student_id)
* Handled missing values
* Encoded categorical variables
* Applied feature scaling using StandardScaler
## 📊 Exploratory Data Analysis (EDA)
* Summary statistics using `describe()`
* Histograms for distribution analysis
* Boxplots for outlier detection
* Scatter plots to identify relationships
* Correlation heatmap to understand feature relationships
## 🤖 Machine Learning Models
Two classification models were trained:
* Logistic Regression
* Decision Tree Classifier
## 📏 Model Evaluation
Models were evaluated using:
* Accuracy Score
* Precision
* Recall
* F1-Score
📌 Result: Decision Tree / Logistic Regression performance compared to select the best model.
## 📈 Key Insights
* Higher study hours lead to better performance
* Attendance strongly impacts student grades
* Active participation improves overall score
* Data shows positive correlation among key features
## 📁 Project Structur
📦 Student-Performance-ML
 ┣ 📜 dataset.csv
 ┣ 📜 notebook.ipynb
 ┣ 📜 README.md

## 🚀 How to Run
1. Clone the repository
2. Install required libraries
3. Open Jupyter Notebook
4. Run all cells step by step
## 📌 Conclusion
This project successfully demonstrates a complete ML pipeline from data preprocessing to model evaluation, providing insights into factors affecting student performance.

