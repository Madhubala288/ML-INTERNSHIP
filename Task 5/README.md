# ♻️ Waste Classification System Using Deep Learning (ANN & CNN)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-red)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CNN-green)

## 📌 Project Overview

Waste management is an important environmental challenge. Manual waste sorting is time-consuming and inefficient. This project presents an **AI-powered Waste Classification System** that automatically identifies waste categories from images using **Deep Learning and Computer Vision techniques**.

The system uses:

- Artificial Neural Network (ANN) for baseline comparison
- Convolutional Neural Network (CNN) for advanced image classification
- TensorFlow & Keras for deep learning implementation
- Streamlit for interactive deployment

Users can upload waste images and the application predicts the waste category along with the confidence score.

---

# 🎯 Project Objectives

The main objectives of this project are:

- Understand Deep Learning workflows
- Perform image preprocessing and augmentation
- Build Artificial Neural Network models
- Build Convolutional Neural Network models
- Train and evaluate image classification models
- Apply optimization techniques
- Deploy a deep learning application using Streamlit

---

# 📂 Dataset Description

## Waste Classification Dataset

The dataset contains images belonging to multiple waste categories.

### Classes:

```
1. Organic
2. Recyclable
3. Hazardous
4. Non-Recyclable
```

Each class contains different waste images used for training and evaluation.

Dataset Structure:

```
dataset/

├── Organic/
├── Recyclable/
├── Hazardous/
└── Non-Recyclable/
```

Dataset Source:

Kaggle Waste Classification Dataset

---

# 🛠️ Technologies Used

## Programming Language

- Python

## Deep Learning Frameworks

- TensorFlow
- Keras

## Data Processing

- NumPy
- Pandas
- OpenCV
- Pillow

## Visualization

- Matplotlib
- Seaborn

## Machine Learning

- Scikit-learn

## Deployment

- Streamlit

## Development Environment

- Jupyter Notebook
- VS Code

---

# 🏗️ Project Architecture


```
Input Image
      |
      |
Image Preprocessing
      |
      |
Data Augmentation
      |
      |
CNN Deep Learning Model
      |
      |
Feature Extraction
      |
      |
Classification Layer
      |
      |
Waste Category Prediction
      |
      |
Confidence Score
```

---

# 🔄 Data Preprocessing

The following preprocessing techniques were applied:

## Image Resizing

All images were resized into:

```
224 x 224 pixels
```

## Normalization

Pixel values were scaled:

```
0 - 255  → 0 - 1
```

## Data Augmentation

To improve model generalization:

Applied techniques:

- Rotation
- Zoom
- Horizontal Flip
- Brightness Adjustment


## Dataset Splitting

Dataset was divided into:

```
Training Data : 80%

Validation Data : 20%
```

---

# 🧠 Deep Learning Models

## Model 1: Artificial Neural Network (ANN)

ANN was developed as a baseline model.

Architecture:

```
Input Layer

↓

Flatten Layer

↓

Dense Layer (512 neurons)

↓

Dropout

↓

Dense Layer (256 neurons)

↓

Dropout

↓

Softmax Output Layer

```

---

# Model 2: Convolutional Neural Network (CNN)

CNN was implemented for image classification.

Architecture:

```
Input Image

↓

Convolution Layer

↓

Batch Normalization

↓

Max Pooling

↓

Convolution Layer

↓

Batch Normalization

↓

Max Pooling

↓

Flatten

↓

Dense Layer

↓

Dropout

↓

Softmax Output Layer

```

---

# ⚙️ Model Optimization Techniques

The following techniques were used to improve performance:

### Dropout Regularization

Reduces overfitting by randomly disabling neurons during training.

### Batch Normalization

Improves training stability and convergence speed.

### Early Stopping

Stops training when validation performance stops improving.

### Learning Rate Adjustment

Automatically reduces learning rate during training.

---

# 📊 Model Evaluation

Models were evaluated using:

## Accuracy Score

Measures overall classification performance.


## Confusion Matrix

Shows correct and incorrect predictions for each class.


## Classification Report

Includes:

- Precision
- Recall
- F1-score


## Training Performance Graphs

Generated:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss


---

# 📈 Results

## Model Comparison

| Model | Accuracy |
|------|----------|
| ANN | XX% |
| CNN | XX% |


CNN achieved better performance because convolution layers can automatically extract important image features.

*(Replace XX with your actual accuracy after training.)*

---

# 🚀 Streamlit Application

The project includes an interactive web application.

## Features:

✅ Upload waste image  
✅ Display uploaded image  
✅ Predict waste category  
✅ Show confidence score  
✅ Display prediction history  
✅ Show model performance metrics  


Run application:

```bash
streamlit run app.py
```

---

# 📁 Project Structure

```
Waste-Classification-DeepLearning/

│
├── data/
│   └── archive/
│
├── notebooks/
│   └── training.ipynb
│
├── models/
│   ├── ann_model.h5
│   └── cnn_model.h5
│
├── screenshots/
│
├── reports/
│
├── utils/
│
├── app.py
│
├── requirements.txt
│
└── README.md

```

---

# 💻 Installation & Setup


## Clone Repository

```bash
git clone https://github.com/yourusername/Waste-Classification-DeepLearning.git
```

Move into project folder:

```bash
cd Waste-Classification-DeepLearning
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Notebook

Open:

```
notebooks/training.ipynb
```

Run all cells to train models.

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 📸 Application Screenshots

Add screenshots here:

```
screenshots/

├── upload.png
├── prediction.png
├── metrics.png

```

---

# 📚 Learning Outcomes

After completing this project, the following skills were developed:

- Deep Learning fundamentals
- Neural Network development
- CNN architecture design
- Image preprocessing
- Data augmentation
- Model optimization
- Performance evaluation
- AI application deployment


---

# 🔮 Future Improvements

Possible improvements:

- Implement Transfer Learning using:
    - MobileNet
    - ResNet
    - EfficientNet

- Deploy online using:
    - Streamlit Cloud
    - AWS
    - Google Cloud

- Add webcam-based waste detection

- Implement Grad-CAM visualization for model explainability

- Improve accuracy using larger datasets


---

# 👨‍💻 Author

**Your Name**

Deep Learning Intern

---

# ⭐ Acknowledgements

Thanks to:

- TensorFlow Community
- Kaggle Dataset Contributors
- Open Source Deep Learning Community

for providing resources that helped in building this project.
