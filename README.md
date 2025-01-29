# Breast_Cancer_Prediction
📌 Project Overview

This project aims to build a machine learning model for predicting breast cancer based on diagnostic features extracted from medical data. The model classifies tumors as malignant (cancerous) or benign (non-cancerous) using supervised learning techniques.

📂 Dataset

The dataset used in this project is the Breast Cancer Wisconsin Diagnostic Dataset (WBCD), available from the UCI Machine Learning Repository. It contains 30 numerical features derived from digitized images of fine needle aspirates (FNAs) of breast masses.

🔹 Features:

Mean, Standard Error, and Worst values for attributes such as:

Radius

Texture

Perimeter

Area

Smoothness

Compactness

Concavity

Concave points

Symmetry

Fractal dimension

Target Variable:

M: Malignant (cancerous)

B: Benign (non-cancerous)

🚀 Model Development

1️⃣ Data Preprocessing

Handling missing values (if any)

Feature scaling (Normalization or Standardization)

Splitting data into training and testing sets

2️⃣ Model Selection

Logistic Regression

Random Forest Classifier

Support Vector Machine (SVM)

k-Nearest Neighbors (k-NN)

3️⃣ Model Evaluation

Accuracy, Precision, Recall, F1-score

Confusion Matrix

ROC Curve & AUC Score

💻 Installation & Usage

🔧 Prerequisites

Ensure you have the following installed:

Python (>=3.7)

Jupyter Notebook / VS Code

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

📥 Installation

Clone the repository:

$ git clone https://github.com/your-username/breast_cancer_prediction.git
$ cd breast_cancer_prediction

Install dependencies:

$ pip install -r requirements.txt

▶️ Run the Model

Execute the Jupyter Notebook:

$ jupyter notebook breast_cancer_prediction.ipynb

📊 Results & Performance

Achieved an accuracy of 95%+ on test data

ROC-AUC score close to 1.0, indicating strong classification ability

Random Forest performed best among tested models

📜 License

This project is licensed under the MIT License.

🤝 Contributing

Feel free to fork this repository, create feature branches, and submit pull requests.

🔗 References

UCI Machine Learning Repository

