# Customer Churn Prediction System

## 📌 Problem Statement
Customer churn is a major business challenge in the telecom industry.  
This project formulates churn prediction as a supervised binary classification problem to identify at-risk customers and support retention strategies.

Dataset: Telco Customer Churn Dataset  
Records: 7,043 customers  
Features: 20 input features after preprocessing  

---

## 🛠 Tech Stack
- Python
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Pandas, NumPy
- Matplotlib, Seaborn

---

## 🔎 Data Preprocessing
- Removed non-informative `customerID`
- Handled inconsistencies in `TotalCharges` (11 blank entries converted to numeric)
- Label encoded categorical features
- Identified class imbalance in target variable
- Applied **SMOTE** to balance minority class (Churn = Yes)

---

## 📊 Exploratory Data Analysis
- Distribution analysis of numerical features (tenure, MonthlyCharges, TotalCharges)
- Boxplots for outlier detection
- Correlation heatmap for numerical features
- Countplots for categorical feature distribution

---

## 🤖 Model Development
Compared multiple classification models using **5-fold cross-validation**:

- Decision Tree
- Random Forest
- XGBoost

Best performing model:
**Random Forest Classifier**
- Cross-validation accuracy: ~0.84

---

## 📈 Model Evaluation (Test Set)
- Accuracy: **78%**
- Precision, Recall, F1-score evaluated using classification report
- Confusion matrix used for detailed error analysis

Churn Recall: ~0.59  
Class imbalance handled using SMOTE before training.

---

## 🚀 Inference Pipeline
- Serialized trained model using `pickle`
- Saved label encoders for consistent feature transformation
- Built predictive system that accepts structured input and outputs:
  - Churn / No Churn
  - Prediction probability

---

## 📂 Repository Structure
