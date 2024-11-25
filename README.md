# Machine Learning Project: Credit Card Fraud Detection
![Logo](https://github.com/SammieBarasa77/employee_performance/blob/main/assets/images/Screenshot%202024-11-23%20222057.png)

## Table of Contents 
- [Introduction](#Introduction)  
  - [Importing Libraries](#importing-libraries)  
  - [Loading the Dataset](#loading-the-dataset)  
  - [Checking Data Types and Missing Values](#checking-data-types-and-missing-values)  
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
    - [Class Distribution](#class-distribution)  
    - [Feature Analysis](#feature-analysis)  
      - [Distribution of Time](#distribution-of-time)  
      - [Distribution of Amount](#distribution-of-amount)  
  - [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)  
    - [Handling Class Imbalance Using SMOTE](#handling-class-imbalance-using-smote)  
    - [Standardizing Features](#standardizing-features)  
  - [Splitting the Dataset](#splitting-the-dataset)  
  - [Model Development](#model-development)  
    - [Training a Random Forest Classifier](#training-a-random-forest-classifier)  
    - [Hyperparameter Tuning with GridSearchCV](#hyperparameter-tuning-with-gridsearchcv)  
  - [Model Evaluation](#model-evaluation)  
    - [Confusion Matrix](#confusion-matrix)  
    - [Precision, Recall, and F1-score](#precision-recall-and-f1-score)  
    - [Precision-Recall Curve and AUC Score](#precision-recall-curve-and-auc-score)  
  - [Results and Conclusions](#results-and-conclusions)  
    - [Model Performance](#model-performance)  
    - [Challenges and Improvements](#challenges-and-improvements)

## Introduction
Fraud detection is a pressing issue for many industries, especially in the financial sector, where digital transactions are increasingly prevalent. Traditional methods, such as manual reviews and rule-based systems, often fall short in addressing the sophisticated techniques employed by fraudsters. These methods are time-consuming, static, and struggle to keep pace with the constantly changing nature of fraudulent activities.

Machine learning has transformed the fight against fraud by offering intelligent, adaptable, and efficient solutions. These models analyze vast amounts of data to detect subtle patterns and anomalies that signal potential fraud. Unlike static systems, machine learning continuously evolves, improving its ability to identify new threats while reducing false alarms. By embracing machine learning, organizations can enhance security, protect customers, and build trust in a digital-first era.

## Importing Libraries  
Import essential libraries for data manipulation, visualization, and machine learning.
```python
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, f1_score
from sklearn.model_selection import GridSearchCV
```
## Loading the Dataset  
Load the `creditcard.csv` dataset into a Pandas DataFrame.
```python
# Loading the dataset 
df = pd.read_csv(r"C:\Users\samue\Downloads\creditcard.csv")
df
```
## Checking Data Types and Missing Values  
Use `df.info()` to inspect data types and check for missing values.
```python
df.info()
```
## Exploratory Data Analysis (EDA)  
- **Class Distribution**: Visualize the distribution of fraud (`Class`) to identify class imbalance.
```python
# Class Imbalance Check
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Class Distribution")
plt.show()
```
- **Feature Analysis**:  
  - Distribution of `Time`.  
  - Distribution of `Amount`.
```python
# Time and Amount Analysis
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
sns.histplot(df['Time'], bins=50, kde=True)
plt.title("Distribution of Time")

plt.subplot(1, 2, 2)
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title("Distribution of Amount")
plt.show()
```
Correlation Analysis
```python
# Correlation Heatmap
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Matrix of Features")
plt.show()
```
Distribution of Principal Components
```python
fig, axes = plt.subplots(7, 4, figsize=(20, 15))
for i, ax in enumerate(axes.flat):
    if i < 28:
        sns.histplot(df[f'V{i+1}'], bins=50, kde=True, ax=ax)
        ax.set_title(f'Distribution of V{i+1}')
plt.tight_layout()
plt.show()
```
## Data Cleaning and Preprocessing  
Standardize features like `Amount` using a `StandardScaler`.
```python
# Remove Duplicates
df = df.drop_duplicates()
print(f"Number of rows after removing duplicates: {len(df)}")

# Handle Nulls
# Check for null values
print(df.isnull().sum())
# Drop rows with nulls (if any) or you can use imputation
df = df.dropna()
print(f"Number of rows after handling nulls: {len(df)}")

#Standardize Features
# Standardize the 'Amount' column
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
print("Standardization of 'Amount' completed.")
```
## Data Processing
```python
# Scaling 'Amount' and 'Time' columns
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])
```
Separating features and target variable
```python
# Separating features and target variable
X = df.drop('Class', axis=1)
y = df['Class']
```
Handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
```python
# Addressing Class Imbalance with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
```
## Splitting the Dataset  
Split the data into training and testing sets using `train_test_split`.
```python
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
```
Feature Selection and Engineering
```python
# All features are PCA-transformed, so feature engineering is limited. Removing low-variance features
from sklearn.feature_selection import VarianceThreshold

# Removing low-variance features (if variance is below 0.1)
selector = VarianceThreshold(threshold=0.1)
X_train = selector.fit_transform(X_train)
X_test = selector.transform(X_test)
```
## Model Development Adn Selection 
- Train a Random Forest Classifier on the balanced and preprocessed dataset.  
- Tune hyperparameters using GridSearchCV for optimal performance.
```python
# Initializing models
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42)
}

# Training and evaluating models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))
```
## Model Evaluation  
Evaluate the model using metrics such as:  
Confusion Matrix.
```python
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```
Random Forest
```python
# Evaluating Random Forest 
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

```
- Precision-Recall Curve and AUC score.
```python
# Precision-Recall Curve
y_scores = model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_scores)
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title("Precision-Recall Curve")
plt.show()
```

Hyperparameter tuning
```pyhon
# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                           param_grid, cv=3, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Print best parameters
print("Best parameters found:", grid_search.best_params_)
```
Interpreting  Insights for business stakeholders
```python
import shap

# SHAP for interpretability (use only with small sample for efficiency)
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test[:100])

# SHAP summary plot
shap.summary_plot(shap_values[1], X_test[:100], plot_type="bar", feature_names=X.columns)
```
Calculating AUPRC
```python
auprc = auc(recall, precision)
print("Area Under Precision-Recall Curve (AUPRC):", auprc)
```

## Results and Conclusions  

