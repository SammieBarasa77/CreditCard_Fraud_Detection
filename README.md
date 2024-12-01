# Machine Learning Project: Credit Card Fraud Detection
![Logo](https://github.com/SammieBarasa77/walmart_sales/blob/main/assets/images/cover_final.png)

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

![Dtaset](https://github.com/SammieBarasa77/CreditCard_Fraud_Detection/blob/main/assets/images/dataset.png)

## Checking Data Types and Missing Values  
Use `df.info()` to inspect data types and check for missing values.
```python
df.info()
```
![Info](https://github.com/SammieBarasa77/CreditCard_Fraud_Detection/blob/main/assets/images/info.png)

## Exploratory Data Analysis (EDA)  
- **Class Distribution**: Visualize the distribution of fraud (`Class`) to identify class imbalance.
```python
# Class Imbalance Check
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Class Distribution")
plt.show()
```
![Class Distribution](https://github.com/SammieBarasa77/CreditCard_Fraud_Detection/blob/main/assets/images/fraud_class_distn.png)
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
![Time-Amount Distribution](https://github.com/SammieBarasa77/CreditCard_Fraud_Detection/blob/main/assets/images/time_ammount_distn.png)

Correlation Analysis
```python
# Correlation Heatmap
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Matrix of Features")
plt.show()
```
![Heatmap](https://github.com/SammieBarasa77/CreditCard_Fraud_Detection/blob/main/assets/images/heatmap.png)

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
![Principal Conponents](https://github.com/SammieBarasa77/CreditCard_Fraud_Detection/blob/main/assets/images/distn_principal_components.png)

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
Area under the Precision-Recall Curve
```python
# Calculating AUPRC
auprc = auc(recall, precision)
print("Area Under Precision-Recall Curve (AUPRC):", auprc)
```
![AURPC](https://github.com/SammieBarasa77/CreditCard_Fraud_Detection/blob/main/assets/images/auprc.png)

## Model Development And Selection 
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

![Model Selection](https://github.com/SammieBarasa77/CreditCard_Fraud_Detection/blob/main/assets/images/model_selecttion_training.png)

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
![Confusion Matrix](https://github.com/SammieBarasa77/CreditCard_Fraud_Detection/blob/main/assets/images/confusion_matrix.png)

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

![Precision Recall Curve](https://github.com/SammieBarasa77/CreditCard_Fraud_Detection/blob/main/assets/images/precsion_recal_curve.png)

Hyperparameter tuning
```python
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
![Area Under Recall Curve](https://github.com/SammieBarasa77/CreditCard_Fraud_Detection/blob/main/assets/images/AUPRC.png)

Evaluating thge best Model
```python
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

importances = best_model.feature_importances_
feature_names = X_train.columns  # Adjust this based on your dataset
sorted_importances = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
print("Feature importances:", sorted_importances)

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)

```
## Findings Recommendations and Conclusions 

Findings

**Logistic Regression**

*Performance*

Achieved an F1-score of 0.95 for both Class 0 (non-fraud) and Class 1 (fraud), demonstrating balanced performance.

Precision (0.97) and Recall (0.92) for the fraud class indicate it is good at identifying fraud cases while minimizing false positives.

Strengths

Simpler and faster to train, making it a suitable option for environments where interpretability and computational efficiency are crucial.

Limitations

May not capture complex patterns in highly non-linear data, resulting in slightly reduced recall for the fraud class.

Random Forest

*Performance*

Achieved perfect scores (1.00) across all metrics, indicating an excellent ability to identify both fraud (Class 1) and non-fraud (Class 0) cases.

Significantly outperformed Logistic Regression on all metrics.

Strengths

Handles non-linear relationships and imbalanced data well, as demonstrated by the results.
High predictive accuracy due to its ensemble nature.

Limitations

Potential risk of overfitting, particularly if the dataset lacks diversity or sufficient variability.

Computationally expensive compared to Logistic Regression.

**Recommendations**

Random Forest for Deployment:

Random Forest is the preferred model for fraud detection due to its superior performance across all metrics.

It ensures high accuracy and minimizes the chances of both false negatives (undetected fraud) and false positives (incorrectly flagged legitimate transactions).

Conduct regular monitoring in production to check for overfitting or changes in data patterns (concept drift).

Further Assessment of Overfitting

Validate the Random Forest model on an unseen validation/test set or through techniques like cross-validation to confirm that the perfect scores are not due to overfitting.

If overfitting is observed, consider limiting the depth of trees (max_depth) or reducing the number of estimators (n_estimators) while maintaining performance.

Deploy Logistic Regression as a Baseline:

While not as precise as Random Forest, Logistic Regression offers a more generalizable alternative that is less prone to overfitting.

It could be deployed alongside Random Forest as a secondary model to cross-check predictions in resource-constrained or real-time settings.

Consider Cost-Effective Thresholding:

Fraud detection often involves a tradeoff between precision and recall. Adjust classification thresholds to prioritize fraud detection (high recall) or minimize false alarms (high precision) based on business requirements.

Ensemble or Hybrid Approach:

Combining the strengths of Logistic Regression (interpretability) and Random Forest (accuracy) in an ensemble approach could be explored to balance precision, recall, and computational efficiency.

Conclusions

*Logistic Regression*

Performed well with a balanced F1-score (0.95) and generalizable results, making it a reliable and interpretable baseline model.

While slightly less accurate in detecting fraud, it is suitable for environments where speed and simplicity are essential.

*Random Forest*

Delivered exceptional performance, achieving perfect metrics across the board, making it the most suitable model for fraud detection.

However, perfect metrics warrant caution, as overfitting could result in suboptimal performance on unseen or real-world data.

Deploy Random Forest as the primary model for fraud detection, with Logistic Regression as a backup or baseline model.

Ensure proper monitoring and retraining as the dataset evolves to maintain high accuracy and prevent model degradation.

