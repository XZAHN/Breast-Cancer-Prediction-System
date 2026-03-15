import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import os
from sklearn.pipeline import Pipeline

# 1. Load Dataset
print("Loading dataset...")
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target # 0: Malignant, 1: Benign

# 2. Feature Selection
# Correct sklearn feature names:
# 'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'
selected_features = [
    'mean radius', 
    'mean texture', 
    'mean perimeter', 
    'mean area', 
    'mean smoothness'
]
print(f"Selected features: {selected_features}")

X = df[selected_features]
y = df['diagnosis']

# 3. Data Preprocessing
if X.isnull().sum().sum() > 0:
    print("Found missing values, filling with mean...")
    X = X.fillna(X.mean())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Implementation & Saving (Pipeline)
print("Training Logistic Regression model...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

pipeline.fit(X_train, y_train)

# 5. Evaluation
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='binary')
rec = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# 6. Save Model
model_filename = 'model/breast_cancer_model.pkl'
print(f"Saving model to {model_filename}...")
joblib.dump(pipeline, model_filename)

# 7. Demonstrate Loading
loaded_model = joblib.load(model_filename)
sample_pred = loaded_model.predict(X_test.iloc[:5])
print(f"Predictions from loaded model: {sample_pred}")
print("Model saved and verified successfully.")
