# ================================
# CREDIT SCORING MODEL (COMPLETE)
# ================================

# 🔹 Step 1: Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 🔹 Step 2: Create Dataset
data = {
    'income': [50000, 60000, 25000, 80000, 30000, 120000, 45000, 70000, 20000, 90000],
    'debt': [10000, 15000, 20000, 10000, 25000, 20000, 15000, 12000, 18000, 16000],
    'payment_history': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # 1 = good, 0 = bad
    'target': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1 = creditworthy, 0 = risky
}

df = pd.DataFrame(data)

print("📊 Dataset:\n", df)

# 🔹 Step 3: Feature Engineering
# Create new feature: debt-to-income ratio
df['debt_ratio'] = df['debt'] / df['income']

print("\n📊 After Feature Engineering:\n", df)

# 🔹 Step 4: Define Features and Target
X = df[['income', 'debt', 'payment_history', 'debt_ratio']]
y = df['target']

# 🔹 Step 5: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Step 6: Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 🔹 Step 7: Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 🔹 Step 8: Evaluation Metrics
print("\n📈 Model Evaluation:")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))

# 🔹 Step 9: User Input Prediction
print("\n🔍 Test New Customer")

income = float(input("Enter Income: "))
debt = float(input("Enter Debt: "))
history = int(input("Enter Payment History (1=Good, 0=Bad): "))

# Feature engineering for input
debt_ratio = debt / income

new_data = np.array([[income, debt, history, debt_ratio]])

prediction = model.predict(new_data)

# 🔹 Step 10: Output Result
if prediction[0] == 1:
    print("\n✅ Creditworthy Customer (Loan Approved)")
else:
    print("\n❌ Risky Customer (Loan Rejected)")
