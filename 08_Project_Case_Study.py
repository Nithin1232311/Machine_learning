
# Project: Student Performance Prediction using ML Algorithms

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample Data
data = {
    "StudyHours": [1,2,3,4,5,6,7,8,9,10],
    "SleepHours": [8,7,6,5,4,6,7,5,4,3],
    "Attendance": [80,82,78,90,85,88,92,96,99,100],
    "Passed": [0,0,0,1,1,1,1,1,1,1]
}
df = pd.DataFrame(data)
X = df[["StudyHours", "SleepHours", "Attendance"]]
y = df["Passed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Models
lr = LogisticRegression().fit(X_train, y_train)
dt = DecisionTreeClassifier().fit(X_train, y_train)
rf = RandomForestClassifier().fit(X_train, y_train)

# Predictions
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr.predict(X_test)))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt.predict(X_test)))
print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))
