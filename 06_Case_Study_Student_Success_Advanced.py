
# Case Study: Predicting Student Success Using Multiple Algorithms
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Simulated realistic data
data = pd.DataFrame({
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'SleepHours': [8, 7, 6, 6, 5, 6, 7, 5, 4, 3],
    'Attendance': [60, 70, 75, 80, 85, 88, 90, 95, 97, 99],
    'Assignments': [1, 2, 2, 3, 3, 3, 4, 5, 5, 5],
    'Passed': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
})

X = data.drop(columns=['Passed'])
y = data['Passed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Models
lr = LogisticRegression().fit(X_train, y_train)
dt = DecisionTreeClassifier().fit(X_train, y_train)
rf = RandomForestClassifier().fit(X_train, y_train)

# Accuracy Comparison
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr.predict(X_test)))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt.predict(X_test)))
print("Random Forest Accuracy:", accuracy_score(y_test, rf.predict(X_test)))
