
# Advanced Random Forest: Wine Classification
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Report:\n", classification_report(y_test, predictions))
