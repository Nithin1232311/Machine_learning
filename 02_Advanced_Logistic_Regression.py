
# Advanced Logistic Regression: Titanic Survival Prediction
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

df = sns.load_dataset('titanic').dropna(subset=['age', 'fare', 'sex', 'class', 'survived'])
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['class'] = df['class'].map({'First': 1, 'Second': 2, 'Third': 3})

X = df[['age', 'fare', 'sex', 'class']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))
