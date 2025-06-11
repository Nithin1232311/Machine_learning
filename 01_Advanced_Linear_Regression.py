
# Advanced Linear Regression: Predicting House Prices
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (sample)
data = pd.DataFrame({
    'Size(sqft)': [1500, 1600, 1700, 1800, 1900, 2000, 2100],
    'Bedrooms': [3, 3, 3, 4, 4, 4, 5],
    'Age': [10, 5, 15, 20, 10, 5, 2],
    'Price': [300000, 320000, 340000, 360000, 380000, 400000, 420000]
})

X = data[['Size(sqft)', 'Bedrooms', 'Age']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("Predicted Prices:", y_pred)
