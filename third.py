import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

data = pd.read_csv("C:/Users/Sowmya/OneDrive/RECOVERY/Desktop/codsoft_internship/third/advertising.csv")

print("Dataset Info:")
print(data.info())
print("\nFirst 5 Rows of Dataset:")
print(data.head())

print("\nMissing Values:")
print(data.isnull().sum())

data.dropna(inplace=True)

data.columns = ['TV', 'Radio', 'Newspaper', 'Sales']

X = data[['TV', 'Radio', 'Newspaper']] 
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2): {r2:.2f}")

joblib.dump(model, 'sales_prediction_model.pkl')
print("\nModel saved as 'sales_prediction_model.pkl'.")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()
