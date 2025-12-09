# -------------------------------
# Homework: Linear Regression (Simple & Multiple)
# -------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm  # For VIF calculation

# -------------------------------
# Load the dataset
# -------------------------------
file_path = r"C:\Users\Zinou_Tech\Desktop\padoc\AI\les cours de l'ENSTA\khelili mohamed\homework.csv"
df = pd.read_csv(file_path)

# Convert 'yes'/'no' to 1/0
for col in df.columns:
    if df[col].dtype == 'object':
        if set(df[col].unique()) == {'yes', 'no'}:
            df[col] = df[col].map({'yes': 1, 'no': 0})

# Remove any remaining non-numeric columns
df = df.select_dtypes(include=['number'])

# Ensure 'price' and 'lotsize' are numeric
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['lotsize'] = pd.to_numeric(df['lotsize'], errors='coerce')

# Drop rows with NaN
df = df.dropna()

# Check that 'price' and 'lotsize' exist
if 'price' not in df.columns or 'lotsize' not in df.columns:
    raise ValueError("'price' or 'lotsize' columns are missing.")

print("Cleaned dataset - Shape:", df.shape)
print("Columns:", list(df.columns))

# -------------------------------
# Simple Linear Regression (lotsize -> price)
# -------------------------------
X = df[['lotsize']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train, y_train)
y_pred = model_simple.predict(X_test)

print("\n--- Simple Linear Regression ---")
print(f"Slope (Coefficient for lotsize): {model_simple.coef_[0]:.4f}")
print(f"Intercept: {model_simple.intercept_:.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# -------------------------------
# Plot: Scatter + Regression Line
# -------------------------------
# Sort X_test and corresponding y_pred for plotting
sorted_idx = X_test['lotsize'].argsort()
X_test_sorted = X_test.iloc[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', label='Actual Values')
plt.plot(X_test_sorted, y_pred_sorted, color='red', linewidth=2, label='Predicted Line')
plt.xlabel("Lotsize")
plt.ylabel("Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()

# Residuals plot for simple regression
residuals_s = y_test - y_pred
plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals_s, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residuals: Simple Linear Regression")
plt.grid(True)
plt.show()

# -------------------------------
# Multiple Linear Regression
# -------------------------------
# Select top 5 features most correlated with 'price'
corr = df.corr()['price'].abs().sort_values(ascending=False)
top_features = corr.index[1:6]  # Exclude 'price' itself

X_multi = df[top_features]
y_multi = df['price']

# VIF to check multicollinearity
X_multi_const = sm.add_constant(X_multi)
vif_data = pd.DataFrame()
vif_data['Feature'] = X_multi_const.columns
vif_data['VIF'] = [1 / (1 - sm.OLS(X_multi_const[col], X_multi_const.drop(col, axis=1)).fit().rsquared)
                   for col in X_multi_const.columns]
print("\nVIF for features:")
print(vif_data)

# Train/test split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)

print("\n--- Multiple Linear Regression ---")
coeff_df = pd.DataFrame({'Feature': top_features, 'Coefficient': model_multi.coef_})
print(coeff_df)
print(f"Intercept: {model_multi.intercept_:.4f}")
print(f"MSE: {mean_squared_error(y_test_m, y_pred_m):.4f}")
print(f"R²: {r2_score(y_test_m, y_pred_m):.4f}")

# Residual plot for multiple regression
residuals_m = y_test_m - y_pred_m
plt.figure(figsize=(8,6))
plt.scatter(y_pred_m, residuals_m, color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residuals: Multiple Linear Regression")
plt.grid(True)
plt.show()

# -------------------------------
# Model Comparison
# -------------------------------
print("\n--- Model Comparison ---")
print(f"Simple R²: {r2_score(y_test, y_pred):.4f} | MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"Multiple R²: {r2_score(y_test_m, y_pred_m):.4f} | MSE: {mean_squared_error(y_test_m, y_pred_m):.4f}")

if r2_score(y_test_m, y_pred_m) > r2_score(y_test, y_pred):
    print("Multiple regression performs better (higher R²).")
else:
    print("Simple regression is sufficient or better.")