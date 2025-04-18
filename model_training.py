import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv("data/real_estate_dataset.csv")

# Keep only 'area' and 'price' columns, and drop missing values
df_model = df[['area', 'price']].dropna()

# Features and target
X = df_model[['area']]
y = df_model['price']

# Split data into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("✅ Model trained successfully!")
print(f"📉 Mean Absolute Error (MAE): {mae}")
print(f"📉 Mean Squared Error (MSE): {mse}")

# Save the model to file
joblib.dump(model, "real_estate_model.pkl")
print("💾 Model saved as 'real_estate_model.pkl'")
