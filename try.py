import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib

# Load the dataset
file1_path = "Phishing_Legitimate_full.csv"
df1 = pd.read_csv(file1_path)

# Check for missing values
print(df1.isnull().sum())

# Define features and target variable
X1 = df1[['NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash', 'PathLength', 'PctExtHyperlinks', 'PctExtResourceUrlsRT', 'NumNumericChars', 'NoHttps']]
y1 = df1['CLASS_LABEL']

# Split the dataset into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X1_train, y1_train)

# Train XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X1_train, y1_train)

# Make predictions using both models
rf_predictions = rf_model.predict(X1_test)
xgb_predictions = xgb_model.predict(X1_test)

# Combine predictions (simple averaging)
ensemble_predictions = (rf_predictions + xgb_predictions) / 2

# Evaluate the combined predictions
ensemble_accuracy = (ensemble_predictions == y1_test).mean()
print("Ensemble Accuracy:", ensemble_accuracy)
