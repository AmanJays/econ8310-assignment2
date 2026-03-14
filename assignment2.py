import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load training data
train = pd.read_csv("assignment2train.csv")

# Convert DateTime to useful features
train["DateTime"] = pd.to_datetime(train["DateTime"])
train["hour"] = train["DateTime"].dt.hour
train["dayofweek"] = train["DateTime"].dt.dayofweek

# Drop columns not used for modeling
train = train.drop(columns=["id", "DateTime"])

# Features and target
X = train.drop(columns=["meal"])
y = train["meal"]

# Create model
model = RandomForestClassifier(n_estimators=200, random_state=42)

# Fit model
modelFit = model.fit(X, y)

# Load test data
test = pd.read_csv("assignment2test.csv")

# Apply same feature engineering
test["DateTime"] = pd.to_datetime(test["DateTime"])
test["hour"] = test["DateTime"].dt.hour
test["dayofweek"] = test["DateTime"].dt.dayofweek

test = test.drop(columns=["id", "DateTime"])

# Make predictions
pred = modelFit.predict(test)

# Convert predictions to list of integers
pred = [int(x) for x in pred]


