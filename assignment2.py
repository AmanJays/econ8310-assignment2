import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load training data (LOCAL FILE)
train = pd.read_csv("assignment2train.csv")

# Feature engineering
train["DateTime"] = pd.to_datetime(train["DateTime"])
train["hour"] = train["DateTime"].dt.hour
train["day"] = train["DateTime"].dt.dayofweek

# Drop unused columns
train = train.drop(columns=["id","DateTime"])

# Features and target
X = train.drop(columns=["meal"])
y = train["meal"]

# Model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=1)

# Fit model
modelFit = model.fit(X, y)

# Load test data (LOCAL FILE)
test = pd.read_csv("assignment2test.csv")

# Same feature engineering
test["DateTime"] = pd.to_datetime(test["DateTime"])
test["hour"] = test["DateTime"].dt.hour
test["day"] = test["DateTime"].dt.dayofweek

test = test.drop(columns=["id","DateTime"])

# Predictions
pred = modelFit.predict(test)

# Convert to list of integers
pred = [int(x) for x in pred]