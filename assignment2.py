import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load training data
train = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv")

# Convert datetime
train["DateTime"] = pd.to_datetime(train["DateTime"])
train["hour"] = train["DateTime"].dt.hour
train["day"] = train["DateTime"].dt.dayofweek

# Drop unused columns
train = train.drop(columns=["id", "DateTime"])

# Split features and target
X = train.drop(columns=["meal"])
y = train["meal"]

# Define model
model = RandomForestClassifier(n_estimators=200, random_state=42)

# Fit model
modelFit = model.fit(X, y)

# Load test data
test = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv")

# Same feature engineering
test["DateTime"] = pd.to_datetime(test["DateTime"])
test["hour"] = test["DateTime"].dt.hour
test["day"] = test["DateTime"].dt.dayofweek

test = test.drop(columns=["id", "DateTime"])

# Generate predictions
pred = modelFit.predict(test)

# Convert to list of integers
pred = [int(x) for x in pred]