import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load training data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
train = pd.read_csv(train_url)

# Drop non-predictive columns
train = train.drop(columns=["id", "DateTime"])

# Features and target
X = train.drop(columns=["meal"])
y = train["meal"]

# Create model
model = RandomForestClassifier(n_estimators=200, random_state=42)

# Fit model
modelFit = model.fit(X, y)

# Load test data
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
test = pd.read_csv(test_url)

# Drop same columns
test = test.drop(columns=["id", "DateTime"])

# Make predictions
pred = modelFit.predict(test)

# Ensure predictions are integers
pred = pred.astype(int)