import pandas as pd
from xgboost import XGBClassifier

# Load training data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
train = pd.read_csv(train_url)

# Drop non-predictive columns
train = train.drop(columns=["id", "DateTime"])

# Split features and target
X = train.drop(columns=["meal"])
y = train["meal"]

# Define model (Boosted Trees)
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

# Fit model
modelFit = model.fit(X, y)

# Load test data
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
test = pd.read_csv(test_url)

# Drop same columns used in training
test = test.drop(columns=["id", "DateTime"])

# Generate predictions
pred = modelFit.predict(test)

# Ensure predictions are integers (0 or 1)
pred = pred.astype(int)