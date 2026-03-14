import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load training data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
train = pd.read_csv(train_url)

# Convert DateTime
train["DateTime"] = pd.to_datetime(train["DateTime"])
train["hour"] = train["DateTime"].dt.hour
train["dayofweek"] = train["DateTime"].dt.dayofweek

# Drop unused columns
train = train.drop(columns=["id","DateTime"])

# Split X and y
X = train.drop(columns=["meal"])
y = train["meal"]

# Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

# Fit model
modelFit = model.fit(X, y)

# Load test data
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
test = pd.read_csv(test_url)

# Same feature engineering
test["DateTime"] = pd.to_datetime(test["DateTime"])
test["hour"] = test["DateTime"].dt.hour
test["dayofweek"] = test["DateTime"].dt.dayofweek

test = test.drop(columns=["id","DateTime"])

# Predictions
pred = modelFit.predict(test)

# Convert dtype to float (important for grader)
pred = pred.astype(float)