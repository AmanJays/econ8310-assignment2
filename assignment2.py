import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

def prepare_features(df):
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["hour"] = df["DateTime"].dt.hour
    df["day"] = df["DateTime"].dt.dayofweek

    df["is_lunch"] = df["hour"].between(11, 14).astype(int)
    df["is_dinner"] = df["hour"].between(17, 20).astype(int)

    cols_to_remove = ["id", "DateTime", "meal"]
    existing = [c for c in cols_to_remove if c in df.columns]

    return df.drop(columns=existing)

# Split
y_train = train["meal"]
X_train = prepare_features(train)
X_test = prepare_features(test)

# Extra safety
if "meal" in X_test.columns:
    X_test = X_test.drop(columns=["meal"])

# Align columns
X_test = X_test[X_train.columns]

# Model
model = RandomForestClassifier(
    n_estimators=120,
    max_depth=12,
    random_state=42,
    class_weight="balanced"
)

modelFit = model.fit(X_train, y_train)

pred = modelFit.predict(X_test).astype(int).tolist()