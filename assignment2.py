import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# URLs for training and test data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"

# Load data
train_data = pd.read_csv(train_url)
test_data = pd.read_csv(test_url)

# Optional: create simple time-based features if timestamp exists
for df in [train_data, test_data]:
    if 'timestamp' in df.columns:
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df.drop(columns=['timestamp'], inplace=True)

# Drop irrelevant columns if they exist
train_data = train_data.drop(columns=['id'], errors='ignore')
test_features = test_data.drop(columns=['id'], errors='ignore')

# Separate target
X_train = train_data.drop(columns=['meal'], errors='ignore')
y_train = train_data['meal']

# Convert categorical columns to dummies
X_train = pd.get_dummies(X_train)
test_features = pd.get_dummies(test_features)
test_features = test_features.reindex(columns=X_train.columns, fill_value=0)

# Fit RandomForest
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
modelFit = model.fit(X_train, y_train)

# Predict on test set (Python ints)
pred = [int(x) for x in modelFit.predict(test_features)]

# Output for sanity check
print("Number of predictions:", len(pred))
print("Sample predictions:", pred[:20])