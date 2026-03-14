import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load training and test data from URLs
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
test_url  = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

# Convert DateTime to datetime
train['DateTime'] = pd.to_datetime(train['DateTime'])
test['DateTime'] = pd.to_datetime(test['DateTime'])

# Feature engineering: extract hour and day of week
train['hour'] = train['DateTime'].dt.hour
train['day_of_week'] = train['DateTime'].dt.dayofweek
test['hour'] = test['DateTime'].dt.hour
test['day_of_week'] = test['DateTime'].dt.dayofweek

# Prepare training and target
drop_cols = ['id', 'DateTime', 'meal']
X_train = train.drop(columns=drop_cols)
y_train = train['meal']

# Prepare test data
X_test = test.drop(columns=['id', 'DateTime'])
X_test = X_test[X_train.columns]  # align columns exactly

# Define model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Fit model
modelFit = model.fit(X_train, y_train)

# Make predictions (integers 0 or 1)
pred = modelFit.predict(X_test).astype(int)
print(pred)