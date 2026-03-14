import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the data
train = pd.read_csv("assignment2train.csv")
test = pd.read_csv("assignment2test.csv")

# Convert DateTime to datetime type
train['DateTime'] = pd.to_datetime(train['DateTime'])
test['DateTime'] = pd.to_datetime(test['DateTime'])

# Feature engineering: extract hour and day of week
train['hour'] = train['DateTime'].dt.hour
train['day_of_week'] = train['DateTime'].dt.dayofweek

test['hour'] = test['DateTime'].dt.hour
test['day_of_week'] = test['DateTime'].dt.dayofweek

# Drop columns not used in training
drop_cols = ['id', 'DateTime', 'meal']  # 'meal' is the target
X_train = train.drop(columns=drop_cols)
y_train = train['meal']

X_test = test.drop(columns=['id', 'DateTime'])  # we don’t have 'meal' here

# Align columns in case train and test have mismatch
X_test = X_test[X_train.columns]

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
modelFit = model.fit(X_train, y_train)

# Make predictions
pred = modelFit.predict(X_test)

# Ensure predictions are integers
pred = pred.astype(int)

# Save predictions
submission = pd.DataFrame({'id': test['id'], 'meal': pred})
submission.to_csv("assignment2_predictions.csv", index=False)
print("Predictions saved to assignment2_predictions.csv")