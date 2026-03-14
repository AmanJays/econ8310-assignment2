import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load data
train = pd.read_csv("assignment2train.csv")
test = pd.read_csv("assignment2test.csv")

# Convert DateTime to datetime
train['DateTime'] = pd.to_datetime(train['DateTime'])
test['DateTime'] = pd.to_datetime(test['DateTime'])

# Feature engineering
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

# Make predictions
pred = modelFit.predict(X_test).astype(int)

# Save predictions if needed
submission = pd.DataFrame({'id': test['id'], 'meal': pred})
submission.to_csv("assignment2_predictions.csv", index=False)
print("Predictions saved to assignment2_predictions.csv")