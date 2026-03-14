# ---------------------------
# 1️⃣ Imports
# ---------------------------
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------
# 2️⃣ Load data
# ---------------------------
data = pd.read_csv('assignment2train.csv')

# ---------------------------
# 3️⃣ Features and target
# ---------------------------
target_col = 'Total'
X = data.drop(columns=[target_col, 'id', 'meal'], errors='ignore')
y = data[target_col]

# ---------------------------
# 4️⃣ Convert categorical features
# ---------------------------
X = pd.get_dummies(X)

# ---------------------------
# 5️⃣ Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# ---------------------------
# 6️⃣ Initialize and fit model
# ---------------------------
model = RandomForestRegressor(random_state=1)
modelFit = model.fit(X_train, y_train)

# ---------------------------
# 7️⃣ Predict on test
# ---------------------------
pred = modelFit.predict(X_test)

# ---------------------------
# 8️⃣ Evaluate
# ---------------------------
rmse = mean_squared_error(y_test, pred, squared=False)
print(f"Test RMSE: {rmse:.2f}")

# ---------------------------
# 9️⃣ Predict on new test set
# ---------------------------
test_data = pd.read_csv('assignment2test.csv')
test_data = test_data.drop(columns=['id','meal'], errors='ignore')
test_data = pd.get_dummies(test_data)
test_data = test_data.reindex(columns=X_train.columns, fill_value=0)

pred_test = modelFit.predict(test_data)