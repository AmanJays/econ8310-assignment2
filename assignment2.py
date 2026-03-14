# ---------------------------
# 1️⃣ Imports
# ---------------------------
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# ---------------------------
# 2️⃣ Load your data
# ---------------------------
data = pd.read_csv('assignment2train.csv') 

# ---------------------------
# 3️⃣ Separate features and target
# ---------------------------
target_col = 'Total'  
X = data.drop(columns=[target_col], errors='ignore')
y = data[target_col]

# ---------------------------
# 4️⃣ Drop high-cardinality IDs
# ---------------------------
high_card_cols = ['meal', 'id']
X = X.drop(columns=[col for col in high_card_cols if col in X.columns], errors='ignore')

# ---------------------------
# 5️⃣ Convert categorical variables
# ---------------------------
X = pd.get_dummies(X)

# ---------------------------
# 6️⃣ Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# ---------------------------
# 7️⃣ Initialize model
# ---------------------------
model = RandomForestRegressor(
    n_estimators=10,
    max_depth=5,
    random_state=1,
    n_jobs=1
)

# ---------------------------
# 8️⃣ Fit model
# ---------------------------
modelFit = model.fit(X_train, y_train)

# ---------------------------
# 9️⃣ Predict
# ---------------------------
pred = modelFit.predict(X_test)

# ---------------------------
# 🔟 Evaluate
# ---------------------------
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
print(f"Test RMSE: {rmse:.2f}")

# ---------------------------
# 11️⃣ Predict on new/test set
# ---------------------------
test_data = pd.read_csv('assignment2test.csv')
test_data = test_data.drop(columns=[col for col in high_card_cols if col in test_data.columns], errors='ignore')
test_data = pd.get_dummies(test_data)
test_data = test_data.reindex(columns=X_train.columns, fill_value=0)

pred_test = modelFit.predict(test_data)
pred_test = [float(x) for x in pred_test]  
print(pred_test[:20])
