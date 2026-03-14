# ---------------------------
# 1️⃣ Imports
# ---------------------------
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
# 4️⃣ Drop high-cardinality IDs (if any)
# ---------------------------
high_card_cols = ['meal', 'id']  # columns that are unique per row
X = X.drop(columns=[col for col in high_card_cols if col in X.columns], errors='ignore')

# ---------------------------
# 5️⃣ Convert categorical variables
# ---------------------------
X = pd.get_dummies(X)  # automatically converts all categorical columns

# ---------------------------
# 6️⃣ Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# ---------------------------
# 7️⃣ Initialize model (optimized)
# ---------------------------
model = RandomForestRegressor(
    n_estimators=10,       # fewer trees = faster
    max_depth=10,          # limit depth to prevent huge trees
    max_features='sqrt',   # speed up and reduce overfitting
    random_state=1,
    n_jobs=1               # single core avoids freezing
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
rmse = mse ** 0.5
print(f"Test RMSE: {rmse:.2f}")

# ---------------------------
# 11️⃣ Predict on new/test set
# ---------------------------
test_data = pd.read_csv('assignment2test.csv')  # <-- replace with your actual test file
test_data = test_data.drop(columns=[col for col in high_card_cols if col in test_data.columns], errors='ignore')
test_data = pd.get_dummies(test_data)
test_data = test_data.reindex(columns=X_train.columns, fill_value=0)  # align columns

pred_test = modelFit.predict(test_data)
print(pred_test[:20])  # print first 20 predictions for sanity check