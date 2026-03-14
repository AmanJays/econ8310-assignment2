# ---------------------------
# 1️⃣ Imports
# ---------------------------
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------
# 2️⃣ Load your data
# ---------------------------
data = pd.read_csv('assignment2train.csv')

# ---------------------------
# 3️⃣ Separate features and target
# ---------------------------
target_col = 'meal'
X = data.drop(columns=[target_col], errors='ignore')
y = data[target_col]

# ---------------------------
# 4️⃣ Drop high-cardinality IDs
# ---------------------------
high_card_cols = ['id']  # meal is target, not dropped
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
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',
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
pred = [int(x) for x in pred]  # convert to plain 0/1 integers

# ---------------------------
# 🔟 Print a small sample
# ---------------------------
print(pred[:20])

# ---------------------------
# 11️⃣ Predict on new/test set
# ---------------------------
test_data = pd.read_csv('assignment2test.csv')
test_data = test_data.drop(columns=[col for col in high_card_cols if col in test_data.columns], errors='ignore')
test_data = pd.get_dummies(test_data)
test_data = test_data.reindex(columns=X_train.columns, fill_value=0)

pred_test = modelFit.predict(test_data)
pred_test = [int(x) for x in pred_test]  # ensure 0/1 integers
print(pred_test[:20])