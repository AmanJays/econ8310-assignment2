import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('assignment2train.csv')
X = train_data.drop(columns=['meal', 'id'], errors='ignore')
y = train_data['meal']
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = RandomForestClassifier(n_estimators=100, max_depth=10, max_features='sqrt', random_state=1, n_jobs=1)
modelFit = model.fit(X_train, y_train)

pred = [int(x) for x in modelFit.predict(X_test)]
print(pred[:20])

test_data = pd.read_csv('assignment2test.csv')
test_data = test_data.drop(columns=['id'], errors='ignore')
test_data = pd.get_dummies(test_data)
test_data = test_data.reindex(columns=X_train.columns, fill_value=0)

pred_test = [int(x) for x in modelFit.predict(test_data)]
print(pred_test[:20])