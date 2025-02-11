import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ridge_regression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

transfer_data = pd.read_csv('data/transfer.csv')
y = transfer_data['transfer_log2'].values
y = np.array(y)
x = transfer_data.iloc[:, 4:9].values
x = np.array(x)

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
print(x.shape)

skf = KFold(n_splits=10, shuffle=True, random_state=0)
k = 0
r2 = []
importance = {}
feature_names = transfer_data.columns.tolist()
importance['name'] = feature_names[4:9] + ['intercept']
for train_index, test_index in skf.split(y, x):
    k += 1
    print('k=', k)
    model = LinearRegression()
    # model = RandomForestRegressor(n_estimators=10, random_state=1)
    x_train, x_test = x[train_index], x[test_index]
    print(test_index)
    y_train, y_test = y[train_index], y[test_index]
    model.fit(x_train, y_train)
    r2.append(r2_score(y_test, model.predict(x_test)))
    print(r2_score(y_test, model.predict(x_test)))
    # print(model.feature_importances_)
    print("intercept:", model.intercept_)
    print("coefficients:", model.coef_)
    importance[str(k)] = list(model.coef_)
    importance[str(k)].append(model.intercept_)
    print(type(model.intercept_))

    equation = f"y = {model.intercept_:.2f}"
    for i, coef in enumerate(model.coef_):
        equation += f" + ({coef:.2f} * x_{i})"

    print("Equation:", equation)

print('r2_ave:', sum(r2) / len(r2))
df = pd.DataFrame(importance)
df.to_csv('data/importance_to_transfer.csv', index=False)
