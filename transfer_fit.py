import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

transfer_data = pd.read_csv('data/transfer.csv')
y = transfer_data['transfer_log2'].values
y = np.array(y)
x = transfer_data.iloc[:, 4:9].values
print(x)
x = np.array(x)

scaler = StandardScaler()
scaler.fit(x)
joblib.dump(scaler, 'data/scaler.joblib')
x = scaler.transform(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(r2_score(y_test, y_pred))
print("回归方程的截距（intercept）:", model.intercept_)
print("回归方程的系数（coefficients）:", model.coef_)

equation = f"y = {model.intercept_:.2f}"
for i, coef in enumerate(model.coef_):
    equation += f" + ({coef:.2f} * x_{i})"

print("线性回归方程:", equation)

df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
df.to_csv('data/transfer_fit_s.csv', index=False)

