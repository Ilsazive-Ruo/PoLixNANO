import pandas as pd
import numpy as np
import joblib
import os
import shutil
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import MultiTaskLasso, Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score

experiment_name = 'Multi-XGB'
print(experiment_name)
if os.path.exists('model7/' + experiment_name):
    shutil.rmtree('model7/' + experiment_name)
os.mkdir('model7/' + experiment_name)

structure_props = {}
SP_data = pd.read_csv('data7/structures_prop.csv')
print(SP_data)
for Num in SP_data['Num'].values:
    structure_props[Num] = SP_data.iloc[Num, 4:].values.tolist()
print(structure_props)
ds = pd.read_csv('data7/20241027.csv')

x = []
for i in ds.index:
    temp = ([ds.iloc[i, 7], ds.iloc[i, 12], ds.iloc[i, 13], ds.iloc[i, 14]]
            + structure_props[ds.iloc[i, 8]] + [ds.iloc[i, 9]]
            + structure_props[ds.iloc[i, 10]] + [ds.iloc[i, 11]])
    x.append(temp)

y = ds.iloc[:, 1:6].values
# y = ds.iloc[:, 11].values
y = np.array(y)
print(y.shape)
x = np.array(x)
print(x.shape)
k = 0

skf = KFold(n_splits=5, shuffle=True, random_state=13) #13
r2 = {}
FI = {}
temp2 = []
score = 0
for train_index, test_index in skf.split(y, x):
    k += 1
    print('k=', k)
    # model = RandomForestRegressor(n_estimators=500, random_state=0)
    # model = MultiTaskLasso(random_state=42)
    # model = KNeighborsRegressor()
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    # model = LinearRegression()
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(x_train, y_train)
    joblib.dump(model, 'model7/' + experiment_name + '/' + experiment_name + '_' + str(k) + '.pkl')
    temp = []
    predict = model.predict(x_test)
    score += r2_score(y_test, predict)

    temp2.append(r2_score(y_test[:, 3], predict[:, 3]))
    for i in range(len(y_test[0, :])):

        temp.append(r2_score(y_test[:, i], predict[:, i]))
    FI['k=' + str(k)] = list(model.feature_importances_)
    r2['k=' + str(k)] = temp


print('AVE r2:', score/5)
print('AVE LE:', sum(temp2)/5)
df = pd.DataFrame(r2)
df2 = pd.DataFrame(FI)
df2.to_csv('model7/' + experiment_name + '/' + experiment_name + '_FI.csv', index=False)
df.to_csv('model7/' + experiment_name + '/' + experiment_name + '_model.csv', index=False)
