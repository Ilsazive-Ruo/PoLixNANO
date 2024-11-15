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

experiment_name = 'Single_XGB'
print(experiment_name)
if os.path.exists('model/' + experiment_name):
    shutil.rmtree('model/' + experiment_name)
os.mkdir('model/' + experiment_name)

structure_props = {}
SP_data = pd.read_csv('data/structures_prop.csv')
for Num in SP_data['Num'].values:
    structure_props[Num] = SP_data.iloc[Num, 4:].values.tolist()
ds = pd.read_csv('data/20241027.csv')

x = []
for i in ds.index:
    temp = ([ds.iloc[i, 7], ds.iloc[i, 12], ds.iloc[i, 13], ds.iloc[i, 14]]
            + structure_props[ds.iloc[i, 8]] + [ds.iloc[i, 9]]
            + structure_props[ds.iloc[i, 10]] + [ds.iloc[i, 11]])
    x.append(temp)

r2 = {}
for feat in ds.columns[1:6]:
    y = ds[feat].values
    # y = ds.iloc[:, 11].values
    y = np.array(y)
    print(y.shape)
    x = np.array(x)
    print(x.shape)
    k = 0

    skf = KFold(n_splits=5, shuffle=True, random_state=1)  # 6, 7

    FI = {}
    temp = []
    score = 0
    for train_index, test_index in skf.split(y, x):
        k += 1
        model = xgb.XGBRegressor()
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        joblib.dump(model, 'weight/' + experiment_name + '/' + experiment_name + '_' + feat + str(k) + '.pkl')

        predict = model.predict(x_test)
        score += r2_score(y_test, predict)
        print(r2_score(y_test, predict))

        FI['k=' + str(k)] = list(model.feature_importances_)
        temp.append(r2_score(y_test, predict))
    r2[feat] = temp

    df2 = pd.DataFrame(FI)
    df2.to_csv('weight/' + experiment_name + '/' + experiment_name + feat + '_FI.csv', index=False)


df = pd.DataFrame(r2)
df.to_csv('weight/' + experiment_name + '/' + experiment_name + '_model.csv', index=False)
