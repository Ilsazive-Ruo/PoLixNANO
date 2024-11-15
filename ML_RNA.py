import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


structure_props = {}
SP_data = pd.read_csv('data7/structures_prop.csv')
for Num in SP_data['Num'].values:
    structure_props[Num] = SP_data.iloc[Num, 4:].values.tolist()
ds = pd.read_csv('data7/20241027.csv')

y = ds.iloc[:, 5].values
y = np.array(y)
x = []
for i in ds.index:
    temp = ([ds.iloc[i, 7], ds.iloc[i, 12], ds.iloc[i, 13], ds.iloc[i, 14]]
            + structure_props[ds.iloc[i, 8]] + [ds.iloc[i, 9]]
            + structure_props[ds.iloc[i, 10]] + [ds.iloc[i, 11]])
    x.append(temp)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
# model = RandomForestRegressor(n_estimators=500, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# for i in range(5):
#     print(r2_score(y_test[:, i], y_pred[:, i]))
print(r2_score(y_test, y_pred))
print(y_pred)

joblib.dump(model, 'data7/DC_XGB.pkl')
df_p = pd.DataFrame(y_pred, columns=['y_pred'])
df_t = pd.DataFrame(y_test, columns=['y_true'])
df = pd.concat([df_p, df_t], axis=1)
df.to_csv('data7/DC_XGB_feature_fit.csv', index=False)
